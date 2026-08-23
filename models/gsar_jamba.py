"""
GSAR-Jamba-V2: Improved Global Semantic Anchor Resonance Jamba Architecture

核心改进:
1. 轻量帧级重要性预打分 (Frame Importance Pre-scoring)
2. 局部变化加权的鲁棒锚点聚合 (Robust Anchor Aggregation)
3. 局部时序动态感知的自适应Δ (Local Dynamic Adaptive Delta)
4. 锚点融合的动态窗口星状注意力 (Anchor-Fused Dynamic Window Attention)
5. 门控自适应残差融合 (Gated Adaptive Residual Fusion)
6. 多源全局特征门控聚合 (Multi-Source Gated Aggregation)

时间复杂度: 严格 O(N) 线性
"""

import math
import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FrameImportanceScorer(nn.Module):
    """
    改进点1: 轻量帧级重要性预打分
    
    学术动机:
    100帧均匀采样中，通常60%以上是无意义的过渡帧。
    提前给出帧级重要性先验，能引导后续模块"聚焦关键帧"。
    
    计算量增加不足0.1%。
    """
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.d_model = d_model
        self.pre_score_linear = nn.Linear(d_model * 2, 1, bias=False)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.pre_score_linear.weight)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) 输入帧特征
        
        Returns:
            x: (B, T, D) 原始特征（不变）
            importance_scores: (B, T) 帧级重要性分数 [0, 1]
        """
        B, T, D = x.shape
        
        delta_x = torch.zeros_like(x)
        delta_x[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        
        delta_x_abs = torch.abs(delta_x)
        
        concat_features = torch.cat([x, delta_x_abs], dim=-1)
        
        importance_logits = self.pre_score_linear(concat_features).squeeze(-1)
        importance_scores = torch.sigmoid(importance_logits)
        
        return x, importance_scores


class RobustAnchorAggregation(nn.Module):
    """
    改进点2: 局部变化加权的鲁棒锚点聚合
    
    学术动机:
    原方案的平均池化假设"所有帧贡献均等"，但实际上爆点帧、钩子帧
    对核心主题的贡献远大于过渡帧。结合预打分的加权聚合，能避免冗余帧的污染。
    """
    
    def __init__(
        self,
        d_model: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = 1e-8
        
        self.anchor_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        importance_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) 输入帧特征
            importance_scores: (B, T) 帧级重要性分数
        
        Returns:
            g_anchor: (B, D) 全局语义锚点
        """
        weights = importance_scores + self.eps
        weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)
        
        weighted_x = x * weights.unsqueeze(-1)
        weighted_pool = weighted_x.sum(dim=1)
        
        g_anchor = self.anchor_proj(weighted_pool)
        
        return g_anchor


class LocalDynamicAdaptiveSSM(nn.Module):
    """
    改进点3: 局部时序动态感知的自适应Δ
    
    核心学术创新点:
    摒弃原方案"全局锚点微弱偏置"的弱自适应，改为基于局部时序上下文的强自适应机制。
    
    SSM数学原理:
    h(t) = A^Δ_t * h(t-1) + B*Δ_t * x(t)
    
    - Δ_t ↑ → A^Δ_t ↓ (历史快速遗忘) → 聚焦当前帧细节
    - Δ_t ↓ → A^Δ_t ↑ (历史高度保留) → 依赖历史上下文
    """
    
    def __init__(
        self,
        d_model: int = 768,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 48,
        dropout: float = 0.1,
        delta_min: float = 0.01,
        delta_max: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.dt_rank = dt_rank
        
        self.delta_min = nn.Parameter(torch.tensor(delta_min))
        self.delta_max = nn.Parameter(torch.tensor(delta_max))
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        
        self.local_dynamic_proj1 = nn.Linear(d_model * 2 + 1, dt_rank)
        self.local_dynamic_proj2 = nn.Linear(dt_rank, 1)
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.local_dynamic_proj1.weight)
        nn.init.xavier_uniform_(self.local_dynamic_proj2.weight)
        nn.init.constant_(self.dt_proj.bias, 1.0)
    
    def forward(
        self,
        x: torch.Tensor,
        importance_scores: torch.Tensor,
        prev_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) 输入序列
            importance_scores: (B, T) 帧级重要性分数
            prev_states: 可选的前一Block的状态
        
        Returns:
            output: (B, T, D) 双向SSM输出
            final_state: (B, D_inner, N) 最终状态（用于多源聚合）
        """
        B, T, D = x.shape
        
        xz = rearrange(self.in_proj(x), 'b l (p n) -> b p l n', p=2)
        x_proj, z = xz[:, 0], xz[:, 1]
        
        x_conv = rearrange(x_proj, 'b l n -> b n l')
        x_conv = self.conv1d(x_conv)[:, :, :T]
        x_conv = rearrange(x_conv, 'b n l -> b l n')
        x_conv = F.silu(x_conv)
        
        delta_x = torch.zeros_like(x)
        delta_x[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        delta_x_abs = torch.abs(delta_x)
        
        local_context = torch.cat([
            delta_x,
            delta_x_abs,
            importance_scores.unsqueeze(-1),
        ], dim=-1)
        
        local_hidden = F.relu(self.local_dynamic_proj1(local_context))
        local_significance = torch.sigmoid(self.local_dynamic_proj2(local_hidden)).squeeze(-1)
        
        delta_min_val = torch.clamp(F.softplus(self.delta_min), min=0.001, max=0.5)
        delta_max_val = torch.clamp(F.softplus(self.delta_max), min=0.5, max=2.0)
        
        x_ssm = self.x_proj(x_conv)
        delta_base, B_ssm, C = torch.split(x_ssm, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        delta_adaptive = delta_min_val + (delta_max_val - delta_min_val) * local_significance.unsqueeze(-1)
        
        delta = self.dt_proj(delta_base + delta_adaptive)
        delta = F.softplus(delta)
        
        A = -torch.exp(self.A_log.float())
        
        y_forward, final_state_forward = self._selective_scan_with_state(
            x_conv, delta, A, B_ssm, C, self.D
        )
        
        y_backward, _ = self._selective_scan_with_state(
            x_conv.flip(1), delta.flip(1), A, B_ssm.flip(1), C.flip(1), self.D
        )
        y_backward = y_backward.flip(1)
        
        y = 0.75 * y_forward + 0.25 * y_backward
        
        y = y * F.silu(z)
        output = self.out_proj(y)
        output = self.dropout(output)
        
        return output, final_state_forward
    
    def _selective_scan_with_state(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        选择性扫描实现，返回最终状态
        """
        B_batch, L, D_inner = u.shape
        N = self.d_state
        
        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        
        h = torch.zeros(B_batch, D_inner, N, device=u.device, dtype=u.dtype)
        ys = torch.zeros(B_batch, L, D_inner, device=u.device, dtype=u.dtype)
        
        for i in range(L):
            h = deltaA[:, i] * h + deltaB_u[:, i]
            ys[:, i, :] = torch.matmul(h, C[:, i].unsqueeze(-1)).squeeze(-1)
        
        ys = ys + u * D.unsqueeze(0).unsqueeze(0)
        
        return ys, h


class AnchorFusedDynamicWindowAttention(nn.Module):
    """
    改进点4: 锚点融合的动态窗口星状注意力
    
    学术动机:
    1. 黑洞Token与全局锚点功能冗余，合并减少参数
    2. 局部窗口固定，无法适配不同时长/节奏的视频
    
    改进:
    1. 用全局锚点初始化黑洞Token
    2. 基于预打分动态调整局部窗口大小
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 12,
        base_window_size: int = 20,
        window_alpha: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.base_window_size = base_window_size
        self.window_alpha = window_alpha
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        g_anchor: torch.Tensor,
        importance_scores: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) 输入序列
            g_anchor: (B, D) 全局语义锚点
            importance_scores: (B, T) 帧级重要性分数
        
        Returns:
            output: (B, T, D) 输出序列
            global_token_out: (B, D) 更新后的全局Token
        """
        B, T, D = x.shape
        
        global_token = g_anchor.unsqueeze(1)
        x_with_global = torch.cat([global_token, x], dim=1)
        
        residual = x_with_global
        x_norm = self.norm(x_with_global)
        
        Q = self.q_proj(x_norm)
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)
        
        Q = rearrange(Q, 'b l (h d) -> b h l d', h=self.n_heads)
        K = rearrange(K, 'b l (h d) -> b h l d', h=self.n_heads)
        V = rearrange(V, 'b l (h d) -> b h l d', h=self.n_heads)
        
        output = torch.zeros_like(Q)
        
        global_q = Q[:, :, 0:1, :]
        global_attn = torch.matmul(global_q, K.transpose(-2, -1)) * self.scale
        global_attn = F.softmax(global_attn, dim=-1)
        global_attn = self.dropout(global_attn)
        global_out = torch.matmul(global_attn, V)
        output[:, :, 0:1, :] = global_out
        
        mean_importance = importance_scores.mean(dim=-1, keepdim=True)
        var_importance = ((importance_scores - mean_importance) ** 2).mean(dim=-1, keepdim=True)
        
        for i in range(1, T + 1):
            frame_idx = i - 1
            frame_importance = importance_scores[:, frame_idx:frame_idx+1]
            
            dynamic_window = int(self.base_window_size * (1 + self.window_alpha * (frame_importance - mean_importance).mean().item()))
            dynamic_window = max(1, min(dynamic_window, T))
            
            start = max(0, i - dynamic_window // 2)
            end = min(T + 1, i + dynamic_window // 2 + 1)
            
            local_indices = list(range(start, end))
            if 0 not in local_indices:
                local_indices = [0] + local_indices
            
            local_k = K[:, :, local_indices, :]
            local_v = V[:, :, local_indices, :]
            
            q_i = Q[:, :, i:i+1, :]
            local_attn = torch.matmul(q_i, local_k.transpose(-2, -1)) * self.scale
            local_attn = F.softmax(local_attn, dim=-1)
            local_attn = self.dropout(local_attn)
            
            local_out = torch.matmul(local_attn, local_v)
            output[:, :, i:i+1, :] = local_out
        
        output = rearrange(output, 'b h l d -> b l (h d)')
        output = self.out_proj(output)
        output = self.dropout(output)
        output = output + residual
        
        global_token_out = output[:, 0, :]
        output = output[:, 1:, :]
        
        return output, global_token_out


class GatedAdaptiveResidualFusion(nn.Module):
    """
    改进点5: 门控自适应残差融合
    
    学术动机:
    原方案的直接相加假设"时序/空间路径同等重要"，但实际上:
    - 剧情类、动作类视频更依赖时序路径
    - 风景类、静态展示类视频更依赖空间路径
    
    门控机制让模型自动学习路径权重。
    """
    
    def __init__(
        self,
        d_model: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        self.gate_proj = nn.Linear(d_model * 2 + 1, 1)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        y_temporal: torch.Tensor,
        y_spatial: torch.Tensor,
        g_anchor: torch.Tensor,
        importance_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) 原始输入
            y_temporal: (B, T, D) 时序路径输出
            y_spatial: (B, T, D) 空间路径输出
            g_anchor: (B, D) 全局锚点
            importance_scores: (B, T) 帧级重要性分数
        
        Returns:
            output: (B, T, D) 融合后的输出
        """
        B, T, D = x.shape
        
        g_anchor_expanded = g_anchor.unsqueeze(1).expand(-1, T, -1)
        mean_importance = importance_scores.mean(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, T, -1)
        
        gate_input = torch.cat([x, g_anchor_expanded, mean_importance], dim=-1)
        
        gate_temporal = torch.sigmoid(self.gate_proj(gate_input))
        gate_spatial = 1 - gate_temporal
        
        output = x + gate_temporal * y_temporal + gate_spatial * y_spatial
        output = self.dropout(output)
        
        return output


class MultiSourceGatedAggregation(nn.Module):
    """
    改进点6: 多源全局特征门控聚合
    
    融合三类全局信号源:
    1. F_pool: 自适应池化特征（复用预打分作为池化权重）
    2. F_blackhole: 最后一个Block输出的黑洞Token
    3. F_ssm: 最后一个Block输出的SSM最终状态
    """
    
    def __init__(
        self,
        d_model: int = 768,
        d_inner: int = 1536,
        d_state: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state
        
        self.ssm_state_proj = nn.Linear(d_inner, d_model)
        
        self.gate_proj1 = nn.Linear(d_model * 3, d_model)
        self.gate_proj2 = nn.Linear(d_model, 3)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.ssm_state_proj.weight)
        nn.init.xavier_uniform_(self.gate_proj1.weight)
        nn.init.xavier_uniform_(self.gate_proj2.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        importance_scores: torch.Tensor,
        global_token: torch.Tensor,
        ssm_final_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) 帧特征
            importance_scores: (B, T) 帧级重要性分数
            global_token: (B, D) 黑洞Token
            ssm_final_state: (B, D_inner, N) SSM最终状态
        
        Returns:
            F_final: (B, D) 最终全局特征
        """
        B, T, D = x.shape
        
        weights = importance_scores + 1e-8
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        F_pool = (x * weights.unsqueeze(-1)).sum(dim=1)
        
        F_blackhole = global_token
        
        # SSM消融: 如果ssm_final_state为None，使用零向量
        if ssm_final_state is not None:
            F_ssm_raw = ssm_final_state.mean(dim=-1)
            F_ssm = self.ssm_state_proj(F_ssm_raw)
        else:
            # 使用零向量替代SSM状态
            B = x.shape[0]
            F_ssm = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
        
        concat_features = torch.cat([F_pool, F_blackhole, F_ssm], dim=-1)
        
        gate_hidden = F.relu(self.gate_proj1(concat_features))
        gate_weights = F.softmax(self.gate_proj2(gate_hidden), dim=-1)
        
        F_final = (gate_weights[:, 0:1] * F_pool + 
                   gate_weights[:, 1:2] * F_blackhole + 
                   gate_weights[:, 2:3] * F_ssm)
        
        F_final = self.dropout(F_final)
        
        return F_final


class GSARJambaBlockV2(nn.Module):
    """
    GSAR-Jamba V2 Block: 完整的改进版单个Jamba块
    
    结构:
    1. FrameImportanceScorer: 帧级重要性预打分
    2. RobustAnchorAggregation: 鲁棒锚点聚合
    3. LocalDynamicAdaptiveSSM: 局部动态自适应SSM
    4. AnchorFusedDynamicWindowAttention: 锚点融合动态窗口注意力
    5. GatedAdaptiveResidualFusion: 门控自适应残差融合
    6. FFN: 前馈网络
    """
    
    def __init__(
        self,
        d_model: int = 768,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 12,
        base_window_size: int = 20,
        window_alpha: float = 0.5,
        dt_rank: int = 48,
        delta_min: float = 0.01,
        delta_max: float = 1.0,
        dropout: float = 0.1,
        use_sparse_attention: bool = True,
        use_ssm: bool = True,
    ):
        super().__init__()
        
        self.use_sparse_attention = use_sparse_attention
        self.use_ssm = use_ssm
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.ssm = LocalDynamicAdaptiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dropout=dropout,
            delta_min=delta_min,
            delta_max=delta_max,
        )
        
        self.attention = AnchorFusedDynamicWindowAttention(
            d_model=d_model,
            n_heads=n_heads,
            base_window_size=base_window_size,
            window_alpha=window_alpha,
            dropout=dropout,
        )
        
        self.fusion = GatedAdaptiveResidualFusion(
            d_model=d_model,
            dropout=dropout,
        )
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        importance_scores: torch.Tensor,
        g_anchor: torch.Tensor,
        return_global_token: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) 输入序列
            importance_scores: (B, T) 帧级重要性分数
            g_anchor: (B, D) 全局语义锚点
            return_global_token: 是否返回全局Token
        
        Returns:
            output: (B, T, D) 输出序列
            global_token: (B, D) 全局Token (可选)
            ssm_final_state: (B, D_inner, N) SSM最终状态
            g_anchor: (B, D) 全局语义锚点
        """
        residual = x
        x_norm = self.norm1(x)
        
        # SSM消融: 如果禁用SSM，使用零向量替代
        if self.use_ssm:
            y_temporal, ssm_final_state = self.ssm(x_norm, importance_scores)
        else:
            # 使用零向量替代SSM输出
            B, T, D = x_norm.shape
            y_temporal = torch.zeros_like(x_norm)
            ssm_final_state = None
        
        residual = x
        x_norm = self.norm2(x)
        
        # Sparse Attention消融: 如果禁用，跳过注意力计算
        if self.use_sparse_attention:
            y_spatial, global_token = self.attention(x_norm, g_anchor, importance_scores)
        else:
            # 使用零向量替代注意力输出
            B, T, D = x_norm.shape
            y_spatial = torch.zeros_like(x_norm)
            global_token = torch.zeros(B, D, device=x_norm.device, dtype=x_norm.dtype)
        
        x = self.fusion(residual, y_temporal, y_spatial, g_anchor, importance_scores)
        
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + residual
        
        if return_global_token:
            return x, global_token, ssm_final_state, g_anchor
        return x, None, ssm_final_state, g_anchor


class TemporalEncoding(nn.Module):
    """时间位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GSARJambaEncoderV2(nn.Module):
    """
    完整的GSAR-Jamba V2编码器
    
    支持2-3个Jamba块的堆叠
    """
    
    def __init__(
        self,
        d_input: int = 768,
        d_model: int = 768,
        n_blocks: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 12,
        base_window_size: int = 20,
        window_alpha: float = 0.5,
        dt_rank: int = 48,
        delta_min: float = 0.01,
        delta_max: float = 1.0,
        dropout: float = 0.1,
        max_frames: int = 100,
        use_frame_scoring: bool = True,
        use_sparse_attention: bool = True,
        use_ssm: bool = True,
    ):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.n_blocks = n_blocks
        self.max_frames = max_frames
        self.use_frame_scoring = use_frame_scoring
        self.use_sparse_attention = use_sparse_attention
        self.use_ssm = use_ssm
        
        self.input_proj = nn.Linear(d_input, d_model)
        
        self.temporal_encoding = TemporalEncoding(d_model, max_len=max_frames, dropout=dropout)
        
        self.frame_scorer = FrameImportanceScorer(d_model)
        
        self.anchor_aggregation = RobustAnchorAggregation(d_model, dropout)
        
        self.blocks = nn.ModuleList([
            GSARJambaBlockV2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                n_heads=n_heads,
                base_window_size=base_window_size,
                window_alpha=window_alpha,
                dt_rank=dt_rank,
                delta_min=delta_min,
                delta_max=delta_max,
                dropout=dropout,
                use_sparse_attention=use_sparse_attention,
                use_ssm=use_ssm,
            )
            for _ in range(n_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        self.multi_source_aggregation = MultiSourceGatedAggregation(
            d_model=d_model,
            d_inner=self.d_inner,
            d_state=d_state,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D_in) 输入序列, D_in 默认为 768
        
        Returns:
            global_feature: (B, D) 全局特征
        """
        x = self.input_proj(x)
        x = self.temporal_encoding(x)
        
        # Frame Scoring消融: 如果禁用，使用均匀权重
        if self.use_frame_scoring:
            x, importance_scores = self.frame_scorer(x)
        else:
            # 使用均匀权重
            B, T, D = x.shape
            importance_scores = torch.ones(B, T, device=x.device) / T
        
        g_anchor = self.anchor_aggregation(x, importance_scores)
        
        global_token = None
        ssm_final_state = None
        
        for i, block in enumerate(self.blocks):
            is_last = (i == len(self.blocks) - 1)
            x, gt, ssm_state, ga = block(
                x, importance_scores, g_anchor, return_global_token=is_last
            )
            if is_last:
                global_token = gt
                ssm_final_state = ssm_state
                g_anchor = ga
        
        x = self.final_norm(x)
        
        global_feature = self.multi_source_aggregation(
            x, importance_scores, global_token, ssm_final_state
        )
        
        return global_feature


def create_gsar_jamba_encoder_v2(
    num_frames: int = 100,
    d_input: int = 768,
    d_model: int = 768,
    n_blocks: int = 2,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    n_heads: int = 12,
    base_window_size: int = 20,
    window_alpha: float = 0.5,
    dt_rank: int = 48,
    delta_min: float = 0.01,
    delta_max: float = 1.0,
    dropout: float = 0.1,
    use_frame_scoring: bool = True,
    use_sparse_attention: bool = True,
    use_ssm: bool = True,
) -> GSARJambaEncoderV2:
    """创建GSAR-Jamba V2编码器的工厂函数"""
    return GSARJambaEncoderV2(
        d_input=d_input,
        d_model=d_model,
        n_blocks=n_blocks,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        n_heads=n_heads,
        base_window_size=base_window_size,
        window_alpha=window_alpha,
        dt_rank=dt_rank,
        delta_min=delta_min,
        delta_max=delta_max,
        dropout=dropout,
        max_frames=num_frames,
        use_frame_scoring=use_frame_scoring,
        use_sparse_attention=use_sparse_attention,
        use_ssm=use_ssm,
    )
