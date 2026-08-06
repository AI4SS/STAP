"""
多模态拓扑感知参数化记忆库
支持 visual + text + user 多模态特征融合查询
"""

import os
import pickle
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TopologyAwarePMBMultimodal(nn.Module):
    """
    多模态拓扑感知参数化记忆库
    
    支持:
    1. 加载预构建的多模态记忆库 (包含 PCA 参数)
    2. 将 visual + text + user 特征融合后查询
    3. 热度-主题联合寻址
    """
    
    def __init__(
        self,
        d_model: int = 768,
        m_heat: int = 6,
        n_topic: int = 32,
        top_k: int = 5,
        temperature: float = 1.0,
        learnable_temperature: bool = True,
        fixed_pmb_path: Optional[str] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.m_heat = m_heat
        self.n_topic = n_topic
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)
        
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature - 0.01))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
        
        if fixed_pmb_path and os.path.exists(fixed_pmb_path):
            self._load_fixed_pmb(fixed_pmb_path)
        else:
            self.pmb = nn.Parameter(torch.randn(m_heat, n_topic, d_model) * 0.02)
            self.register_buffer('topic_centers', None)
            self.register_buffer('pca_components', None)
            self.register_buffer('pca_mean', None)
        
        self.query_proj = nn.Linear(d_model * 3, d_model)
    
    def _load_fixed_pmb(self, path: str):
        """加载预构建的多模态记忆库"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        grid = data['grid']
        topic_centers = data.get('topic_centers')
        pca_components = data.get('pca_components')
        pca_mean = data.get('pca_mean')
        
        self.pmb = nn.Parameter(torch.tensor(grid, dtype=torch.float32))
        
        if topic_centers is not None:
            self.register_buffer('topic_centers', torch.tensor(topic_centers, dtype=torch.float32))
        else:
            self.register_buffer('topic_centers', self.pmb.data.mean(dim=0))
        
        if pca_components is not None:
            self.register_buffer('pca_components', torch.tensor(pca_components, dtype=torch.float32))
        else:
            self.register_buffer('pca_components', None)
        
        if pca_mean is not None:
            self.register_buffer('pca_mean', torch.tensor(pca_mean, dtype=torch.float32))
        else:
            self.register_buffer('pca_mean', None)
        
        print(f"Loaded multimodal memory bank from {path}")
        print(f"  Shape: {self.pmb.shape}")
        print(f"  PCA components: {self.pca_components.shape if self.pca_components is not None else 'None'}")
    
    def get_temperature(self) -> float:
        return F.softplus(self.temperature).item() + 0.01
    
    def compute_joint_addressing(
        self,
        query: torch.Tensor,
        use_top_k: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        计算联合寻址权重
        
        Args:
            query: 多模态融合查询向量 (B, D) 或 (B, D*3)
            use_top_k: 是否使用 top-k 选择
        
        Returns:
            包含寻址结果的字典
        """
        B = query.shape[0]
        device = query.device
        
        if query.shape[-1] == self.d_model * 3:
            query = self.query_proj(query)
        
        query = self.dropout(query)
        
        M, N, D = self.pmb.shape
        
        pmb_flat = self.pmb.view(-1, D)
        
        similarities = torch.matmul(query, pmb_flat.T) / self.get_temperature()
        
        joint_weights = F.softmax(similarities, dim=-1)
        joint_weights = joint_weights.view(B, M, N)
        
        heat_marginal = joint_weights.sum(dim=-1)
        topic_marginal = joint_weights.sum(dim=1)
        
        if use_top_k:
            top_k_weights, top_k_indices = torch.topk(joint_weights.view(B, -1), self.top_k, dim=-1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
            
            z_aug = torch.zeros(B, D, device=device)
            for b in range(B):
                for k in range(self.top_k):
                    idx = top_k_indices[b, k]
                    h = idx // N
                    t = idx % N
                    z_aug[b] += top_k_weights[b, k] * self.pmb[h, t]
        else:
            z_aug = torch.matmul(joint_weights.view(B, -1), pmb_flat)
            top_k_weights = joint_weights.view(B, -1)[:, :self.top_k]
        
        return {
            'joint_weights': joint_weights,
            'heat_marginal': heat_marginal,
            'topic_marginal': topic_marginal,
            'top_k_weights': top_k_weights,
            'z_aug': z_aug,
            'temperature': self.get_temperature(),
        }
    
    def forward(
        self,
        query: torch.Tensor,
        use_top_k: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 多模态融合查询向量 (B, D*3) 或 (B, D)
                   如果是 D*3，则通过 query_proj 投影到 D 维
        
        Returns:
            包含寻址结果的字典
        """
        return self.compute_joint_addressing(query, use_top_k)


def create_topo_pmb_multimodal(
    d_model: int = 768,
    m_heat: int = 6,
    n_topic: int = 32,
    top_k: int = 5,
    temperature: float = 1.0,
    learnable_temperature: bool = True,
    fixed_pmb_path: Optional[str] = None,
    dropout: float = 0.1,
) -> TopologyAwarePMBMultimodal:
    """创建多模态拓扑感知参数化记忆库"""
    return TopologyAwarePMBMultimodal(
        d_model=d_model,
        m_heat=m_heat,
        n_topic=n_topic,
        top_k=top_k,
        temperature=temperature,
        learnable_temperature=learnable_temperature,
        fixed_pmb_path=fixed_pmb_path,
        dropout=dropout,
    )
