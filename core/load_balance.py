"""
Anti-Collapse Load Balancing Mechanisms

严格按照ACM MM 2026审稿规范实现

防止记忆坍塌的负载均衡更新机制 (Anti-Collapse Update via Load Balancing)

核心公式:
L_bal = log(M × N) + Σ_{i,j} p_{i,j} log(p_{i,j})

其中:
p_{i,j} = (1/|B|) Σ_{x ∈ B} w_{i,j}(x)

当p_{i,j}为均匀分布1/(M×N)时，该项达到最小值0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class EntropyMaximizationLoadBalancing(nn.Module):
    """
    熵最大化负载均衡损失
    
    严格按照审稿人公式:
    L_bal = log(M × N) + Σ_{i,j} p_{i,j} log(p_{i,j})
    
    正则化参数列表:
    - B: 当前训练的Mini-batch集合，大小为|B|
    - p_{i,j}: 单个节点(i,j)在当前Batch中的边际激活概率
    - γ: 正则化损失的权重超参数（通常设为0.01左右）
    """
    
    def __init__(
        self,
        m_heat: int = 6,
        n_topic: int = 32,
        gamma: float = 0.01,
        eps: float = 1e-8,
    ):
        """
        Args:
            m_heat: 热度层级数 M
            n_topic: 主题数量 N
            gamma: 正则化损失权重
            eps: 数值稳定性常数
        """
        super().__init__()
        
        self.m_heat = m_heat
        self.n_topic = n_topic
        self.total_slots = m_heat * n_topic
        self.gamma = gamma
        self.eps = eps
        
        self.register_buffer(
            'max_entropy',
            torch.log(torch.tensor(self.total_slots, dtype=torch.float32))
        )
    
    def compute_slot_utilization(
        self,
        joint_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算每个记忆槽在Batch内的平均利用率
        
        p_{i,j} = (1/|B|) Σ_{x ∈ B} w_{i,j}(x)
        
        Args:
            joint_weights: (B, M, N) 联合寻址权重
        
        Returns:
            p: (M, N) 平均利用率分布
        """
        p = joint_weights.mean(dim=0)
        return p
    
    def forward(
        self,
        joint_weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算熵最大化负载均衡损失
        
        L_bal = log(M × N) + Σ_{i,j} p_{i,j} log(p_{i,j})
        
        当p_{i,j}为均匀分布1/(M×N)时，该项达到最小值0
        
        Args:
            joint_weights: (B, M, N) 联合寻址权重
        
        Returns:
            包含以下键的字典:
            - loss: 负载均衡损失
            - entropy: 当前分布的熵
            - max_entropy: 最大熵
            - utilization: (M, N) 平均利用率分布
            - dead_slots: 死节点数量
            - utilization_ratio: 利用率比例
        """
        p = self.compute_slot_utilization(joint_weights)
        
        p_safe = p + self.eps
        
        entropy = -(p_safe * torch.log(p_safe)).sum()
        
        loss = self.max_entropy - entropy
        
        loss = self.gamma * loss
        
        dead_slots = (p < 0.01).sum().item()
        
        utilization_ratio = entropy / self.max_entropy
        
        return {
            'loss': loss,
            'entropy': entropy,
            'max_entropy': self.max_entropy,
            'utilization': p,
            'dead_slots': dead_slots,
            'utilization_ratio': utilization_ratio,
        }


class AsymmetricMarginalLoadBalancing(nn.Module):
    """
    非对称边缘负载均衡
    
    基于边缘概率的非对称正则化:
    L_balance = α · D_KL(p̄^h || p^h_prior) + β · D_KL(p̄^t || U)
    
    其中:
    - 热度边缘分布: w^h_i = Σ_j w_{i,j}
    - 主题边缘分布: w^t_j = Σ_i w_{i,j}
    - p^h_prior: 数据集的热度先验分布（长尾分布）
    - U: 均匀分布
    """
    
    def __init__(
        self,
        m_heat: int = 6,
        n_topic: int = 32,
        alpha: float = 0.01,
        beta: float = 0.01,
        heat_prior: Optional[torch.Tensor] = None,
        long_tail_exponent: float = 1.5,
        eps: float = 1e-8,
    ):
        """
        Args:
            m_heat: 热度层级数
            n_topic: 主题数量
            alpha: 热度维度的损失权重
            beta: 主题维度的损失权重
            heat_prior: 预定义的热度先验分布 (M_heat,)
            long_tail_exponent: 长尾指数
            eps: 数值稳定性常数
        """
        super().__init__()
        
        self.m_heat = m_heat
        self.n_topic = n_topic
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        
        if heat_prior is not None:
            self.register_buffer('heat_prior', heat_prior)
        else:
            heat_prior = self._generate_long_tail_prior(m_heat, long_tail_exponent)
            self.register_buffer('heat_prior', heat_prior)
        
        uniform_topic = torch.ones(n_topic) / n_topic
        self.register_buffer('uniform_topic', uniform_topic)
    
    def _generate_long_tail_prior(
        self,
        m_heat: int,
        exponent: float,
    ) -> torch.Tensor:
        """
        生成长尾先验分布
        
        假设热度分布遵循幂律分布:
        p(i) ∝ i^(-exponent)
        
        低热度样本多，高热度样本少
        """
        ranks = torch.arange(1, m_heat + 1, dtype=torch.float32)
        prior = ranks.pow(-exponent)
        prior = prior / prior.sum()
        return prior
    
    def compute_marginals(
        self,
        joint_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从联合权重计算边缘分布
        
        热度边缘分布: w^h_i = Σ_j w_{i,j}
        主题边缘分布: w^t_j = Σ_i w_{i,j}
        
        Args:
            joint_weights: (B, M, N) 联合寻址权重
        
        Returns:
            heat_marginal: (B, M) 热度边缘分布
            topic_marginal: (B, N) 主题边缘分布
        """
        heat_marginal = joint_weights.sum(dim=-1)
        topic_marginal = joint_weights.sum(dim=1)
        return heat_marginal, topic_marginal
    
    def compute_batch_averaged_marginals(
        self,
        joint_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算批次内平均边缘分布
        
        p̄^h = (1/|B|) Σ w^h
        p̄^t = (1/|B|) Σ w^t
        
        Args:
            joint_weights: (B, M, N) 联合寻址权重
        
        Returns:
            p_heat: (M,) 批次平均热度边缘分布
            p_topic: (N,) 批次平均主题边缘分布
        """
        heat_marginal, topic_marginal = self.compute_marginals(joint_weights)
        
        p_heat = heat_marginal.mean(dim=0)
        p_topic = topic_marginal.mean(dim=0)
        
        return p_heat, p_topic
    
    def forward(
        self,
        joint_weights: torch.Tensor,
        heat_marginal: Optional[torch.Tensor] = None,
        topic_marginal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算非对称负载均衡损失
        
        L_balance = α · D_KL(p̄^h || p^h_prior) + β · D_KL(p̄^t || U)
        
        Args:
            joint_weights: (B, M, N) 联合寻址权重
            heat_marginal: (B, M) 热度边缘分布
            topic_marginal: (B, N) 主题边缘分布
        
        Returns:
            包含各项损失的字典
        """
        if heat_marginal is None or topic_marginal is None:
            heat_marginal, topic_marginal = self.compute_marginals(joint_weights)
        
        p_heat, p_topic = self.compute_batch_averaged_marginals(joint_weights)
        
        p_heat = p_heat + self.eps
        p_heat = p_heat / p_heat.sum()
        
        p_topic = p_topic + self.eps
        p_topic = p_topic / p_topic.sum()
        
        heat_prior = self.heat_prior + self.eps
        heat_prior = heat_prior / heat_prior.sum()
        
        heat_loss = F.kl_div(
            torch.log(p_heat + self.eps),
            heat_prior,
            reduction='sum'
        )
        
        topic_loss = F.kl_div(
            torch.log(p_topic + self.eps),
            self.uniform_topic,
            reduction='sum'
        )
        
        loss = self.alpha * heat_loss + self.beta * topic_loss
        
        heat_entropy = -(p_heat * torch.log(p_heat + self.eps)).sum()
        topic_entropy = -(p_topic * torch.log(p_topic + self.eps)).sum()
        
        return {
            'loss': loss,
            'heat_loss': heat_loss,
            'topic_loss': topic_loss,
            'p_heat': p_heat,
            'p_topic': p_topic,
            'heat_entropy': heat_entropy,
            'topic_entropy': topic_entropy,
        }
    
    def update_heat_prior_from_data(
        self,
        labels: torch.Tensor,
        num_bins: Optional[int] = None,
    ):
        """
        根据真实标签更新热度先验分布
        
        Args:
            labels: (N,) 真实热度标签
            num_bins: 分箱数量，默认使用m_heat
        """
        if num_bins is None:
            num_bins = self.m_heat
        
        labels_np = labels.detach().cpu().numpy()
        
        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(labels_np, percentiles)
        
        counts, _ = np.histogram(labels_np, bins=bins)
        
        prior = counts / counts.sum()
        prior = np.clip(prior, 1e-8, None)
        prior = prior / prior.sum()
        
        self.heat_prior = torch.from_numpy(prior).float().to(self.heat_prior.device)


class CommitmentLoss(nn.Module):
    """
    承诺损失
    
    确保查询向量与被选中的记忆槽保持一致性
    """
    
    def __init__(
        self,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.commitment_cost = commitment_cost
    
    def forward(
        self,
        query: torch.Tensor,
        z_aug: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算承诺损失
        
        L_commit = ||query - stop_gradient(z_aug)||^2
        
        Args:
            query: (B, D) 查询向量
            z_aug: (B, D) 增强特征
        
        Returns:
            commitment_loss: 承诺损失
        """
        return self.commitment_cost * F.mse_loss(query, z_aug.detach())


class CombinedLoadBalancing(nn.Module):
    """
    组合负载均衡损失
    
    结合熵最大化和非对称KL散度约束
    """
    
    def __init__(
        self,
        m_heat: int = 6,
        n_topic: int = 32,
        gamma: float = 0.01,
        alpha: float = 0.01,
        beta: float = 0.01,
        long_tail_exponent: float = 1.5,
        use_entropy_maximization: bool = True,
        use_asymmetric_kl: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        
        self.m_heat = m_heat
        self.n_topic = n_topic
        self.use_entropy_maximization = use_entropy_maximization
        self.use_asymmetric_kl = use_asymmetric_kl
        
        if use_entropy_maximization:
            self.entropy_lb = EntropyMaximizationLoadBalancing(
                m_heat=m_heat,
                n_topic=n_topic,
                gamma=gamma,
                eps=eps,
            )
        
        if use_asymmetric_kl:
            self.asymmetric_lb = AsymmetricMarginalLoadBalancing(
                m_heat=m_heat,
                n_topic=n_topic,
                alpha=alpha,
                beta=beta,
                long_tail_exponent=long_tail_exponent,
                eps=eps,
            )
        
        self.eps = eps
    
    def forward(
        self,
        joint_weights: torch.Tensor,
        heat_marginal: Optional[torch.Tensor] = None,
        topic_marginal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算组合负载均衡损失
        
        Args:
            joint_weights: (B, M, N) 联合寻址权重
            heat_marginal: (B, M) 热度边缘分布
            topic_marginal: (B, N) 主题边缘分布
        
        Returns:
            包含各项损失的字典
        """
        output = {}
        total_loss = torch.tensor(0.0, device=joint_weights.device)
        
        if self.use_entropy_maximization:
            entropy_output = self.entropy_lb(joint_weights)
            output['entropy_loss'] = entropy_output['loss']
            output['entropy'] = entropy_output['entropy']
            output['max_entropy'] = entropy_output['max_entropy']
            output['utilization'] = entropy_output['utilization']
            output['dead_slots'] = entropy_output['dead_slots']
            total_loss = total_loss + entropy_output['loss']
        
        if self.use_asymmetric_kl:
            kl_output = self.asymmetric_lb(joint_weights, heat_marginal, topic_marginal)
            output['kl_loss'] = kl_output['loss']
            output['heat_loss'] = kl_output['heat_loss']
            output['topic_loss'] = kl_output['topic_loss']
            output['p_heat'] = kl_output['p_heat']
            output['p_topic'] = kl_output['p_topic']
            output['heat_entropy'] = kl_output['heat_entropy']
            output['topic_entropy'] = kl_output['topic_entropy']
            total_loss = total_loss + kl_output['loss']
        
        output['loss'] = total_loss
        
        return output


class MemoryUtilizationMonitor:
    """
    记忆库利用率监控器
    
    用于可视化和分析PMB各槽位的使用情况
    """
    
    def __init__(
        self,
        m_heat: int = 6,
        n_topic: int = 32,
        window_size: int = 100,
    ):
        self.m_heat = m_heat
        self.n_topic = n_topic
        self.window_size = window_size
        
        self.heat_history = []
        self.topic_history = []
        self.utilization_history = []
    
    def update(
        self,
        joint_weights: torch.Tensor,
    ):
        """
        更新历史记录
        
        Args:
            joint_weights: (B, M, N) 联合寻址权重
        """
        utilization = joint_weights.mean(dim=0).detach().cpu()
        self.utilization_history.append(utilization)
        
        heat_marginal = joint_weights.sum(dim=-1).mean(dim=0).detach().cpu()
        topic_marginal = joint_weights.sum(dim=1).mean(dim=0).detach().cpu()
        
        self.heat_history.append(heat_marginal)
        self.topic_history.append(topic_marginal)
        
        if len(self.heat_history) > self.window_size:
            self.heat_history.pop(0)
            self.topic_history.pop(0)
            self.utilization_history.pop(0)
    
    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        if not self.heat_history:
            return {}
        
        utilization_tensor = torch.stack([
            u.view(-1) for u in self.utilization_history
        ])
        mean_utilization = utilization_tensor.mean(dim=0)
        
        entropy = -(mean_utilization * torch.log(mean_utilization + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(self.m_heat * self.n_topic, dtype=torch.float32))
        
        dead_slots = (mean_utilization < 0.01).sum().item()
        
        return {
            'entropy': entropy.item(),
            'max_entropy': max_entropy.item(),
            'utilization_ratio': (entropy / max_entropy).item(),
            'dead_slots': dead_slots,
            'total_slots': self.m_heat * self.n_topic,
        }
    
    def reset(self):
        """重置历史记录"""
        self.heat_history = []
        self.topic_history = []
        self.utilization_history = []


def create_load_balancing(
    m_heat: int = 6,
    n_topic: int = 32,
    gamma: float = 0.01,
    alpha: float = 0.01,
    beta: float = 0.01,
    use_entropy_maximization: bool = True,
    use_asymmetric_kl: bool = True,
    long_tail_exponent: float = 1.5,
) -> nn.Module:
    """创建负载均衡模块的工厂函数"""
    return CombinedLoadBalancing(
        m_heat=m_heat,
        n_topic=n_topic,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        use_entropy_maximization=use_entropy_maximization,
        use_asymmetric_kl=use_asymmetric_kl,
        long_tail_exponent=long_tail_exponent,
    )
