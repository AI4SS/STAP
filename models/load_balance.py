"""
Asymmetric Load Balancing Loss

实现非对称边缘负载均衡机制:
1. 热度维度: KL散度约束到数据长尾先验分布
2. 主题维度: KL散度约束到均匀分布

数学公式:
L_balance = α · D_KL(p̄^h || p^h_prior) + β · D_KL(p̄^t || U)

其中:
- p̄^h: 批次内平均热度边缘分布
- p^h_prior: 数据集的热度先验分布 (长尾分布)
- p̄^t: 批次内平均主题边缘分布
- U: 均匀分布

核心创新:
- 热度维度不强制均匀，而是符合数据真实分布
- 主题维度强制均匀，确保所有语义槽位都被利用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class AsymmetricLoadBalancing(nn.Module):
    """
    非对称负载均衡损失
    
    核心思想:
    - 热度维度: 约束到长尾先验，保持数据物理规律
    - 主题维度: 约束到均匀分布，确保语义多样性
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
            long_tail_exponent: 长尾指数，用于生成默认先验
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
        
        Args:
            joint_weights: (B, M, N) 联合寻址权重
            heat_marginal: (B, M) 热度边缘分布 (可选，若提供则跳过计算)
            topic_marginal: (B, N) 主题边缘分布 (可选，若提供则跳过计算)
        
        Returns:
            包含以下键的字典:
            - loss: 总负载均衡损失
            - heat_loss: 热度维度的KL散度损失
            - topic_loss: 主题维度的KL散度损失
            - p_heat: 批次平均热度边缘分布
            - p_topic: 批次平均主题边缘分布
            - heat_entropy: 热度边缘分布的熵
            - topic_entropy: 主题边缘分布的熵
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


class AdaptiveLoadBalancing(nn.Module):
    """
    自适应负载均衡
    
    根据训练进度动态调整α和β权重
    """
    
    def __init__(
        self,
        m_heat: int = 6,
        n_topic: int = 32,
        alpha_init: float = 0.001,
        beta_init: float = 0.001,
        alpha_max: float = 0.01,
        beta_max: float = 0.01,
        warmup_epochs: int = 10,
        heat_prior: Optional[torch.Tensor] = None,
        long_tail_exponent: float = 1.5,
    ):
        super().__init__()
        
        self.m_heat = m_heat
        self.n_topic = n_topic
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.alpha_max = alpha_max
        self.beta_max = beta_max
        self.warmup_epochs = warmup_epochs
        
        self.register_buffer('current_epoch', torch.tensor(0))
        
        if heat_prior is not None:
            self.register_buffer('heat_prior', heat_prior)
        else:
            heat_prior = self._generate_long_tail_prior(m_heat, long_tail_exponent)
            self.register_buffer('heat_prior', heat_prior)
        
        uniform_topic = torch.ones(n_topic) / n_topic
        self.register_buffer('uniform_topic', uniform_topic)
    
    def _generate_long_tail_prior(self, m_heat: int, exponent: float) -> torch.Tensor:
        ranks = torch.arange(1, m_heat + 1, dtype=torch.float32)
        prior = ranks.pow(-exponent)
        return prior / prior.sum()
    
    def get_current_weights(self) -> Tuple[float, float]:
        """获取当前的α和β权重"""
        epoch = self.current_epoch.item()
        
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            alpha = self.alpha_init + (self.alpha_max - self.alpha_init) * progress
            beta = self.beta_init + (self.beta_max - self.beta_init) * progress
        else:
            alpha = self.alpha_max
            beta = self.beta_max
        
        return alpha, beta
    
    def step(self):
        """更新epoch计数"""
        self.current_epoch += 1
    
    def forward(
        self,
        joint_weights: torch.Tensor,
        heat_marginal: Optional[torch.Tensor] = None,
        topic_marginal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算自适应负载均衡损失"""
        B, M, N = joint_weights.shape
        
        if heat_marginal is None:
            heat_marginal = joint_weights.sum(dim=-1)
        if topic_marginal is None:
            topic_marginal = joint_weights.sum(dim=1)
        
        p_heat = heat_marginal.mean(dim=0) + 1e-8
        p_heat = p_heat / p_heat.sum()
        
        p_topic = topic_marginal.mean(dim=0) + 1e-8
        p_topic = p_topic / p_topic.sum()
        
        alpha, beta = self.get_current_weights()
        
        heat_loss = F.kl_div(
            torch.log(p_heat),
            self.heat_prior,
            reduction='sum'
        )
        
        topic_loss = F.kl_div(
            torch.log(p_topic),
            self.uniform_topic,
            reduction='sum'
        )
        
        loss = alpha * heat_loss + beta * topic_loss
        
        return {
            'loss': loss,
            'heat_loss': heat_loss,
            'topic_loss': topic_loss,
            'alpha': alpha,
            'beta': beta,
            'p_heat': p_heat,
            'p_topic': p_topic,
        }


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
    
    def update(
        self,
        heat_marginal: torch.Tensor,
        topic_marginal: torch.Tensor,
    ):
        """
        更新历史记录
        
        Args:
            heat_marginal: (B, M) 或 (M,)
            topic_marginal: (B, N) 或 (N,)
        """
        if heat_marginal.dim() == 2:
            heat_marginal = heat_marginal.mean(dim=0)
        if topic_marginal.dim() == 2:
            topic_marginal = topic_marginal.mean(dim=0)
        
        self.heat_history.append(heat_marginal.detach().cpu())
        self.topic_history.append(topic_marginal.detach().cpu())
        
        if len(self.heat_history) > self.window_size:
            self.heat_history.pop(0)
            self.topic_history.pop(0)
    
    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        if not self.heat_history:
            return {}
        
        heat_tensor = torch.stack(self.heat_history)
        topic_tensor = torch.stack(self.topic_history)
        
        heat_mean = heat_tensor.mean(dim=0)
        topic_mean = topic_tensor.mean(dim=0)
        
        heat_entropy = -(heat_mean * torch.log(heat_mean + 1e-8)).sum()
        topic_entropy = -(topic_mean * torch.log(topic_mean + 1e-8)).sum()
        
        max_entropy_heat = torch.log(torch.tensor(self.m_heat, dtype=torch.float32))
        max_entropy_topic = torch.log(torch.tensor(self.n_topic, dtype=torch.float32))
        
        return {
            'heat_entropy': heat_entropy.item(),
            'topic_entropy': topic_entropy.item(),
            'heat_utilization': (heat_entropy / max_entropy_heat).item(),
            'topic_utilization': (topic_entropy / max_entropy_topic).item(),
            'heat_dead_slots': (heat_mean < 0.01).sum().item(),
            'topic_dead_slots': (topic_mean < 0.01).sum().item(),
        }
    
    def reset(self):
        """重置历史记录"""
        self.heat_history = []
        self.topic_history = []


def create_load_balancing(
    m_heat: int = 6,
    n_topic: int = 32,
    alpha: float = 0.01,
    beta: float = 0.01,
    adaptive: bool = False,
    heat_prior: Optional[torch.Tensor] = None,
    long_tail_exponent: float = 1.5,
) -> nn.Module:
    """创建负载均衡模块的工厂函数"""
    if adaptive:
        return AdaptiveLoadBalancing(
            m_heat=m_heat,
            n_topic=n_topic,
            alpha_init=alpha * 0.1,
            beta_init=beta * 0.1,
            alpha_max=alpha,
            beta_max=beta,
            heat_prior=heat_prior,
            long_tail_exponent=long_tail_exponent,
        )
    return AsymmetricLoadBalancing(
        m_heat=m_heat,
        n_topic=n_topic,
        alpha=alpha,
        beta=beta,
        heat_prior=heat_prior,
        long_tail_exponent=long_tail_exponent,
    )
