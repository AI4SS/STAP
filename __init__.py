"""
STAMP: Topology-Aware Parameterized Memory Bank

核心创新:
1. 全局锚点引导的双路径自适应编码
   - 共鸣驱动选择性扫描
   - 黑洞Token星状稀疏注意力

2. 拓扑感知参数化记忆库
   - 解耦拓扑寻址
   - 可学习温度系数
   - 边缘分布提取

3. 非对称边缘负载均衡
   - 热度维度: KL散度约束到长尾先验分布
   - 主题维度: KL散度约束到均匀分布

4. DPPO偏好优化
   - 成对偏好学习
   - 检索权重优化
"""

__version__ = '1.0.0'
__author__ = 'STAMP Team'

from .models.stamp_v2 import STAMPEncoderV2, create_stamp_encoder_v2
from .models.topo_pmb import TopologyAwarePMB, create_topo_pmb
from .models.load_balance import CombinedLoadBalancing, create_load_balancing

__all__ = [
    'STAMPEncoderV2',
    'create_stamp_encoder_v2',
    'TopologyAwarePMB',
    'create_topo_pmb',
    'CombinedLoadBalancing',
    'create_load_balancing',
]
