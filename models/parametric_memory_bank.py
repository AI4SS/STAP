"""
Parametric Memory Bank (PMB) Module
参数化记忆库构建模块

Phase I: Memory Initialization & Anchoring
- 横轴 (Latent Topics): K-Means 无监督发现的 N 个内容赛道
- 纵轴 (Heat Levels): 由 Ground Truth 监督划分的 M 个热度等级

学术包装：
- HL-Grid → Parametric Memory Bank (PMB)
- Prototype → Experience Prototype / Memory Slot
- Grid Filling → Memory Initialization
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch


class ParametricMemoryBank:
    """
    Parametric Memory Bank (参数化记忆库)

    M × N 的正交网格记忆库:
    - M: Heat Levels (热度层级)
    - N: Latent Topics (隐语义主题)

    每个 Memory Slot 存储一个 Experience Prototype (经验原型)
    """

    def __init__(self, m_heat=4, n_topic=32, feature_dim=768, random_state=42):
        """
        初始化 Parametric Memory Bank

        Args:
            m_heat: 热度层数 (默认4层: Viral, Hot, Mid, Cold)
            n_topic: 隐语义主题数 (默认32个聚类)
            feature_dim: 特征维度
            random_state: 随机种子
        """
        self.m_heat = m_heat
        self.n_topic = n_topic
        self.feature_dim = feature_dim
        self.random_state = random_state

        # 记忆库: [M, N, feature_dim]
        self.memory_bank = np.zeros((m_heat, n_topic, feature_dim), dtype=np.float32)

        # 主题中心: [N, feature_dim]
        self.topic_centers = np.zeros((n_topic, feature_dim), dtype=np.float32)

        # 热度边界: [M-1] (分位数)
        self.heat_boundaries = None

        # 统计信息
        self.slot_counts = np.zeros((m_heat, n_topic), dtype=np.int32)

    def build_from_data(self, train_df, content_feature_col='content_feature', label_col='label'):
        """
        从训练数据构建 Parametric Memory Bank

        Phase I: Memory Initialization & Anchoring

        Step 1: Latent Topic Discovery (K-Means)
        Step 2: Heat Stratification (分位数划分)
        Step 3: Memory Bank Initialization

        Args:
            train_df: 训练数据 DataFrame
            content_feature_col: 内容特征列名
            label_col: 标签列名
        """
        print("=" * 60)
        print("Parametric Memory Bank Construction")
        print("(Phase I: Memory Initialization & Anchoring)")
        print("=" * 60)

        # 提取内容特征和标签
        content_features = np.array(train_df[content_feature_col].tolist())
        labels = np.array(train_df[label_col].tolist())

        print(f"\n[Step 1] Latent Topic Discovery (K-Means)")
        print(f"  Content feature shape: {content_features.shape}")
        print(f"  Number of topics (N): {self.n_topic}")

        # Step 1: K-Means 聚类发现隐式主题
        kmeans = KMeans(
            n_clusters=self.n_topic,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        topic_ids = kmeans.fit_predict(content_features)
        self.topic_centers = kmeans.cluster_centers_.astype(np.float32)

        print(f"  K-Means converged: {kmeans.n_iter_} iterations")
        print(f"  Topic centers shape: {self.topic_centers.shape}")

        # 统计每个主题的样本数
        unique, counts = np.unique(topic_ids, return_counts=True)
        print(f"  Topic distribution (min/avg/max): {counts.min()} / {counts.mean():.1f} / {counts.max()}")

        print(f"\n[Step 2] Heat Stratification")
        print(f"  Label shape: {labels.shape}")
        print(f"  Number of heat levels (M): {self.m_heat}")

        # Step 2: 根据热度分位数划分
        self.heat_boundaries = np.percentile(labels, np.linspace(0, 100, self.m_heat + 1)[1:-1])
        print(f"  Heat boundaries: {self.heat_boundaries}")

        # 分配热度层
        heat_ids = np.zeros(len(labels), dtype=np.int32)
        for m in range(self.m_heat):
            if m == 0:
                mask = labels <= self.heat_boundaries[0]
            elif m == self.m_heat - 1:
                mask = labels > self.heat_boundaries[-1]
            else:
                mask = (labels > self.heat_boundaries[m - 1]) & (labels <= self.heat_boundaries[m])
            heat_ids[mask] = m

        # 统计每个热度层的样本数
        unique, counts = np.unique(heat_ids, return_counts=True)
        print(f"  Heat distribution: {dict(zip(unique, counts))}")

        print(f"\n[Step 3] Memory Bank Initialization")

        # Step 3: 填充记忆库
        self.slot_counts.fill(0)
        self.memory_bank.fill(0)

        for idx in tqdm(range(len(train_df)), desc="Initializing memory slots"):
            t_id = topic_ids[idx]  # 主题 ID
            h_id = heat_ids[idx]   # 热度 ID

            # 累积特征（用于后续平均）
            self.memory_bank[h_id, t_id] += content_features[idx]
            self.slot_counts[h_id, t_id] += 1

        # 计算平均值
        for h in range(self.m_heat):
            for t in range(self.n_topic):
                if self.slot_counts[h, t] > 0:
                    self.memory_bank[h, t] /= self.slot_counts[h, t]
                else:
                    # Empty Slot Handling: 使用该主题的全局中心
                    self.memory_bank[h, t] = self.topic_centers[t]

        print(f"\n  Memory Bank shape: {self.memory_bank.shape}")
        print(f"  Non-empty slots: {(self.slot_counts > 0).sum()} / {self.m_heat * self.n_topic}")

        # 打印网格统计
        print(f"\n  Memory Slot Statistics:")
        for h in range(self.m_heat):
            non_empty = (self.slot_counts[h] > 0).sum()
            print(f"    Heat Level {h}: {non_empty}/{self.n_topic} slots initialized")

        print("\n" + "=" * 60)
        print("Parametric Memory Bank Construction Complete!")
        print("=" * 60)

    def query(self, q, top_k=5):
        """
        Generative Routing: 查询记忆库，返回 Top-K 最相似的经验原型

        Args:
            q: 查询向量 [feature_dim] 或 [batch, feature_dim]
            top_k: 返回 Top-K 个经验原型

        Returns:
            codes: 选中的经验原型 [top_k, feature_dim]
            weights: 相似度权重 [top_k]
            positions: 记忆槽位置 [(h, t), ...]
        """
        if isinstance(q, np.ndarray):
            # NumPy 数组路径
            if q.ndim == 1:
                q = q.reshape(1, -1)
            q_flat = q.reshape(q.shape[0], -1)
            bank_flat = self.memory_bank.reshape(self.m_heat * self.n_topic, -1)

            # 计算余弦相似度
            q_norm = q_flat / (np.linalg.norm(q_flat, axis=1, keepdims=True) + 1e-8)
            bank_norm = bank_flat / (np.linalg.norm(bank_flat, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(q_norm, bank_norm.T)[0]

            top_indices = np.argsort(similarities)[-top_k:][::-1]
            top_similarities = similarities[top_indices]
        else:
            # PyTorch tensor 路径
            if q.dim() == 1:
                q = q.unsqueeze(0)
            q_flat = q.reshape(q.shape[0], -1)
            bank_flat = torch.tensor(self.memory_bank, device=q.device).reshape(
                self.m_heat * self.n_topic, -1
            )

            # 计算余弦相似度
            q_norm = torch.nn.functional.normalize(q_flat, dim=1)
            bank_norm = torch.nn.functional.normalize(bank_flat, dim=1)
            similarities = torch.matmul(q_norm, bank_norm.T)[0]

            top_indices = torch.argsort(similarities)[-top_k:]
            top_indices_sorted = torch.argsort(similarities[top_indices], descending=True)
            top_indices = top_indices[top_indices_sorted]
            top_similarities = similarities[top_indices].cpu().numpy()

        # 转换为记忆槽坐标
        codes = []
        positions = []
        for idx in top_indices:
            if isinstance(idx, np.ndarray) or isinstance(idx, int):
                idx_int = int(idx)
            else:
                idx_int = idx.item()
            h = idx_int // self.n_topic
            t = idx_int % self.n_topic
            codes.append(self.memory_bank[h, t])
            positions.append((h, t))

        codes = np.array(codes)
        weights = top_similarities / (np.abs(top_similarities).sum() + 1e-8)

        return codes, weights, positions

    def get_prototype_aggregation(self, q, top_k=5):
        """
        Prototype Aggregation: 获取查询的加权聚合原型

        Args:
            q: 查询向量
            top_k: Top-K 原型

        Returns:
            aggregated_prompt: 加权聚合的原型 [feature_dim]
        """
        codes, weights, _ = self.query(q, top_k)
        aggregated_prompt = np.sum(codes * weights.reshape(-1, 1), axis=0)
        return aggregated_prompt

    def update_memory_slot(self, slot_position, q, gamma, similarity_weight=0.0):
        """
        Error-Driven Memory Consolidation: 更新记忆槽

        c <- gamma * c + (1 - gamma) * q

        Args:
            slot_position: (h, t) 记忆槽位置
            q: 查询向量
            gamma: 动量系数 (误差大 -> gamma小 -> 大幅更新)
            similarity_weight: 相似度权重 (可选)
        """
        h, t = slot_position
        current_prototype = self.memory_bank[h, t]

        if isinstance(gamma, torch.Tensor):
            gamma = gamma.item()

        # 更新规则
        new_prototype = gamma * current_prototype + (1 - gamma) * q

        # 平滑更新
        self.memory_bank[h, t] = (1 - similarity_weight) * new_prototype + similarity_weight * current_prototype

    def save(self, path):
        """保存 Parametric Memory Bank 到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'memory_bank': self.memory_bank,
                'topic_centers': self.topic_centers,
                'heat_boundaries': self.heat_boundaries,
                'slot_counts': self.slot_counts,
                'm_heat': self.m_heat,
                'n_topic': self.n_topic,
                'feature_dim': self.feature_dim,
                'random_state': self.random_state
            }, f)
        print(f"Parametric Memory Bank saved to: {path}")

    def load(self, path):
        """从文件加载 Parametric Memory Bank（兼容旧格式）"""
        with open(path, 'rb') as f:
            data = pickle.load(f)

            # 兼容旧格式（使用 'grid' 键）和新格式（使用 'memory_bank' 键）
            if 'memory_bank' in data:
                self.memory_bank = data['memory_bank']
            elif 'grid' in data:
                self.memory_bank = data['grid']  # 兼容旧 HL-Grid 格式
            else:
                raise ValueError(f"Invalid PMB file: missing 'memory_bank' or 'grid' key")

            self.topic_centers = data.get('topic_centers', np.zeros((self.n_topic, self.feature_dim)))
            self.heat_boundaries = data.get('heat_boundaries')
            self.slot_counts = data.get('grid_counts', np.zeros((self.m_heat, self.n_topic), dtype=np.int32))

            # 从文件中读取或保持初始化值
            self.m_heat = data.get('m_heat', self.m_heat)
            self.n_topic = data.get('n_topic', self.n_topic)
            self.feature_dim = data.get('feature_dim', self.feature_dim)
            self.random_state = data.get('random_state', self.random_state)

        print(f"Parametric Memory Bank loaded from: {path}")
        print(f"  Memory Bank shape: {self.memory_bank.shape}")
        print(f"  Non-empty slots: {(self.slot_counts > 0).sum()}")

    def to_tensor(self, device='cpu'):
        """将记忆库转换为 PyTorch 张量"""
        return torch.tensor(self.memory_bank, dtype=torch.float32, device=device)


def extract_content_features_from_df(df, feature_dim=768):
    """
    从 DataFrame 提取内容特征

    Args:
        df: 数据 DataFrame
        feature_dim: 特征维度

    Returns:
        content_features: 内容特征数组 [N, feature_dim]
    """
    visual_cls_list = df['visual_feature_embedding_cls'].tolist()
    textual_list = df['textual_feature_embedding'].tolist()

    content_features = []

    for i in range(len(df)):
        # 提取视觉特征 (平均所有帧)
        visual_cls = np.array(visual_cls_list[i])
        visual_pooled = visual_cls.mean(axis=0)

        # 提取文本特征
        textual = np.array(textual_list[i]).flatten()

        # 拼接
        if len(visual_pooled) > feature_dim:
            visual_pooled = visual_pooled[:feature_dim]
        if len(textual) > feature_dim:
            textual = textual[:feature_dim]

        combined = np.concatenate([visual_pooled, textual])
        if combined.shape[0] > feature_dim:
            content_feat = combined[:feature_dim]
        elif combined.shape[0] < feature_dim:
            content_feat = np.pad(combined, (0, feature_dim - combined.shape[0]))
        else:
            content_feat = combined

        content_features.append(content_feat)

    return np.array(content_features, dtype=np.float32)


def build_pmb_main(train_path, output_path, m_heat=4, n_topic=32, feature_dim=768):
    """
    主函数：构建 Parametric Memory Bank

    Args:
        train_path: 训练数据 pkl 路径
        output_path: 输出记忆库路径
        m_heat: 热度层数
        n_topic: 主题数
        feature_dim: 特征维度
    """
    print(f"\nLoading training data from: {train_path}")
    train_df = pd.read_pickle(train_path)

    # 提取内容特征
    print(f"\nExtracting content features...")
    content_features = extract_content_features_from_df(train_df, feature_dim)
    train_df['content_feature'] = content_features.tolist()

    # 构建记忆库
    pmb = ParametricMemoryBank(m_heat=m_heat, n_topic=n_topic, feature_dim=feature_dim)
    pmb.build_from_data(train_df)

    # 保存
    pmb.save(output_path)

    return pmb


# ============ 兼容性支持 ============

class HLGrid(ParametricMemoryBank):
    """向后兼容 HL-Grid 类名"""
    pass


def build_hl_grid_main(train_path, output_path, m_heat=4, n_topic=32, feature_dim=768):
    """向后兼容函数"""
    return build_pmb_main(train_path, output_path, m_heat, n_topic, feature_dim)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build Parametric Memory Bank for NPM')
    parser.add_argument('--train_path', type=str, default='../data/train.pkl',
                        help='Path to training data pkl')
    parser.add_argument('--output_path', type=str, default='npm_cache/parametric_memory_bank.pkl',
                        help='Output path for PMB')
    parser.add_argument('--m_heat', type=int, default=4,
                        help='Number of heat levels (M)')
    parser.add_argument('--n_topic', type=int, default=32,
                        help='Number of latent topics (N)')
    parser.add_argument('--feature_dim', type=int, default=768,
                        help='Feature dimension')

    args = parser.parse_args()

    build_pmb_main(
        train_path=args.train_path,
        output_path=args.output_path,
        m_heat=args.m_heat,
        n_topic=args.n_topic,
        feature_dim=args.feature_dim
    )
