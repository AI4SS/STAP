"""
构建 informs 数据集的多模态记忆库
visual + textual + user 特征融合
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

M_HEAT = 6
N_TOPIC = 32
FEATURE_DIM = 768

def load_informs_data():
    data_path = os.path.join(PROJECT_ROOT, 'data_v2')
    with open(f'{data_path}/train.pkl', 'rb') as f:
        train_df = pickle.load(f)
    return train_df

def extract_multimodal_features(train_df):
    """
    提取 visual + text + user 多模态特征
    """
    visual_features = train_df['visual_feature_embedding_mean'].tolist()
    textual_features = train_df['textual_feature_embedding'].tolist()
    user_features = train_df['user_feature_vector'].tolist()
    labels = train_df['heat_value'].values
    
    raw_features = []
    for i in tqdm(range(len(train_df)), desc="提取多模态特征"):
        visual = np.array(visual_features[i]).mean(axis=0)  # (768,)
        text = np.array(textual_features[i])  # (768,)
        user = np.array(user_features[i])  # (6,)
        
        combined = np.concatenate([visual, text, user])  # (1542,)
        raw_features.append(combined)
    
    return np.array(raw_features), labels

def fit_pca_and_transform(features, target_dim=768):
    """
    使用PCA降维到目标维度
    """
    print(f"\n使用PCA降维: {features.shape[1]} -> {target_dim}")
    pca = PCA(n_components=target_dim, random_state=42)
    transformed = pca.fit_transform(features)
    print(f"  降维后形状: {transformed.shape}")
    print(f"  解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
    return transformed, pca

def build_memory_bank(features, labels, m_heat=6, n_topic=32):
    """
    构建记忆库
    """
    print(f"\n构建记忆库: {m_heat} x {n_topic}")
    
    # K-Means 聚类发现主题
    print("  K-Means 聚类...")
    kmeans = KMeans(n_clusters=n_topic, random_state=42, n_init=10, max_iter=300)
    topic_ids = kmeans.fit_predict(features)
    topic_centers = kmeans.cluster_centers_
    
    # 热度分层
    print("  热度分层...")
    heat_boundaries = np.percentile(labels, np.linspace(0, 100, m_heat + 1)[1:-1])
    print(f"  热度边界: {heat_boundaries}")
    
    # 分配热度层
    heat_ids = np.zeros(len(labels), dtype=np.int32)
    for m in range(m_heat):
        if m == 0:
            mask = labels <= heat_boundaries[0]
        elif m == m_heat - 1:
            mask = labels > heat_boundaries[-1]
        else:
            mask = (labels > heat_boundaries[m - 1]) & (labels <= heat_boundaries[m])
        heat_ids[mask] = m
    
    # 初始化记忆库
    memory_bank = np.zeros((m_heat, n_topic, features.shape[1]), dtype=np.float32)
    slot_counts = np.zeros((m_heat, n_topic), dtype=np.int32)
    
    print("  填充记忆槽...")
    for idx in tqdm(range(len(labels)), desc="填充记忆槽"):
        t_id = topic_ids[idx]
        h_id = heat_ids[idx]
        memory_bank[h_id, t_id] += features[idx]
        slot_counts[h_id, t_id] += 1
    
    # 计算平均值
    for h in range(m_heat):
        for t in range(n_topic):
            if slot_counts[h, t] > 0:
                memory_bank[h, t] /= slot_counts[h, t]
            else:
                memory_bank[h, t] = topic_centers[t]
    
    print(f"  记忆库形状: {memory_bank.shape}")
    print(f"  非空槽位: {(slot_counts > 0).sum()} / {m_heat * n_topic}")
    
    return memory_bank, topic_centers, heat_boundaries, slot_counts

def main():
    print("=" * 60)
    print("构建 informs 多模态记忆库 (visual + text + user)")
    print("=" * 60)
    
    # 加载数据
    print("\n[1] 加载 informs 训练数据...")
    train_df = load_informs_data()
    print(f"  训练集大小: {len(train_df)}")
    
    # 提取多模态特征
    print("\n[2] 提取多模态特征...")
    raw_features, labels = extract_multimodal_features(train_df)
    print(f"  原始特征形状: {raw_features.shape}")
    print(f"  特征组成: visual(768) + text(768) + user(6) = 1542")
    
    # PCA降维
    print("\n[3] PCA降维...")
    features, pca = fit_pca_and_transform(raw_features, FEATURE_DIM)
    
    # 构建记忆库
    print("\n[4] 构建记忆库...")
    memory_bank, topic_centers, heat_boundaries, slot_counts = build_memory_bank(
        features, labels, M_HEAT, N_TOPIC
    )
    
    # 保存
    output_path = os.path.join(PROJECT_ROOT, '.fixed_assets/master_grid_M6_N32_informs_multimodal.pkl')
    print(f"\n[5] 保存记忆库到: {output_path}")
    
    save_data = {
        'grid': memory_bank,
        'topic_centers': topic_centers,
        'heat_boundaries': heat_boundaries,
        'grid_counts': slot_counts,
        'm_heat': M_HEAT,
        'n_topic': N_TOPIC,
        'feature_dim': FEATURE_DIM,
        'pca_components': pca.components_,
        'pca_mean': pca.mean_,
        'explained_variance_ratio': pca.explained_variance_ratio_.sum(),
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print("\n" + "=" * 60)
    print("记忆库构建完成!")
    print("=" * 60)
    print(f"  文件: {output_path}")
    print(f"  形状: {memory_bank.shape}")
    print(f"  特征: visual + text + user (PCA降维)")
    print(f"  解释方差比: {save_data['explained_variance_ratio']:.4f}")

if __name__ == '__main__':
    main()
