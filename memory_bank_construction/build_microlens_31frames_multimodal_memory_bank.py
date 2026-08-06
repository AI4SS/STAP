"""
构建 MicroLens 数据集的多模态记忆库 - 使用 31 帧特征
数据路径: data_MicroLens/processed_31frames
visual (31, 768) -> mean pooling -> 768 + textual (1024) + user (6) 特征融合 -> PCA降维到768维
"""
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

M_HEAT = 6
N_TOPIC = 32
FEATURE_DIM = 768

def load_microlens_31frames_data():
    """加载 31 帧特征数据"""
    data_path = os.path.join(PROJECT_ROOT, 'data_MicroLens/processed_31frames')
    
    with open(f'{data_path}/train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    print(f"训练数据:")
    print(f"  样本数: {len(train_data['labels'])}")
    print(f"  visual: {train_data['visual_features'][0].shape}")
    print(f"  text: {train_data['text_features'][0].shape}")
    print(f"  user: {train_data['user_features'][0].shape}")
    
    return train_data

def extract_multimodal_features(train_data):
    """
    提取 visual + text + user 多模态特征
    visual: (31, 768) -> mean pooling -> (768,)
    text: (1024,)
    user: (6,)
    """
    visual_features = train_data['visual_features']
    text_features = train_data['text_features']
    user_features = train_data['user_features']
    labels = train_data['labels']
    
    raw_features = []
    for i in tqdm(range(len(labels)), desc="提取多模态特征"):
        visual = np.array(visual_features[i]).mean(axis=0)  # (31, 768) -> (768,)
        text = np.array(text_features[i])  # (1024,)
        user = np.array(user_features[i])  # (6,)
        
        combined = np.concatenate([visual, text, user])  # (1798,)
        raw_features.append(combined)
    
    return np.array(raw_features), np.array(labels)

def fit_pca_and_transform(features, target_dim=768):
    """使用PCA降维到目标维度"""
    print(f"\n使用PCA降维: {features.shape[1]} -> {target_dim}")
    pca = PCA(n_components=target_dim, random_state=42)
    transformed = pca.fit_transform(features)
    print(f"  降维后形状: {transformed.shape}")
    print(f"  解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
    return transformed, pca

def build_memory_bank(features, labels, m_heat=6, n_topic=32):
    """构建记忆库"""
    print(f"\n构建记忆库: {m_heat} x {n_topic}")
    
    print("  K-Means 聚类...")
    kmeans = KMeans(n_clusters=n_topic, random_state=42, n_init=10, max_iter=300)
    topic_ids = kmeans.fit_predict(features)
    topic_centers = kmeans.cluster_centers_
    
    print("  热度分层...")
    heat_boundaries = np.percentile(labels, np.linspace(0, 100, m_heat + 1)[1:-1])
    print(f"  热度边界: {heat_boundaries}")
    
    heat_ids = np.zeros(len(labels), dtype=np.int32)
    for m in range(m_heat):
        if m == 0:
            mask = labels <= heat_boundaries[0]
        elif m == m_heat - 1:
            mask = labels > heat_boundaries[-1]
        else:
            mask = (labels > heat_boundaries[m - 1]) & (labels <= heat_boundaries[m])
        heat_ids[mask] = m
    
    memory_bank = np.zeros((m_heat, n_topic, features.shape[1]), dtype=np.float32)
    slot_counts = np.zeros((m_heat, n_topic), dtype=np.int32)
    
    print("  填充记忆槽...")
    for idx in tqdm(range(len(labels)), desc="填充记忆槽"):
        t_id = topic_ids[idx]
        h_id = heat_ids[idx]
        memory_bank[h_id, t_id] += features[idx]
        slot_counts[h_id, t_id] += 1
    
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
    print("构建 MicroLens 多模态记忆库 (31帧特征)")
    print("=" * 60)
    
    print("\n[1] 加载训练数据...")
    train_data = load_microlens_31frames_data()
    
    print("\n[2] 提取多模态特征...")
    raw_features, labels = extract_multimodal_features(train_data)
    print(f"  原始特征形状: {raw_features.shape}")
    print(f"  特征组成: visual(768) + text(1024) + user(6) = 1798")
    
    print("\n[3] PCA降维...")
    features, pca = fit_pca_and_transform(raw_features, FEATURE_DIM)
    
    print("\n[4] 构建记忆库...")
    memory_bank, topic_centers, heat_boundaries, slot_counts = build_memory_bank(
        features, labels, M_HEAT, N_TOPIC
    )
    
    output_path = os.path.join(PROJECT_ROOT, '.fixed_assets/master_grid_M6_N32_microlens_31frames_multimodal.pkl')
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
    print(f"  特征: visual(31帧mean) + text + user (PCA降维)")
    print(f"  解释方差比: {save_data['explained_variance_ratio']:.4f}")

if __name__ == '__main__':
    main()
