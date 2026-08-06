"""
新数据集 (V2) 记忆库预初始化
使用K-means聚类训练集特征，将聚类中心作为记忆库初始值
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_v2")
FIXED_ASSETS_DIR = os.path.join(PROJECT_ROOT, ".fixed_assets")

os.makedirs(FIXED_ASSETS_DIR, exist_ok=True)

M_HEAT = 6
N_TOPIC = 32
D_MODEL = 768


def load_train_features():
    """加载训练集的视觉特征"""
    print("加载训练集视觉特征...")
    
    train_path = os.path.join(OUTPUT_DIR, 'train.pkl')
    df = pd.read_pickle(train_path)
    
    visual_features = df['visual_feature_embedding_cls'].tolist()
    
    all_frame_features = []
    for video_feat in tqdm(visual_features, desc="收集帧特征"):
        if video_feat and len(video_feat) > 0:
            for frame_feat in video_feat:
                if frame_feat and len(frame_feat) == D_MODEL:
                    all_frame_features.append(frame_feat)
    
    features = np.array(all_frame_features)
    print(f"总帧数: {len(features)}, 特征维度: {features.shape[1]}")
    
    return features


def kmeans_clustering(features, n_clusters, random_state=42):
    """K-means聚类"""
    print(f"\n执行K-means聚类 (n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    kmeans.fit(features)
    return kmeans.cluster_centers_, kmeans.labels_


def create_memory_bank():
    """创建记忆库"""
    print("=" * 60)
    print("记忆库预初始化 (K-means)")
    print("=" * 60)
    
    features = load_train_features()
    
    total_clusters = M_HEAT * N_TOPIC
    print(f"\n总聚类数: {M_HEAT} × {N_TOPIC} = {total_clusters}")
    
    cluster_centers, labels = kmeans_clustering(features, total_clusters)
    
    print(f"\n聚类中心形状: {cluster_centers.shape}")
    
    master_grid = cluster_centers.reshape(M_HEAT, N_TOPIC, D_MODEL)
    
    print(f"记忆库形状: {master_grid.shape}")
    
    save_path = os.path.join(FIXED_ASSETS_DIR, f"master_grid_M{M_HEAT}_N{N_TOPIC}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(master_grid, f)
    
    print(f"\n记忆库已保存到: {save_path}")
    
    print("\n[统计信息]")
    print(f"  记忆库范数: {np.linalg.norm(master_grid):.4f}")
    print(f"  记忆库均值: {master_grid.mean():.6f}")
    print(f"  记忆库标准差: {master_grid.std():.6f}")
    print(f"  记忆库最小值: {master_grid.min():.6f}")
    print(f"  记忆库最大值: {master_grid.max():.6f}")
    
    return master_grid


if __name__ == '__main__':
    create_memory_bank()
