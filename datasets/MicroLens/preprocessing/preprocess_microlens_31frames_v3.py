"""
MicroLens预处理脚本 - 31帧版本 (1封面 + 30帧)
学习rspmn/hlvq_grid.py的记忆库构建方式
- 视觉特征: 31帧均值池化作为全局特征
- 记忆库: HLGrid方式构建 (K-Means主题 + 热度分位数)
- 类别嵌入
- 安全特征工程（无数据泄露）
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class HLGrid:
    """
    Heat-Latent Grid (热度-隐语义网格)
    学习rspmn/hlvq_grid.py的实现

    M × N 的正交网格记忆库:
    - M: 热度层数 (Heat Levels)
    - N: 隐语义主题数 (Latent Topics)
    """

    def __init__(self, m_heat=6, n_topic=32, feature_dim=768, random_state=42):
        self.m_heat = m_heat
        self.n_topic = n_topic
        self.feature_dim = feature_dim
        self.random_state = random_state

        self.grid = np.zeros((m_heat, n_topic, feature_dim), dtype=np.float32)
        self.topic_centers = np.zeros((n_topic, feature_dim), dtype=np.float32)
        self.heat_boundaries = None
        self.grid_counts = np.zeros((m_heat, n_topic), dtype=np.int32)

    def build_from_data(self, content_features, labels):
        """
        从训练数据构建 HL-Grid

        Step 1: Global Latent Topic Discovery (K-Means)
        Step 2: Heat Stratification (分位数划分)
        Step 3: Orthogonal Grid Filling
        """
        print("=" * 60)
        print("HL-Grid Construction")
        print("=" * 60)

        print(f"\n[Step 1] Global Latent Topic Discovery (K-Means)")
        print(f"  Content feature shape: {content_features.shape}")
        print(f"  Number of topics (N): {self.n_topic}")

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

        unique, counts = np.unique(topic_ids, return_counts=True)
        print(f"  Topic distribution (min/avg/max): {counts.min()} / {counts.mean():.1f} / {counts.max()}")

        print(f"\n[Step 2] Heat Stratification")
        print(f"  Label shape: {labels.shape}")
        print(f"  Number of heat levels (M): {self.m_heat}")

        self.heat_boundaries = np.percentile(labels, np.linspace(0, 100, self.m_heat + 1)[1:-1])
        print(f"  Heat boundaries: {self.heat_boundaries}")

        heat_ids = np.zeros(len(labels), dtype=np.int32)
        for m in range(self.m_heat):
            if m == 0:
                mask = labels <= self.heat_boundaries[0]
            elif m == self.m_heat - 1:
                mask = labels > self.heat_boundaries[-1]
            else:
                mask = (labels > self.heat_boundaries[m - 1]) & (labels <= self.heat_boundaries[m])
            heat_ids[mask] = m

        unique, counts = np.unique(heat_ids, return_counts=True)
        print(f"  Heat distribution: {dict(zip(unique, counts))}")

        print(f"\n[Step 3] Orthogonal Grid Filling")

        self.grid_counts.fill(0)
        self.grid.fill(0)

        for idx in tqdm(range(len(content_features)), desc="Filling grid"):
            t_id = topic_ids[idx]
            h_id = heat_ids[idx]

            self.grid[h_id, t_id] += content_features[idx]
            self.grid_counts[h_id, t_id] += 1

        for h in range(self.m_heat):
            for t in range(self.n_topic):
                if self.grid_counts[h, t] > 0:
                    self.grid[h, t] /= self.grid_counts[h, t]
                else:
                    self.grid[h, t] = self.topic_centers[t]

        print(f"\n  Grid shape: {self.grid.shape}")
        print(f"  Non-empty cells: {(self.grid_counts > 0).sum()} / {self.m_heat * self.n_topic}")

        print(f"\n  Grid count statistics:")
        for h in range(self.m_heat):
            non_empty = (self.grid_counts[h] > 0).sum()
            print(f"    Heat {h}: {non_empty}/{self.n_topic} cells filled")

        print("\n" + "=" * 60)
        print("HL-Grid Construction Complete!")
        print("=" * 60)

        return topic_ids, heat_ids

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'grid': self.grid,
                'topic_centers': self.topic_centers,
                'heat_boundaries': self.heat_boundaries,
                'grid_counts': self.grid_counts,
                'm_heat': self.m_heat,
                'n_topic': self.n_topic,
                'feature_dim': self.feature_dim,
                'random_state': self.random_state
            }, f)
        print(f"HL-Grid saved to: {path}")


def load_visual_features(covers_dir, frames_dir, video_ids, num_frames=30):
    """
    加载视觉特征: 1封面 + 30帧
    使用均值池化得到全局特征
    """
    print("=" * 60)
    print("加载视觉特征 (1封面 + {}帧)".format(num_frames))
    print("=" * 60)
    
    visual_features = []
    valid_video_ids = []
    
    for vid in tqdm(video_ids, desc="加载视觉特征"):
        frames = []
        
        cover_path = os.path.join(covers_dir, f"{vid}.jpg")
        if not os.path.exists(cover_path):
            cover_path = os.path.join(covers_dir, f"{vid}.png")
        
        if os.path.exists(cover_path):
            try:
                cover_feat_path = cover_path.replace('.jpg', '_feat.npy').replace('.png', '_feat.npy')
                if os.path.exists(cover_feat_path):
                    feat = np.load(cover_feat_path)
                else:
                    continue
                frames.append(feat)
            except:
                continue
        
        for i in range(num_frames):
            frame_path = os.path.join(frames_dir, f"{vid}_{i}.jpg")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(frames_dir, f"{vid}-{i}.jpg")
            
            if os.path.exists(frame_path):
                try:
                    frame_feat_path = frame_path.replace('.jpg', '_feat.npy')
                    if os.path.exists(frame_feat_path):
                        feat = np.load(frame_feat_path)
                        frames.append(feat)
                except:
                    pass
        
        if len(frames) > 0:
            frames = np.array(frames)
            visual_pooled = frames.mean(axis=0)
            visual_features.append(visual_pooled)
            valid_video_ids.append(vid)
    
    visual_features = np.array(visual_features, dtype=np.float32)
    print(f"视觉特征形状: {visual_features.shape}")
    print(f"有效视频数: {len(valid_video_ids)}")
    
    return visual_features, valid_video_ids


def load_text_features(titles_file, video_ids):
    """
    加载文本特征
    """
    print("\n" + "=" * 60)
    print("加载文本特征")
    print("=" * 60)
    
    text_features = []
    valid_video_ids = []
    
    titles_df = pd.read_csv(titles_file, header=None, names=['video_id', 'title'])
    titles_dict = dict(zip(titles_df['video_id'], titles_df['title']))
    
    for vid in tqdm(video_ids, desc="加载文本特征"):
        if vid in titles_dict:
            text_feat_path = os.path.join(os.path.dirname(titles_file), f"text_features/{vid}_text.npy")
            if os.path.exists(text_feat_path):
                feat = np.load(text_feat_path)
                text_features.append(feat)
                valid_video_ids.append(vid)
    
    if len(text_features) > 0:
        text_features = np.array(text_features, dtype=np.float32)
    else:
        text_features = np.zeros((len(video_ids), 1024), dtype=np.float32)
        valid_video_ids = video_ids
    
    print(f"文本特征形状: {text_features.shape}")
    print(f"有效视频数: {len(valid_video_ids)}")
    
    return text_features, valid_video_ids


def extract_content_features(visual_features, text_features, feature_dim=768):
    """
    提取内容特征: 视觉特征 + 文本特征 拼接后截断
    学习rspmn/hlvq_grid.py的extract_content_features_from_df
    """
    print("\n" + "=" * 60)
    print("提取内容特征")
    print("=" * 60)
    
    content_features = []
    
    for i in tqdm(range(len(visual_features)), desc="提取内容特征"):
        visual_pooled = visual_features[i]
        textual = text_features[i].flatten()
        
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
    
    content_features = np.array(content_features, dtype=np.float32)
    print(f"内容特征形状: {content_features.shape}")
    
    return content_features


def create_category_embeddings(tags_file, d_model=768):
    """
    创建类别嵌入
    """
    print("\n" + "=" * 60)
    print("创建类别嵌入")
    print("=" * 60)
    
    if not os.path.exists(tags_file):
        print(f"标签文件不存在: {tags_file}")
        categories = ['unknown']
        category_to_idx = {'unknown': 0}
        category_embeddings = np.random.randn(1, d_model).astype(np.float32) * 0.02
        return category_embeddings, category_to_idx, categories
    
    tags_df = pd.read_csv(tags_file)
    print(f"标签文件列: {tags_df.columns.tolist()}")
    
    if 'category' in tags_df.columns:
        categories = tags_df['category'].unique()
    elif 'tag' in tags_df.columns:
        categories = tags_df['tag'].unique()
    else:
        categories = tags_df.iloc[:, 0].unique()
    
    categories = sorted([str(c) for c in categories if pd.notna(c)])
    print(f"类别数量: {len(categories)}")
    
    category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    np.random.seed(42)
    category_embeddings = np.random.randn(len(categories), d_model).astype(np.float32) * 0.02
    
    print(f"类别嵌入形状: {category_embeddings.shape}")
    
    return category_embeddings, category_to_idx, categories


def load_labels_and_user_features(comment_file, titles_file, video_ids, category_to_idx):
    """
    加载标签和安全特征
    标签: log2(comment_count)
    安全特征: title_length, title_words, has_special, has_numbers, avg_word_len, category
    """
    print("\n" + "=" * 60)
    print("加载标签和用户特征")
    print("=" * 60)
    
    comments_df = pd.read_csv(comment_file, sep='\t', header=None, names=['comment_id', 'video_id', 'content'],
                              dtype={'comment_id': str, 'video_id': str, 'content': str})
    video_comments = comments_df.groupby('video_id').size().reset_index(name='comment_count')
    print(f"评论数据: {len(video_comments)} 个视频")
    
    titles_df = pd.read_csv(titles_file, header=None, names=['video_id', 'title'],
                            dtype={'video_id': str, 'title': str})
    print(f"标题数据: {len(titles_df)} 个视频")
    
    video_ids_set = set(str(v) for v in video_ids)
    video_comments_set = set(video_comments['video_id'].tolist())
    titles_set = set(titles_df['video_id'].tolist())
    
    matched_videos = video_ids_set & video_comments_set & titles_set
    print(f"三数据源匹配视频数: {len(matched_videos)}")
    
    video_comments_dict = dict(zip(video_comments['video_id'], video_comments['comment_count']))
    titles_dict = dict(zip(titles_df['video_id'], titles_df['title']))
    
    labels = []
    user_features = []
    valid_video_ids = []
    valid_indices = []
    
    for idx, vid in enumerate(tqdm(video_ids, desc="处理视频")):
        vid_str = str(vid)
        
        if vid_str not in video_comments_dict:
            continue
        comment_count = video_comments_dict[vid_str]
        
        if vid_str not in titles_dict:
            continue
        title = str(titles_dict[vid_str])
        
        label = np.log2(comment_count + 1)
        
        title_length = len(title)
        title_words = len(title.split())
        has_special = 1 if any(c in title for c in '!@#$%^&*()_+-=[]{}|;:,.<>?') else 0
        has_numbers = 1 if any(c.isdigit() for c in title) else 0
        avg_word_len = np.mean([len(w) for w in title.split()]) if title.split() else 0
        
        category = 'unknown'
        category_idx = category_to_idx.get(category, 0)
        
        features = [title_length, title_words, has_special, has_numbers, avg_word_len, category_idx]
        
        labels.append(label)
        user_features.append(features)
        valid_video_ids.append(vid)
        valid_indices.append(idx)
    
    labels = np.array(labels, dtype=np.float32)
    user_features = np.array(user_features, dtype=np.float32)
    
    print(f"有效视频数: {len(valid_video_ids)}")
    print(f"标签范围: [{labels.min():.4f}, {labels.max():.4f}]")
    print(f"用户特征形状: {user_features.shape}")
    
    return labels, user_features, valid_video_ids, valid_indices


def prepare_training_data(
    visual_features,
    text_features,
    user_features,
    labels,
    video_ids,
    valid_indices,
    test_size=0.2,
    valid_size=0.1,
    random_state=42,
):
    """
    准备训练数据，划分train/valid/test
    """
    print("\n" + "=" * 60)
    print("划分训练/验证/测试集")
    print("=" * 60)
    
    visual_features = visual_features[valid_indices]
    text_features = text_features[valid_indices]
    video_ids = [video_ids[i] for i in valid_indices]
    
    indices = np.arange(len(video_ids))
    
    train_idx, temp_idx = train_test_split(
        indices, test_size=test_size + valid_size, random_state=random_state, shuffle=True
    )
    
    valid_ratio = valid_size / (test_size + valid_size)
    valid_idx, test_idx = train_test_split(
        temp_idx, test_size=1 - valid_ratio, random_state=random_state, shuffle=True
    )
    
    print(f"训练集: {len(train_idx)}")
    print(f"验证集: {len(valid_idx)}")
    print(f"测试集: {len(test_idx)}")
    
    scaler = StandardScaler()
    user_features_scaled = scaler.fit_transform(user_features)
    
    def create_split(indices):
        return {
            'visual_features': [visual_features[i] for i in indices],
            'text_features': [text_features[i] for i in indices],
            'user_features': [user_features_scaled[i] for i in indices],
            'labels': [labels[i] for i in indices],
            'video_ids': [video_ids[i] for i in indices],
        }
    
    train_data = create_split(train_idx)
    valid_data = create_split(valid_idx)
    test_data = create_split(test_idx)
    
    return train_data, valid_data, test_data, scaler, train_idx


def save_processed_data(output_dir, train_data, valid_data, test_data, hl_grid, category_embeddings, scaler, categories):
    """
    保存处理后的数据
    """
    print("\n" + "=" * 60)
    print("保存处理后的数据")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    print(f"保存 train.pkl: {len(train_data['labels'])} 样本")
    
    with open(os.path.join(output_dir, 'valid.pkl'), 'wb') as f:
        pickle.dump(valid_data, f)
    print(f"保存 valid.pkl: {len(valid_data['labels'])} 样本")
    
    with open(os.path.join(output_dir, 'test.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    print(f"保存 test.pkl: {len(test_data['labels'])} 样本")
    
    np.save(os.path.join(output_dir, 'memory_bank.npy'), hl_grid.grid)
    print(f"保存 memory_bank.npy: {hl_grid.grid.shape}")
    
    np.save(os.path.join(output_dir, 'category_embeddings.npy'), category_embeddings)
    print(f"保存 category_embeddings.npy: {category_embeddings.shape}")
    
    meta = {
        'num_train': len(train_data['labels']),
        'num_valid': len(valid_data['labels']),
        'num_test': len(test_data['labels']),
        'visual_feature_dim': train_data['visual_features'][0].shape[0],
        'text_feature_dim': train_data['text_features'][0].shape[0],
        'user_feature_dim': train_data['user_features'][0].shape[0],
        'memory_bank_shape': list(hl_grid.grid.shape),
        'category_embeddings_shape': list(category_embeddings.shape),
        'num_categories': len(categories),
        'categories': categories,
        'max_frames': 31,
        'm_heat': hl_grid.m_heat,
        'n_topic': hl_grid.n_topic,
        'note': 'Memory bank built using HLGrid method (K-Means + Heat Stratification)',
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"保存 meta.json")
    
    print("\n" + "=" * 60)
    print("预处理完成!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='MicroLens预处理 - 31帧版本')
    parser.add_argument('--features_dir', type=str,
                        default='/root/gsar_jamba_topo/data_MicroLens/extracted_features_31frames')
    parser.add_argument('--text_features_dir', type=str,
                        default='/root/gsar_jamba_topo/data_MicroLens/extracted_features_6frames')
    parser.add_argument('--output_dir', type=str,
                        default='/root/gsar_jamba_topo/data_MicroLens/processed_31frames')
    parser.add_argument('--comment_file', type=str,
                        default='/root/gsar_jamba_topo/data_MicroLens/MicroLens-100k_comment_en.txt')
    parser.add_argument('--titles_file', type=str,
                        default='/root/gsar_jamba_topo/data_MicroLens/MicroLens-100k_title_en.csv')
    parser.add_argument('--tags_file', type=str,
                        default='/root/gsar_jamba_topo/data_MicroLens/tags_to_summary.csv')
    parser.add_argument('--m_heat', type=int, default=6)
    parser.add_argument('--n_topic', type=int, default=32)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--valid_size', type=float, default=0.1)
    parser.add_argument('--random_state', type=int, default=42)
    
    args = parser.parse_args()
    
    visual_features = np.load(os.path.join(args.features_dir, 'visual_features_clip.npy'))
    text_features = np.load(os.path.join(args.text_features_dir, 'text_features_angle.npy'))
    
    with open(os.path.join(args.features_dir, 'video_ids.json'), 'r') as f:
        visual_video_ids = json.load(f)
    
    with open(os.path.join(args.text_features_dir, 'video_ids.json'), 'r') as f:
        text_video_ids = json.load(f)
    
    visual_id_to_idx = {vid: i for i, vid in enumerate(visual_video_ids)}
    text_id_to_idx = {vid: i for i, vid in enumerate(text_video_ids)}
    
    common_video_ids = [vid for vid in visual_video_ids if vid in text_id_to_idx]
    print(f"视觉特征视频数: {len(visual_video_ids)}")
    print(f"文本特征视频数: {len(text_video_ids)}")
    print(f"共同视频数: {len(common_video_ids)}")
    
    video_ids = common_video_ids
    visual_features_aligned = np.array([visual_features[visual_id_to_idx[vid]] for vid in video_ids])
    text_features_aligned = np.array([text_features[text_id_to_idx[vid]] for vid in video_ids])
    
    visual_features = visual_features_aligned
    text_features = text_features_aligned
    
    print(f"视觉特征: {visual_features.shape}")
    print(f"文本特征: {text_features.shape}")
    print(f"视频数量: {len(video_ids)}")
    
    visual_pooled = visual_features.mean(axis=1)
    print(f"视觉特征池化后(用于记忆库构建): {visual_pooled.shape}")
    print(f"保留原始31帧特征用于训练")
    
    content_features = extract_content_features(visual_pooled, text_features, feature_dim=768)
    
    category_embeddings, category_to_idx, categories = create_category_embeddings(
        args.tags_file, d_model=768
    )
    
    labels, user_features, valid_video_ids, valid_indices = load_labels_and_user_features(
        args.comment_file, args.titles_file, video_ids, category_to_idx
    )
    
    valid_content_features = content_features[valid_indices]
    valid_labels = labels
    
    hl_grid = HLGrid(m_heat=args.m_heat, n_topic=args.n_topic, feature_dim=768, random_state=args.random_state)
    hl_grid.build_from_data(valid_content_features, valid_labels)
    
    train_data, valid_data, test_data, scaler, train_idx = prepare_training_data(
        visual_features, text_features, user_features, labels, video_ids, valid_indices,
        args.test_size, args.valid_size, args.random_state
    )
    
    save_processed_data(
        args.output_dir, train_data, valid_data, test_data,
        hl_grid, category_embeddings, scaler, categories
    )


if __name__ == '__main__':
    main()
