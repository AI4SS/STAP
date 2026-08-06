"""
新数据集 (V2) 用户特征工程
处理作者社交网络特征
与原有代码完全分离
"""
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_v2")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")

os.makedirs(FEATURES_DIR, exist_ok=True)


def get_feature_npy_path(data_path):
    """获取特征npy文件路径"""
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    return os.path.join(FEATURES_DIR, f'{base_name}_user_features.npy')


def save_user_features_to_npy(user_features, engagement_scores, influence_scores, data_path):
    """保存用户特征到npy文件"""
    npy_path = get_feature_npy_path(data_path)
    np.save(npy_path, {
        'user_features': np.array(user_features),
        'engagement_scores': np.array(engagement_scores),
        'influence_scores': np.array(influence_scores)
    })
    print(f"  [保存] 用户特征已保存到: {npy_path}")


def normalize_user_features(df, global_stats=None):
    """
    归一化用户特征
    
    新数据集的用户特征列:
    - author_follower_count
    - author_following_count
    - author_total_heart_count
    - author_total_video_count
    """
    user_feature_cols = [
        'author_follower_count',
        'author_following_count',
        'author_total_heart_count',
        'author_total_video_count'
    ]

    stats = {}
    
    for col in user_feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            df[col] = df[col].clip(lower=0)
            
            df[f'{col}_log'] = np.log1p(df[col].astype(float))

            if global_stats is not None and col in global_stats:
                mean = global_stats[col]['mean']
                std = global_stats[col]['std']
            else:
                mean = df[f'{col}_log'].mean()
                std = df[f'{col}_log'].std()
                stats[col] = {'mean': mean, 'std': std}
            
            if std > 0:
                df[f'{col}_normalized'] = (df[f'{col}_log'] - mean) / std
            else:
                df[f'{col}_normalized'] = 0

    if global_stats is None:
        return df, stats
    else:
        return df


def compute_user_engagement_score(row):
    """
    计算用户参与度分数
    基于关注者数和点赞数
    """
    follower_count = max(0, row.get('author_follower_count', 0) or 0)
    heart_count = max(0, row.get('author_total_heart_count', 0) or 0)
    video_count = max(1, row.get('author_total_video_count', 0) or 0)

    avg_hearts_per_video = heart_count / video_count

    score = np.log1p(follower_count) + 0.5 * np.log1p(avg_hearts_per_video)

    return score


def compute_user_influence_score(row):
    """
    计算用户影响力分数
    基于关注者关注比
    """
    follower_count = max(1, row.get('author_follower_count', 0) or 0)
    following_count = max(1, row.get('author_following_count', 0) or 0)

    follow_ratio = follower_count / following_count

    score = np.log1p(follow_ratio)

    return score


def extract_user_features_for_dataset(data_path, global_stats=None, engagement_stats=None, influence_stats=None):
    """
    为数据集提取用户特征
    """
    df = pd.read_pickle(data_path)

    print(f"\nProcessing {len(df)} items...")

    if global_stats is None:
        df, normalize_stats = normalize_user_features(df)
    else:
        df = normalize_user_features(df, global_stats)
        normalize_stats = None

    engagement_scores = []
    influence_scores = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        engagement = compute_user_engagement_score(row)
        influence = compute_user_influence_score(row)

        engagement_scores.append(engagement)
        influence_scores.append(influence)

    df['user_engagement_score'] = engagement_scores
    df['user_influence_score'] = influence_scores

    if engagement_stats is None:
        engagement_mean = np.mean(engagement_scores)
        engagement_std = np.std(engagement_scores)
        computed_engagement_stats = {'mean': engagement_mean, 'std': engagement_std}
    else:
        engagement_mean = engagement_stats['mean']
        engagement_std = engagement_stats['std']
        computed_engagement_stats = None

    if engagement_std > 0:
        df['user_engagement_score_normalized'] = (df['user_engagement_score'] - engagement_mean) / engagement_std
    else:
        df['user_engagement_score_normalized'] = 0

    if influence_stats is None:
        influence_mean = np.mean(influence_scores)
        influence_std = np.std(influence_scores)
        computed_influence_stats = {'mean': influence_mean, 'std': influence_std}
    else:
        influence_mean = influence_stats['mean']
        influence_std = influence_stats['std']
        computed_influence_stats = None

    if influence_std > 0:
        df['user_influence_score_normalized'] = (df['user_influence_score'] - influence_mean) / influence_std
    else:
        df['user_influence_score_normalized'] = 0

    user_feature_cols = [
        'author_follower_count_normalized',
        'author_following_count_normalized',
        'author_total_heart_count_normalized',
        'author_total_video_count_normalized',
        'user_engagement_score_normalized',
        'user_influence_score_normalized'
    ]

    user_features_list = []
    for idx, row in df.iterrows():
        user_feature = [row.get(col, 0) for col in user_feature_cols]
        user_features_list.append(user_feature)

    df['user_feature_vector'] = user_features_list

    print(f"\n[样本展示] 前3个用户特征样本:")
    for i in range(min(3, len(user_features_list))):
        feat = user_features_list[i]
        engagement = engagement_scores[i]
        influence = influence_scores[i]
        print(f"\n样本 {i+1}:")
        print(f"  特征向量 (6维): {[f'{x:.2f}' for x in feat]}")
        print(f"  参与度分数: {engagement:.4f}")
        print(f"  影响力分数: {influence:.4f}")

    save_user_features_to_npy(user_features_list, engagement_scores, influence_scores, data_path)

    df.to_pickle(data_path)
    print(f"\nUpdated {data_path} with user features")

    print("\n用户特征统计:")
    print(f"  用户参与度 - 均值: {engagement_mean:.4f}, 标准差: {engagement_std:.4f}")
    print(f"  用户影响力 - 均值: {influence_mean:.4f}, 标准差: {influence_std:.4f}")

    if global_stats is None:
        return {
            'normalize_stats': normalize_stats,
            'engagement_stats': computed_engagement_stats,
            'influence_stats': computed_influence_stats
        }
    return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract user features from V2 dataset')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test', 'all'],
                        help='Which dataset to process')

    args = parser.parse_args()

    if args.mode == 'all':
        train_path = os.path.join(OUTPUT_DIR, 'train.pkl')
        if os.path.exists(train_path):
            print(f"\n{'='*50}")
            print(f"Processing train set (computing global statistics)")
            print(f"{'='*50}")
            train_stats = extract_user_features_for_dataset(train_path)
            
            for mode in ['valid', 'test']:
                data_path = os.path.join(OUTPUT_DIR, f'{mode}.pkl')
                if os.path.exists(data_path):
                    print(f"\n{'='*50}")
                    print(f"Processing {mode} set (using train set statistics)")
                    print(f"{'='*50}")
                    extract_user_features_for_dataset(
                        data_path,
                        global_stats=train_stats['normalize_stats'],
                        engagement_stats=train_stats['engagement_stats'],
                        influence_stats=train_stats['influence_stats']
                    )
        else:
            print(f"Training data not found: {train_path}")
    else:
        data_path = os.path.join(OUTPUT_DIR, f'{args.mode}.pkl')
        if os.path.exists(data_path):
            if args.mode == 'train':
                extract_user_features_for_dataset(data_path)
            else:
                print("Warning: Processing validation/test set alone may cause data leakage.")
                extract_user_features_for_dataset(data_path)
        else:
            print(f"Data file not found: {data_path}")
