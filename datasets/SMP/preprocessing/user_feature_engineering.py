"""
SMP-Video用户特征工程
处理用户社交网络特征，这是SMP-Video数据集特有的创新点
遵循MMRA-main的特征处理思想
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def get_feature_npy_path(data_path):
    """获取特征npy文件路径"""
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    feature_dir = os.path.join(os.path.dirname(data_path), 'features')
    os.makedirs(feature_dir, exist_ok=True)
    return os.path.join(feature_dir, f'{base_name}_user_features.npy')


def save_user_features_to_npy(user_features, engagement_scores, influence_scores, data_path):
    """保存用户特征到npy文件"""
    npy_path = get_feature_npy_path(data_path)
    np.save(npy_path, {
        'user_features': np.array(user_features),
        'engagement_scores': np.array(engagement_scores),
        'influence_scores': np.array(influence_scores)
    })
    print(f"  [保存] 用户特征已保存到: {npy_path}")


def show_sample_user_features(user_features_list, engagement_scores, influence_scores, df, n=3):
    """展示样本用户特征"""
    print(f"\n[样本展示] 前{n}个用户特征样本:")
    print("-" * 60)
    for i in range(min(n, len(user_features_list))):
        feat = user_features_list[i]
        engagement = engagement_scores[i]
        influence = influence_scores[i]
        print(f"\n样本 {i+1}:")
        print(f"  用户ID: {df['uid'].iloc[i]}")
        print(f"  特征向量 (9维): {[f'{x:.2f}' for x in feat]}")
        print(f"  参与度分数: {engagement:.4f}")
        print(f"  影响力分数: {influence:.4f}")
    print("-" * 60)


def validate_user_features(df):
    """
    验证用户特征数据完整性

    Args:
        df: 数据DataFrame

    Returns:
        stats: 各特征的统计信息
    """
    user_feature_cols = [
        'user_following_count',
        'user_follower_count',
        'user_likes_count',
        'user_video_count',
        'user_digg_count',
        'user_heart_count',
        'user_friend_count'
    ]

    stats = {}
    for col in user_feature_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            non_negative = (df[col] >= 0).sum()
            stats[col] = {
                'non_null': non_null,
                'non_negative': non_negative,
                'total': len(df)
            }
    return stats


def validate_user_feature_vectors(df, expected_dim=9):
    """
    验证用户特征向量

    Args:
        df: 数据DataFrame
        expected_dim: 期望的特征维度

    Returns:
        valid_count: 有效特征向量数量
        invalid_count: 无效特征向量数量
    """
    valid_count = 0
    invalid_count = 0

    for feat in df['user_feature_vector']:
        if feat and len(feat) == expected_dim:
            valid_count += 1
        else:
            invalid_count += 1

    return valid_count, invalid_count


def normalize_user_features(df, global_stats=None):
    """
    归一化用户特征

    Args:
        df: 数据DataFrame
        global_stats: 可选的全局统计量字典 {col_name: {'mean': float, 'std': float}}
                     如果提供，则使用此统计量进行归一化（用于验证集和测试集）
                     如果为None，则计算当前数据集的统计量（用于训练集）

    Returns:
        df: 添加归一化用户特征的DataFrame
        stats: 计算得到的统计量字典（仅当global_stats为None时返回）
    """
    # 用户特征列
    user_feature_cols = [
        'user_following_count',
        'user_follower_count',
        'user_likes_count',
        'user_video_count',
        'user_digg_count',
        'user_heart_count',
        'user_friend_count'
    ]

    stats = {}
    
    # 对数变换 + 归一化
    for col in user_feature_cols:
        if col in df.columns:
            # 首先将负数转换为0，处理NaN值
            df[col] = df[col].fillna(0)
            df[col] = df[col].clip(lower=0)  # 将负数设为0
            
            # 对数变换 (加1避免log(0))
            df[f'{col}_log'] = np.log1p(df[col].astype(float))

            # Z-score归一化
            if global_stats is not None and col in global_stats:
                # 使用提供的全局统计量（验证集/测试集）
                mean = global_stats[col]['mean']
                std = global_stats[col]['std']
            else:
                # 计算当前数据集的统计量（训练集）
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
    综合考虑关注者数、点赞数、视频数等

    Args:
        row: 数据行

    Returns:
        score: 参与度分数
    """
    # 获取值并确保非负（处理匿名化后的负数）
    follower_count = max(0, row.get('user_follower_count', 0) or 0)
    likes_count = max(0, row.get('user_likes_count', 0) or 0)
    video_count = max(1, row.get('user_video_count', 0) or 0)  # 至少为1避免除零
    heart_count = max(0, row.get('user_heart_count', 0) or 0)

    # 平均每个视频的点赞数
    avg_likes_per_video = likes_count / video_count

    # 平均每个视频的心心数
    avg_hearts_per_video = heart_count / video_count

    # 综合分数 (对数变换) - 现在输入保证非负
    score = np.log1p(follower_count) + 0.5 * np.log1p(avg_likes_per_video) + 0.3 * np.log1p(avg_hearts_per_video)

    return score


def compute_user_influence_score(row):
    """
    计算用户影响力分数
    基于关注者关注比和总互动量

    Args:
        row: 数据行

    Returns:
        score: 影响力分数
    """
    # 获取值并确保非负（处理匿名化后的负数）
    follower_count = max(1, row.get('user_follower_count', 0) or 0)  # 至少为1
    following_count = max(1, row.get('user_following_count', 0) or 0)  # 至少为1

    # 关注比率
    follow_ratio = follower_count / following_count

    # 总互动（确保非负）
    likes = max(0, row.get('user_likes_count', 0) or 0)
    hearts = max(0, row.get('user_heart_count', 0) or 0)
    diggs = max(0, row.get('user_digg_count', 0) or 0)
    total_engagement = likes + hearts + diggs

    # 影响力分数 (对数变换) - 现在输入保证非负
    score = np.log1p(follow_ratio) + 0.3 * np.log1p(total_engagement)

    return score


def extract_user_features_for_dataset(data_path, global_stats=None, engagement_stats=None, influence_stats=None):
    """
    为数据集提取用户特征

    Args:
        data_path: 数据pkl文件路径
        global_stats: 可选的全局归一化统计量（来自训练集）
        engagement_stats: 可选的参与度分数统计量 {'mean': float, 'std': float}
        influence_stats: 可选的影响力分数统计量 {'mean': float, 'std': float}

    Returns:
        stats_dict: 如果 global_stats 为 None，返回计算得到的统计量字典
    """
    df = pd.read_pickle(data_path)

    # === 自检：验证输入用户特征 ===
    print("\n[自检] 验证用户特征数据...")
    feature_stats = validate_user_features(df)
    for col, stats in feature_stats.items():
        non_null_rate = stats['non_null'] / stats['total'] * 100
        non_negative_rate = stats['non_negative'] / stats['total'] * 100
        print(f"  {col}: {stats['non_null']}/{stats['total']} 非空 ({non_null_rate:.1f}%), "
              f"{stats['non_negative']}/{stats['total']} 非负 ({non_negative_rate:.1f}%)")

    print(f"\nProcessing {len(df)} items...")

    # 归一化用户特征
    if global_stats is None:
        # 训练集：计算统计量
        df, normalize_stats = normalize_user_features(df)
    else:
        # 验证集/测试集：使用训练集的统计量
        df = normalize_user_features(df, global_stats)
        normalize_stats = None

    # 计算用户参与度和影响力分数
    engagement_scores = []
    influence_scores = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        engagement = compute_user_engagement_score(row)
        influence = compute_user_influence_score(row)

        engagement_scores.append(engagement)
        influence_scores.append(influence)

    # 添加到DataFrame
    df['user_engagement_score'] = engagement_scores
    df['user_influence_score'] = influence_scores

    # 归一化分数
    if engagement_stats is None:
        # 训练集：计算统计量
        engagement_mean = np.mean(engagement_scores)
        engagement_std = np.std(engagement_scores)
        computed_engagement_stats = {'mean': engagement_mean, 'std': engagement_std}
    else:
        # 验证集/测试集：使用训练集的统计量
        engagement_mean = engagement_stats['mean']
        engagement_std = engagement_stats['std']
        computed_engagement_stats = None

    if engagement_std > 0:
        df['user_engagement_score_normalized'] = (df['user_engagement_score'] - engagement_mean) / engagement_std
    else:
        df['user_engagement_score_normalized'] = 0

    if influence_stats is None:
        # 训练集：计算统计量
        influence_mean = np.mean(influence_scores)
        influence_std = np.std(influence_scores)
        computed_influence_stats = {'mean': influence_mean, 'std': influence_std}
    else:
        # 验证集/测试集：使用训练集的统计量
        influence_mean = influence_stats['mean']
        influence_std = influence_stats['std']
        computed_influence_stats = None

    if influence_std > 0:
        df['user_influence_score_normalized'] = (df['user_influence_score'] - influence_mean) / influence_std
    else:
        df['user_influence_score_normalized'] = 0

    # 创建用户特征向量（用于模型输入）
    # 组合归一化后的特征
    user_feature_cols = [
        'user_following_count_normalized',
        'user_follower_count_normalized',
        'user_likes_count_normalized',
        'user_video_count_normalized',
        'user_digg_count_normalized',
        'user_heart_count_normalized',
        'user_friend_count_normalized',
        'user_engagement_score_normalized',
        'user_influence_score_normalized'
    ]

    user_features_list = []
    for idx, row in df.iterrows():
        user_feature = [row.get(col, 0) for col in user_feature_cols]
        user_features_list.append(user_feature)

    df['user_feature_vector'] = user_features_list

    # 展示样本特征
    show_sample_user_features(user_features_list, engagement_scores, influence_scores, df, n=3)

    # === 自检：验证输出特征向量 ===
    print("\n[自检] 验证用户特征向量...")
    valid_feat, invalid_feat = validate_user_feature_vectors(df, expected_dim=9)
    print(f"  有效特征向量 (9维): {valid_feat} 个")
    print(f"  无效特征向量: {invalid_feat} 个")

    # 保存npy文件
    save_user_features_to_npy(user_features_list, engagement_scores, influence_scores, data_path)

    # 保存更新后的数据
    df.to_pickle(data_path)
    print(f"\nUpdated {data_path} with user features")

    # 打印统计信息
    print("\n用户特征统计:")
    print(f"  用户参与度 - 均值: {engagement_mean:.4f}, 标准差: {engagement_std:.4f}")
    print(f"  用户影响力 - 均值: {influence_mean:.4f}, 标准差: {influence_std:.4f}")

    # 返回统计量（仅训练集）
    if global_stats is None:
        return {
            'normalize_stats': normalize_stats,
            'engagement_stats': computed_engagement_stats,
            'influence_stats': computed_influence_stats
        }
    return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract user features from SMP-Video')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test', 'all'],
                        help='Which dataset to process')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing data pkl files')

    args = parser.parse_args()

    if args.mode == 'all':
        # 首先处理训练集，获取全局统计量
        train_path = os.path.join(args.data_dir, 'train.pkl')
        if os.path.exists(train_path):
            print(f"\n{'='*50}")
            print(f"Processing train set (computing global statistics)")
            print(f"{'='*50}")
            train_stats = extract_user_features_for_dataset(train_path)
            
            # 使用训练集的统计量处理验证集和测试集
            for mode in ['valid', 'test']:
                data_path = os.path.join(args.data_dir, f'{mode}.pkl')
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
        data_path = os.path.join(args.data_dir, f'{args.mode}.pkl')
        if os.path.exists(data_path):
            if args.mode == 'train':
                extract_user_features_for_dataset(data_path)
            else:
                # 对于单独处理验证集或测试集的情况，先检查并加载训练集统计量
                print("Warning: Processing validation/test set alone may cause data leakage.")
                print("Recommendation: Use --mode all to ensure proper normalization.")
                extract_user_features_for_dataset(data_path)
        else:
            print(f"Data file not found: {data_path}")
