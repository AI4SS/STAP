"""
SMP-Video Dataset Split
Train:Validation:Test = 8:1:1

支持两种划分方式:
1. chronological: 按时间顺序划分 (默认，避免数据泄露)
2. random: 随机划分

遵循MMRA-main的数据集划分思想
"""
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import os


def load_smp_data(data_dir):
    """
    加载SMP-Video数据集的四个JSONL文件
    """
    # 加载数据
    posts_data = []
    videos_data = []
    popularity_data = []
    users_data = []

    # 读取posts数据
    posts_path = os.path.join(data_dir, 'SMP-Video_anonymized_posts_train.jsonl')
    with open(posts_path, 'r', encoding='utf-8') as f:
        for line in f:
            posts_data.append(json.loads(line.strip()))

    # 读取videos数据
    videos_path = os.path.join(data_dir, 'SMP-Video_anonymized_videos_train.jsonl')
    with open(videos_path, 'r', encoding='utf-8') as f:
        for line in f:
            videos_data.append(json.loads(line.strip()))

    # 读取popularity数据
    popularity_path = os.path.join(data_dir, 'SMP-Video_anonymized_popularity_train.jsonl')
    with open(popularity_path, 'r', encoding='utf-8') as f:
        for line in f:
            popularity_data.append(json.loads(line.strip()))

    # 读取users数据
    users_path = os.path.join(data_dir, 'SMP-Video_anonymized_users_train.jsonl')
    if os.path.exists(users_path):
        with open(users_path, 'r', encoding='utf-8') as f:
            for line in f:
                users_data.append(json.loads(line.strip()))

    return posts_data, videos_data, popularity_data, users_data


def merge_data(posts_data, videos_data, popularity_data, users_data):
    """
    合并所有数据，以pid为键进行关联
    """
    # 创建用户字典
    users_dict = {u['uid']: u for u in users_data}

    # 创建视频字典
    videos_dict = {v['pid']: v for v in videos_data}

    # 创建popularity字典
    popularity_dict = {p['pid']: p['popularity'] for p in popularity_data}

    # 合并数据
    merged_data = []
    for post in posts_data:
        pid = post['pid']
        if pid in popularity_dict:
            # 修复post_content：将逗号替换为空格，形成完整句子
            post_content = post.get('post_content', '')
            post_content = post_content.replace(',', ' ')

            item = {
                'item_id': pid,  # 使用pid作为item_id
                'pid': pid,
                'uid': post['uid'],
                'post_content': post_content,
                'post_location': post.get('post_location', ''),
                'post_suggested_words': post.get('post_suggested_words', []),
                'post_text_language': post.get('post_text_language', 'en'),
                'video_path': post.get('video_path', ''),
                'post_time': post.get('post_time', ''),  # 保留时间字段用于时序划分
                'label': popularity_dict[pid],  # 原始popularity分数
            }

            # 添加视频信息
            if pid in videos_dict:
                video_info = videos_dict[pid]
                item.update({
                    'vid': video_info.get('vid', ''),
                    'video_height': video_info.get('video_height', 0),
                    'video_width': video_info.get('video_width', 0),
                    'video_duration': video_info.get('video_duration', 0),
                    'video_ratio': video_info.get('video_ratio', ''),
                    'video_format': video_info.get('video_format', 'mp4'),
                    'music_title': video_info.get('music_title', ''),
                    'music_duration': video_info.get('music_duration', 0),
                })

            # 添加用户信息
            uid = post['uid']
            if uid in users_dict:
                user_info = users_dict[uid]
                item.update({
                    'user_following_count': user_info.get('user_following_count', 0),
                    'user_follower_count': user_info.get('user_follower_count', 0),
                    'user_likes_count': user_info.get('user_likes_count', 0),
                    'user_video_count': user_info.get('user_video_count', 0),
                    'user_digg_count': user_info.get('user_digg_count', 0),
                    'user_heart_count': user_info.get('user_heart_count', 0),
                    'user_friend_count': user_info.get('user_friend_count', 0),
                })
            else:
                # 用户信息缺失时填0
                item.update({
                    'user_following_count': 0,
                    'user_follower_count': 0,
                    'user_likes_count': 0,
                    'user_video_count': 0,
                    'user_digg_count': 0,
                    'user_heart_count': 0,
                    'user_friend_count': 0,
                })

            merged_data.append(item)

    return merged_data


def split_dataset_chronological(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    按时间顺序划分数据集（避免数据泄露）

    Args:
        df: 数据DataFrame (必须包含post_time列)
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例

    Returns:
        train_df, val_df, test_df
    """
    # 按时间排序
    df_sorted = df.sort_values('post_time').reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].reset_index(drop=True)
    val_df = df_sorted.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df_sorted.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def split_dataset_random(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    随机划分数据集

    Args:
        df: 数据DataFrame
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子

    Returns:
        train_df, val_df, test_df
    """
    # 第一次划分：训练集 和 临时集
    train_data, temp_data = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=random_state
    )

    # 第二次划分：验证集 和 测试集
    val_data, test_data = train_test_split(
        temp_data,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=random_state
    )

    return train_data, val_data, test_data


if __name__ == '__main__':
    # 数据目录
    data_dir = '../data/train-orgin'
    output_dir = '../data'

    # 划分方式: 'chronological' (时序) 或 'random' (随机)
    SPLIT_MODE = 'chronological'  # 默认使用时序划分

    print("Loading SMP-Video data...")
    posts_data, videos_data, popularity_data, users_data = load_smp_data(data_dir)

    print(f"Loaded {len(posts_data)} posts")
    print(f"Loaded {len(videos_data)} videos")
    print(f"Loaded {len(popularity_data)} popularity records")
    print(f"Loaded {len(users_data)} users")

    print("\nMerging data...")
    merged_data = merge_data(posts_data, videos_data, popularity_data, users_data)
    print(f"Merged {len(merged_data)} items")

    # 创建DataFrame
    df = pd.DataFrame(merged_data)

    # 直接使用原始 label 值，不进行 log2 转换
    print("\nLabel statistics (original values):")
    print(f"  Min: {df['label'].min():.4f}")
    print(f"  Max: {df['label'].max():.4f}")
    print(f"  Mean: {df['label'].mean():.4f}")
    print(f"  Std: {df['label'].std():.4f}")

    # 自检：验证post_content已去除逗号
    print("\n[自检] 验证post_content格式...")
    has_commas = 0
    for content in df['post_content']:
        if ',' in content:
            has_commas += 1
    print(f"  包含逗号的post_content: {has_commas}/{len(df)}")
    if has_commas > 0:
        print(f"  [警告] 发现{has_commas}条post_content仍包含逗号!")
    else:
        print(f"  [通过] 所有post_content的逗号已去除，转为空格分隔")

    # 保存完整数据
    df.to_pickle(os.path.join(output_dir, 'data.pkl'))
    print(f"\nSaved full data to {os.path.join(output_dir, 'data.pkl')}")

    # 划分数据集 (8:1:1)
    if SPLIT_MODE == 'chronological':
        print("\nSplitting dataset chronologically (train:val:test = 8:1:1)...")
        print("This ensures no data leakage by using temporal split!")
        train_df, val_df, test_df = split_dataset_chronological(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

        # 打印时间范围
        print("\nTime ranges:")
        print(f"  Train: {train_df['post_time'].min()} to {train_df['post_time'].max()}")
        print(f"  Valid: {val_df['post_time'].min()} to {val_df['post_time'].max()}")
        print(f"  Test:  {test_df['post_time'].min()} to {test_df['post_time'].max()}")
    else:
        print("\nSplitting dataset randomly (train:val:test = 8:1:1)...")
        train_df, val_df, test_df = split_dataset_random(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    print(f"\nTrain set: {len(train_df)} items")
    print(f"Validation set: {len(val_df)} items")
    print(f"Test set: {len(test_df)} items")

    # 保存划分后的数据
    train_df.to_pickle(os.path.join(output_dir, 'train.pkl'))
    val_df.to_pickle(os.path.join(output_dir, 'valid.pkl'))
    test_df.to_pickle(os.path.join(output_dir, 'test.pkl'))

    print(f"\nSaved train.pkl to {output_dir}")
    print(f"Saved valid.pkl to {output_dir}")
    print(f"Saved test.pkl to {output_dir}")

    print("\nDataset split completed!")
