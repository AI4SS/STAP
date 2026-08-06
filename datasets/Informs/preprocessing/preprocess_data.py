"""
新数据集 (V2) 数据预处理
处理 /root/dataset 中的数据
与原有代码完全分离
"""
import os
import sys
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime
import holidays
import re
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

BASE_DIR = "/root/dataset"
TRAIN_EXCEL = os.path.join(BASE_DIR, "training.xlsx")
TEST_EXCEL = os.path.join(BASE_DIR, "testing.xlsx")
TEST_LABELS_EXCEL = os.path.join(BASE_DIR, "testing_set_with_labels.xlsx")

TRAIN_VIDEO_DIR = os.path.join(BASE_DIR, "train_converted_mp4")
TEST_VIDEO_DIR = os.path.join(BASE_DIR, "test_converted_mp4")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_v2")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "raw_frames")


def extract_tags(description):
    """从视频描述中提取 hashtag 和 mention"""
    description = str(description)
    hashtags = re.findall(r'#(\w+)', description)
    mentions = re.findall(r'@(\w+)', description)
    return hashtags, mentions


def total_tag_frequency(tags_list, frequency_dict):
    """计算标签列表的总频率权重"""
    return sum(frequency_dict[tag] for tag in tags_list if tag in frequency_dict)


def extract_date_components_and_check_holidays(row):
    """提取时间组件并检查美国节假日"""
    try:
        date = datetime.utcfromtimestamp(int(row['video_create_date']))
    except:
        date = datetime.now()

    is_holiday = date in holidays.UnitedStates()

    if 9 <= date.hour < 18:
        time_period = 'Work Time'
    elif 18 <= date.hour < 23:
        time_period = 'Leisure Time'
    else:
        time_period = 'Sleep Time'

    return pd.Series({
        'year': date.year,
        'month': date.month,
        'day': date.day,
        'hour': date.hour,
        'is_holiday': is_holiday,
        'time_period': time_period
    })


def get_video_details(video_path):
    """获取视频详细信息: 时长, 帧数, 帧率, 宽, 高 (使用OpenCV)"""
    try:
        if not os.path.exists(video_path):
            return None, None, None, None, None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, None, None, None

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if frame_rate > 0:
            duration = total_frames / frame_rate
        else:
            duration = 0
        
        cap.release()

        return duration, total_frames, frame_rate, width, height
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return None, None, None, None, None


def extract_frames_from_video(video_path, output_dir, num_frames=10, target_size=(224, 224)):
    """从视频中提取均匀分布的帧"""
    if not os.path.exists(video_path):
        return []

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frame_paths = []

    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, target_size)
            frame_filename = f"frame_{idx:04d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

    cap.release()
    return frame_paths


def process_data():
    """主处理函数"""
    print("=" * 60)
    print("新数据集 (V2) 数据预处理")
    print("=" * 60)

    print("\n[Step 1] 加载Excel数据...")
    df_train = pd.read_excel(TRAIN_EXCEL, engine='openpyxl')
    df_test = pd.read_excel(TEST_EXCEL, engine='openpyxl')
    df_test_labels = pd.read_excel(TEST_LABELS_EXCEL, engine='openpyxl')

    df_train['video_id'] = df_train['video_id'].astype(str).str.replace("'", "")
    df_test['video_id'] = df_test['video_id'].astype(str).str.replace("'", "")
    df_test_labels['video_id'] = df_test_labels['video_id'].astype(str).str.replace("'", "")

    print(f"  训练集: {len(df_train)} 条")
    print(f"  测试集: {len(df_test)} 条")
    print(f"  测试集标签: {len(df_test_labels)} 条")

    print("\n[Step 2] 提取视频物理特征...")
    df_train['video_details'] = df_train['video_id'].apply(
        lambda x: get_video_details(os.path.join(TRAIN_VIDEO_DIR, f'{x}.mp4'))
    )
    df_train[['video_duration', 'total_frames', 'frame_rate', 'width', 'height']] = \
        pd.DataFrame(df_train['video_details'].tolist(), index=df_train.index)

    df_test['video_details'] = df_test['video_id'].apply(
        lambda x: get_video_details(os.path.join(TEST_VIDEO_DIR, f'{x}.mp4'))
    )
    df_test[['video_duration', 'total_frames', 'frame_rate', 'width', 'height']] = \
        pd.DataFrame(df_test['video_details'].tolist(), index=df_test.index)

    print("\n[Step 3] 填充缺失视频特征...")
    columns_to_fill = ['video_duration', 'total_frames', 'frame_rate', 'width', 'height']

    for df in [df_train, df_test]:
        for column in columns_to_fill:
            df[column] = df.groupby('author_id')[column].transform(lambda x: x.fillna(x.median()))
            df[column] = df[column].fillna(df[column].median())

    print("\n[Step 4] 归一化视频创建时间...")
    min_date = min(df_train['video_create_date'].min(), df_test['video_create_date'].min())
    max_date = max(df_train['video_create_date'].max(), df_test['video_create_date'].max())

    df_train['video_create_date_normalized'] = (df_train['video_create_date'] - min_date) / (max_date - min_date)
    df_test['video_create_date_normalized'] = (df_test['video_create_date'] - min_date) / (max_date - min_date)

    print("\n[Step 5] 提取Hashtags和Mentions...")
    df_train['hashtags'], df_train['mentions'] = zip(*df_train['video_description'].apply(extract_tags))
    df_test['hashtags'], df_test['mentions'] = zip(*df_test['video_description'].apply(extract_tags))

    all_hashtags = [tag for sublist in df_train['hashtags'].tolist() + df_test['hashtags'].tolist() for tag in sublist]
    all_mentions = [tag for sublist in df_train['mentions'].tolist() + df_test['mentions'].tolist() for tag in sublist]

    hashtag_frequency = Counter(all_hashtags)
    mention_frequency = Counter(all_mentions)

    for df in [df_train, df_test]:
        df['total_hashtag_frequency'] = df['hashtags'].apply(lambda tags: total_tag_frequency(tags, hashtag_frequency))
        df['total_mention_frequency'] = df['mentions'].apply(lambda tags: total_tag_frequency(tags, mention_frequency))
        df['hashtag_count'] = df['hashtags'].apply(len)
        df['mention_count'] = df['mentions'].apply(len)

    print("\n[Step 6] 处理时间组件 (节假日/工作时间)...")
    df_train[['year', 'month', 'day', 'hour', 'is_holiday', 'time_period']] = \
        df_train.apply(extract_date_components_and_check_holidays, axis=1)
    df_test[['year', 'month', 'day', 'hour', 'is_holiday', 'time_period']] = \
        df_test.apply(extract_date_components_and_check_holidays, axis=1)

    print("\n[Step 7] Log变换...")
    columns_to_log = [
        'author_follower_count', 'author_following_count',
        'author_total_heart_count', 'author_total_video_count'
    ]

    for column in columns_to_log:
        if column in df_train.columns:
            df_train[column] = np.log1p(df_train[column])
        if column in df_test.columns:
            df_test[column] = np.log1p(df_test[column])

    print("\n[Step 8] 提取视频帧...")
    train_video_ids = set(df_train['video_id'].tolist())
    train_video_files = set([f.replace('.mp4', '') for f in os.listdir(TRAIN_VIDEO_DIR) if f.endswith('.mp4')])
    train_missing = train_video_ids - train_video_files
    print(f"  训练集视频: {len(train_video_ids)} 个")
    print(f"  实际存在: {len(train_video_files)} 个")
    print(f"  缺失视频: {len(train_missing)} 个")

    test_video_ids = set(df_test['video_id'].tolist())
    test_video_files = set([f.replace('.mp4', '') for f in os.listdir(TEST_VIDEO_DIR) if f.endswith('.mp4')])
    test_missing = test_video_ids - test_video_files
    print(f"  测试集视频: {len(test_video_ids)} 个")
    print(f"  实际存在: {len(test_video_files)} 个")
    print(f"  缺失视频: {len(test_missing)} 个")

    frame_paths_train = {}
    frame_paths_test = {}

    print("\n  提取训练集帧...")
    for video_id in tqdm(df_train['video_id'].tolist()):
        video_path = os.path.join(TRAIN_VIDEO_DIR, f'{video_id}.mp4')
        output_dir = os.path.join(FRAMES_DIR, 'train', video_id)
        if os.path.exists(video_path):
            frame_paths_train[video_id] = extract_frames_from_video(video_path, output_dir, num_frames=100)
        else:
            frame_paths_train[video_id] = []

    print("\n  提取测试集帧...")
    for video_id in tqdm(df_test['video_id'].tolist()):
        video_path = os.path.join(TEST_VIDEO_DIR, f'{video_id}.mp4')
        output_dir = os.path.join(FRAMES_DIR, 'test', video_id)
        if os.path.exists(video_path):
            frame_paths_test[video_id] = extract_frames_from_video(video_path, output_dir, num_frames=100)
        else:
            frame_paths_test[video_id] = []

    df_train['frame_paths'] = df_train['video_id'].map(frame_paths_train)
    df_test['frame_paths'] = df_test['video_id'].map(frame_paths_test)

    df_train['num_extracted_frames'] = df_train['frame_paths'].apply(len)
    df_test['num_extracted_frames'] = df_test['frame_paths'].apply(len)

    print("\n[Step 9] 合并测试集标签...")
    df_test = df_test.merge(
        df_test_labels[['identifier', 'video_comment_count', 'video_heart_count', 'video_play_count', 'video_share_count']],
        on='identifier',
        how='left'
    )

    print("\n[Step 10] 过滤缺失视频...")
    df_train_valid = df_train[df_train['num_extracted_frames'] > 0].reset_index(drop=True)
    df_test_valid = df_test[df_test['num_extracted_frames'] > 0].reset_index(drop=True)

    print(f"  有效训练集: {len(df_train_valid)} 条 (过滤了 {len(df_train) - len(df_train_valid)} 条)")
    print(f"  有效测试集: {len(df_test_valid)} 条 (过滤了 {len(df_test) - len(df_test_valid)} 条)")

    print("\n[Step 11] 划分训练集和验证集 (8:2)...")
    n_train = len(df_train_valid)
    train_end = int(n_train * 0.8)

    df_train_final = df_train_valid.iloc[:train_end].reset_index(drop=True)
    df_valid_final = df_train_valid.iloc[train_end:].reset_index(drop=True)
    df_test_final = df_test_valid.reset_index(drop=True)

    print(f"  训练集: {len(df_train_final)} 条")
    print(f"  验证集: {len(df_valid_final)} 条")
    print(f"  测试集: {len(df_test_final)} 条")

    print("\n[Step 11.5] 对标签进行Log变换...")
    target_columns = ['video_comment_count', 'video_heart_count', 'video_play_count', 'video_share_count']
    for col in target_columns:
        if col in df_train_final.columns:
            df_train_final[col] = np.log1p(df_train_final[col])
        if col in df_valid_final.columns:
            df_valid_final[col] = np.log1p(df_valid_final[col])
        if col in df_test_final.columns:
            df_test_final[col] = np.log1p(df_test_final[col])
    print("  已对标签进行log1p变换")

    print("\n[Step 12] 保存数据...")
    columns_to_drop = ['video_description', 'video_definition', 'video_format', 'hashtags', 'mentions', 'video_details']

    df_train_final.drop(columns=[c for c in columns_to_drop if c in df_train_final.columns], inplace=True)
    df_valid_final.drop(columns=[c for c in columns_to_drop if c in df_valid_final.columns], inplace=True)
    df_test_final.drop(columns=[c for c in columns_to_drop if c in df_test_final.columns], inplace=True)

    df_train_final.to_pickle(os.path.join(OUTPUT_DIR, 'train.pkl'))
    df_valid_final.to_pickle(os.path.join(OUTPUT_DIR, 'valid.pkl'))
    df_test_final.to_pickle(os.path.join(OUTPUT_DIR, 'test.pkl'))

    print(f"\n保存完成!")
    print(f"  {os.path.join(OUTPUT_DIR, 'train.pkl')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'valid.pkl')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'test.pkl')}")

    print("\n[统计信息]")
    print(f"  训练集标签 (video_play_count): min={df_train_final['video_play_count'].min()}, max={df_train_final['video_play_count'].max()}, mean={df_train_final['video_play_count'].mean():.2f}")
    print(f"  验证集标签 (video_play_count): min={df_valid_final['video_play_count'].min()}, max={df_valid_final['video_play_count'].max()}, mean={df_valid_final['video_play_count'].mean():.2f}")
    print(f"  测试集标签 (video_play_count): min={df_test_final['video_play_count'].min()}, max={df_test_final['video_play_count'].max()}, mean={df_test_final['video_play_count'].mean():.2f}")

    return df_train_final, df_valid_final, df_test_final


if __name__ == '__main__':
    process_data()
