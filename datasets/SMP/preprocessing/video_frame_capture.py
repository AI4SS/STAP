"""
SMP-Video视频帧提取
从MP4视频中提取均匀分布的帧
遵循MMRA-main的视频处理思想
"""
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm


def validate_video_paths(df, video_base_dir):
    """
    验证视频文件路径是否正确

    Args:
        df: 数据DataFrame
        video_base_dir: 视频文件基础目录

    Returns:
        valid_count: 有效视频数量
        missing_count: 缺失视频数量
        sample_errors: 示例错误列表
    """
    valid_count = 0
    missing_count = 0
    sample_errors = []

    for idx, row in df.head(100).iterrows():  # 检查前100个样本
        video_path = row.get('video_path', '')
        if not video_path:
            missing_count += 1
            continue

        # 构建完整路径
        path_parts = video_path.replace('/', os.sep).split(os.sep)
        if len(path_parts) >= 2:
            video_path_clean = os.sep.join(path_parts[1:])
        else:
            video_path_clean = video_path
        full_video_path = os.path.join(video_base_dir, video_path_clean)
        full_video_path = os.path.normpath(full_video_path)

        if os.path.exists(full_video_path):
            valid_count += 1
        else:
            missing_count += 1
            if len(sample_errors) < 5:
                sample_errors.append(f"  {video_path} -> {full_video_path}")

    return valid_count, missing_count, sample_errors


def validate_frame_extraction(df):
    """
    验证帧提取结果

    Args:
        df: 数据DataFrame

    Returns:
        success_count: 成功提取的视频数量
        failed_count: 提取失败的视频数量
        avg_frames: 平均提取帧数
    """
    frame_counts = df['num_extracted_frames'].values
    success_count = (frame_counts > 0).sum()
    failed_count = (frame_counts == 0).sum()
    avg_frames = frame_counts[frame_counts > 0].mean() if success_count > 0 else 0

    return success_count, failed_count, avg_frames


def extract_frames_from_video(video_path, output_dir, num_frames=10, target_size=(224, 224)):
    """
    从视频中提取均匀分布的帧

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        num_frames: 提取的帧数
        target_size: 目标尺寸

    Returns:
        提取的帧文件路径列表
    """
    if not os.path.exists(video_path):
        return []

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames < num_frames:
        # 如果视频帧数不足，提取所有帧
        frame_indices = list(range(total_frames))
    else:
        # 均匀采样帧
        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frame_paths = []

    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if ret:
            # 调整大小
            frame = cv2.resize(frame, target_size)

            # 保存帧
            frame_filename = f"frame_{idx:04d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

    cap.release()

    return frame_paths


def extract_frames_for_dataset(data_path, video_base_dir, output_base_dir, num_frames=10):
    """
    为数据集中的所有视频提取帧

    Args:
        data_path: 数据pkl文件路径
        video_base_dir: 视频文件基础目录
        output_base_dir: 输出基础目录
        num_frames: 每个视频提取的帧数
    """
    df = pd.read_pickle(data_path)

    # === 自检：验证视频路径 ===
    print("\n[自检] 验证视频路径...")
    valid_count, missing_count, sample_errors = validate_video_paths(df, video_base_dir)
    print(f"  抽查样本: {valid_count + missing_count} 个")
    print(f"  有效路径: {valid_count} 个")
    print(f"  缺失路径: {missing_count} 个")
    if sample_errors:
        print(f"  示例错误:")
        for err in sample_errors:
            print(err)
    if missing_count > valid_count:
        print(f"  [警告] 大量视频文件缺失，请检查路径配置!")
        print(f"  video_base_dir: {video_base_dir}")

    # 初始化frame_paths列
    frame_paths_list = []
    num_extracted_frames_list = []

    print(f"\nProcessing {len(df)} videos...")

    missing_videos = 0
    success_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        item_id = row['item_id']
        video_path = row.get('video_path', '')

        if not video_path:
            frame_paths_list.append([])
            num_extracted_frames_list.append(0)
            missing_videos += 1
            continue

        # 构建完整视频路径
        # video_path可能是 "train/USER00001467/VIDEO00001122.mp4"
        # 实际文件在 dataset/USER00001467/VIDEO00001122.mp4
        # 需要去掉 train/valid/test 前缀
        path_parts = video_path.replace('/', os.sep).split(os.sep)
        # 跳过第一个部分（train/valid/test），只保留 USER.../VIDEO...
        if len(path_parts) >= 2:
            video_path_clean = os.sep.join(path_parts[1:])
        else:
            video_path_clean = video_path
        full_video_path = os.path.join(video_base_dir, video_path_clean)
        full_video_path = os.path.normpath(full_video_path)

        # 构建输出目录
        output_dir = os.path.join(output_base_dir, item_id)

        # 提取帧
        frame_paths = extract_frames_from_video(
            full_video_path,
            output_dir,
            num_frames=num_frames
        )

        if frame_paths:
            success_count += 1
        else:
            missing_videos += 1

        frame_paths_list.append(frame_paths)
        num_extracted_frames_list.append(len(frame_paths))

    # 批量更新DataFrame
    df['frame_paths'] = frame_paths_list
    df['num_extracted_frames'] = num_extracted_frames_list

    # === 自检：验证提取结果 ===
    print("\n[自检] 验证提取结果...")
    success_final, failed_final, avg_frames = validate_frame_extraction(df)
    print(f"  成功提取: {success_final} 个视频")
    print(f"  提取失败: {failed_final} 个视频")
    print(f"  平均帧数: {avg_frames:.1f} 帧/视频")
    if failed_final > len(df) * 0.1:
        print(f"  [警告] 失败率超过10%，请检查视频文件是否完整!")

    # 保存更新后的数据
    df.to_pickle(data_path)
    print(f"\nUpdated {data_path} with frame paths")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract frames from SMP-Video videos')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test', 'all'],
                        help='Which dataset to process')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing data pkl files')
    parser.add_argument('--video_dir', type=str, default='../dataset',
                        help='Base directory for video files')
    parser.add_argument('--output_dir', type=str, default='../data/raw_frames',
                        help='Base directory for extracted frames')
    parser.add_argument('--num_frames', type=int, default=10,
                        help='Number of frames to extract per video')

    args = parser.parse_args()

    if args.mode == 'all':
        for mode in ['train', 'valid', 'test']:
            data_path = os.path.join(args.data_dir, f'{mode}.pkl')
            if os.path.exists(data_path):
                print(f"\n{'='*50}")
                print(f"Processing {mode} set")
                print(f"{'='*50}")
                extract_frames_for_dataset(
                    data_path,
                    args.video_dir,
                    args.output_dir,
                    args.num_frames
                )
    else:
        data_path = os.path.join(args.data_dir, f'{args.mode}.pkl')
        if os.path.exists(data_path):
            extract_frames_for_dataset(
                data_path,
                args.video_dir,
                args.output_dir,
                args.num_frames
            )
        else:
            print(f"Data file not found: {data_path}")
