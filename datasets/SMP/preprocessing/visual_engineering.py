"""
SMP-Video视觉特征提取
使用ViT (Vision Transformer) 提取视频帧的视觉特征
遵循MMRA-main的视觉处理思想
"""
import os
import shutil
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
from PIL import Image


def get_cache_path(data_path, suffix='_visual_features.npz'):
    """获取缓存文件路径"""
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    cache_dir = os.path.join(os.path.dirname(data_path), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'{base_name}{suffix}')


def get_feature_npy_path(data_path, feature_type='cls'):
    """获取特征npy文件路径"""
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    feature_dir = os.path.join(os.path.dirname(data_path), 'features')
    os.makedirs(feature_dir, exist_ok=True)
    return os.path.join(feature_dir, f'{base_name}_visual_{feature_type}_features.npy')


def save_visual_features_to_npy(cls_features, mean_features, data_path):
    """保存视觉特征到npy文件"""
    cls_npy_path = get_feature_npy_path(data_path, 'cls')
    mean_npy_path = get_feature_npy_path(data_path, 'mean')

    np.save(cls_npy_path, np.array(cls_features, dtype=object))
    np.save(mean_npy_path, np.array(mean_features, dtype=object))
    print(f"  [保存] CLS特征已保存到: {cls_npy_path}")
    print(f"  [保存] Mean特征已保存到: {mean_npy_path}")


def load_cached_visual_features(cache_path):
    """从缓存加载视觉特征"""
    if os.path.exists(cache_path):
        print(f"  [缓存] 发现缓存文件: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data['cls_features'].tolist(), data['mean_features'].tolist()
    return None, None


def save_cached_visual_features(cache_path, cls_features, mean_features):
    """保存视觉特征到缓存"""
    np.savez(cache_path,
             cls_features=np.array(cls_features, dtype=object),
             mean_features=np.array(mean_features, dtype=object))
    print(f"  [缓存] 视觉特征已缓存到: {cache_path}")


def validate_frame_paths(df):
    """
    验证帧路径是否有效

    Args:
        df: 数据DataFrame

    Returns:
        has_frames: 有帧的数量
        no_frames: 无帧的数量
        total_frames: 总帧数
    """
    has_frames = 0
    no_frames = 0
    total_frames = 0

    for fps in df['frame_paths']:
        if fps and len(fps) > 0:
            has_frames += 1
            total_frames += len(fps)
        else:
            no_frames += 1

    return has_frames, no_frames, total_frames


def validate_visual_features(df, expected_frames=10):
    """
    验证视觉特征提取结果

    Args:
        df: 数据DataFrame
        expected_frames: 期望的帧数

    Returns:
        valid_cls: 有效CLS特征数量
        valid_mean: 有效Mean特征数量
        avg_cls_norm: 平均CLS范数
        avg_mean_norm: 平均Mean范数
    """
    valid_cls = 0
    valid_mean = 0
    cls_norms = []
    mean_norms = []

    for cls_feat, mean_feat in zip(df['visual_feature_embedding_cls'], df['visual_feature_embedding_mean']):
        if cls_feat and len(cls_feat) == expected_frames and all(len(f) == 768 for f in cls_feat):
            valid_cls += 1
            for f in cls_feat:
                cls_norms.append(np.linalg.norm(f))
        if mean_feat and len(mean_feat) == expected_frames and all(len(f) == 768 for f in mean_feat):
            valid_mean += 1
            for f in mean_feat:
                mean_norms.append(np.linalg.norm(f))

    avg_cls_norm = np.mean(cls_norms) if cls_norms else 0
    avg_mean_norm = np.mean(mean_norms) if mean_norms else 0
    return valid_cls, valid_mean, avg_cls_norm, avg_mean_norm


def load_vit_model(model_name='google/vit-base-patch16-224-in21k', device=None, local_only=False):
    """
    加载ViT模型和处理器

    Args:
        model_name: 预训练模型名称
        device: 指定设备 ('cuda', 'cpu', 或 None 自动检测)
        local_only: 是否仅使用本地缓存（离线模式）

    Returns:
        processor, model, device
    """
    print(f"Loading ViT model: {model_name}")

    # 自动检测设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, using CPU")
        device = 'cpu'

    # 设置加载参数
    load_kwargs = {}
    if local_only:
        load_kwargs['local_files_only'] = True
        print("[离线模式] 仅使用本地缓存")

    try:
        # 尝试加载处理器和模型
        processor = ViTImageProcessor.from_pretrained(model_name, **load_kwargs)
        model = ViTModel.from_pretrained(model_name, **load_kwargs)
    except OSError as e:
        if local_only:
            print(f"\n[错误] 离线模式下找不到本地模型: {model_name}")
            print(f"请先在有网络的环境下运行一次以下载模型，或者:")
            print(f"  1. 手动下载模型到 ~/.cache/huggingface/hub/")
            print(f"  2. 或使用 local_only=False 参数尝试在线下载")
            raise

        # 如果不是离线模式，重试一次（可能是网络问题）
        print(f"[警告] 模型加载失败: {e}")
        print("[重试] 尝试重新加载...")
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTModel.from_pretrained(model_name)

    model = model.to(device)
    model.eval()

    if device == 'cuda':
        print(f"Model loaded on CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Model loaded on CPU")

    return processor, model, device


def extract_frame_features(processor, model, frame_path, device):
    """
    从单帧图像提取视觉特征

    Args:
        processor: ViT图像处理器
        model: ViT模型
        frame_path: 帧图像路径
        device: 运行设备

    Returns:
        cls_token: CLS token特征向量 (768维)
        mean_token: 所有token的平均特征
    """
    try:
        image = Image.open(frame_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # CLS token (第一维)
        cls_token = outputs.last_hidden_state[0, 0, :].cpu().numpy()

        # Mean token (所有token的平均)
        mean_token = outputs.last_hidden_state[0, :, :].mean(dim=0).cpu().numpy()

        return cls_token.tolist(), mean_token.tolist()

    except Exception as e:
        print(f"Error processing {frame_path}: {e}")
        # 返回零向量
        return [0.0] * 768, [0.0] * 768


def extract_video_features(processor, model, frame_paths, device, max_frames=10):
    """
    从视频的所有帧提取特征

    Args:
        processor: ViT图像处理器
        model: ViT模型
        frame_paths: 帧路径列表
        device: 运行设备
        max_frames: 最大帧数

    Returns:
        visual_features_cls: CLS token特征列表
        visual_features_mean: Mean token特征列表
    """
    visual_features_cls = []
    visual_features_mean = []

    for frame_path in frame_paths[:max_frames]:
        cls_token, mean_token = extract_frame_features(processor, model, frame_path, device)
        visual_features_cls.append(cls_token)
        visual_features_mean.append(mean_token)

    # 填充到max_frames
    while len(visual_features_cls) < max_frames:
        visual_features_cls.append([0.0] * 768)
        visual_features_mean.append([0.0] * 768)

    return visual_features_cls, visual_features_mean


def extract_features_for_dataset(data_path, processor, model, device, max_frames=10, use_cache=True):
    """
    为数据集中的所有视频提取视觉特征（支持缓存）

    Args:
        data_path: 数据pkl文件路径
        processor: ViT图像处理器
        model: ViT模型
        device: 运行设备
        max_frames: 每个视频的最大帧数
        use_cache: 是否使用缓存
    """
    df = pd.read_pickle(data_path)
    cache_path = get_cache_path(data_path)

    # 检查缓存
    if use_cache:
        cached_cls, cached_mean = load_cached_visual_features(cache_path)
        if cached_cls is not None and len(cached_cls) == len(df):
            print("  [缓存] 使用缓存的视觉特征，跳过提取")
            df['visual_feature_embedding_cls'] = cached_cls
            df['visual_feature_embedding_mean'] = cached_mean

            # 验证缓存特征
            print("\n[自检] 验证缓存的视觉特征...")
            valid_cls, valid_mean, avg_cls_norm, avg_mean_norm = validate_visual_features(df, max_frames)
            print(f"  有效CLS特征 (10帧×768维): {valid_cls} 个")
            print(f"  有效Mean特征 (10帧×768维): {valid_mean} 个")
            print(f"  平均CLS范数: {avg_cls_norm:.4f}")
            print(f"  平均Mean范数: {avg_mean_norm:.4f}")
            show_sample_visual_features(cached_cls, cached_mean, n=2)

            # 保存到pkl
            df.to_pickle(data_path)
            print(f"\nUpdated {data_path} with cached visual features")
            return
        elif cached_cls is not None:
            print(f"  [缓存] 缓存样本数不匹配 ({len(cached_cls)} vs {len(df)})，重新提取")

    # === 自检：验证帧路径 ===
    print("\n[自检] 验证视频帧路径...")
    has_frames, no_frames, total_frames = validate_frame_paths(df)
    print(f"  有帧文件: {has_frames} 个视频")
    print(f"  无帧文件: {no_frames} 个视频")
    print(f"  总帧数: {total_frames} 帧")
    if no_frames > 0:
        print(f"  [警告] {no_frames} 个视频没有帧，将使用零向量!")
    if has_frames > 0:
        avg_frames = total_frames / has_frames
        print(f"  平均每视频: {avg_frames:.1f} 帧")

    print(f"\nProcessing {len(df)} videos...")

    visual_features_cls_list = []
    visual_features_mean_list = []

    # 检查是否有临时保存的进度
    cls_npy_path = get_feature_npy_path(data_path, 'cls')
    mean_npy_path = get_feature_npy_path(data_path, 'mean')
    temp_cls_path = cls_npy_path.replace('.npy', '_temp.npy')
    temp_mean_path = mean_npy_path.replace('.npy', '_temp.npy')
    start_idx = 0

    if os.path.exists(temp_cls_path) and os.path.exists(temp_mean_path):
        print(f"\n[恢复] 发现临时文件，从断点继续...")
        temp_cls = np.load(temp_cls_path, allow_pickle=True)
        temp_mean = np.load(temp_mean_path, allow_pickle=True)
        visual_features_cls_list = temp_cls.tolist()
        visual_features_mean_list = temp_mean.tolist()
        start_idx = len(visual_features_cls_list)
        print(f"  已处理: {start_idx} 个视频，剩余: {len(df) - start_idx} 个")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if idx < start_idx:
            continue

        frame_paths = row.get('frame_paths', [])

        if frame_paths:
            cls_features, mean_features = extract_video_features(
                processor, model, frame_paths, device, max_frames
            )
        else:
            # 没有帧，使用零向量
            cls_features = [[0.0] * 768 for _ in range(max_frames)]
            mean_features = [[0.0] * 768 for _ in range(max_frames)]

        visual_features_cls_list.append(cls_features)
        visual_features_mean_list.append(mean_features)

        # 每处理完100个视频就保存一次（支持断点续传）
        if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
            np.save(temp_cls_path, np.array(visual_features_cls_list, dtype=object))
            np.save(temp_mean_path, np.array(visual_features_mean_list, dtype=object))
            tqdm.write(f"  [保存] 已处理 {idx + 1}/{len(df)} 个视频")

    # 展示样本特征
    show_sample_visual_features(visual_features_cls_list, visual_features_mean_list, n=2)

    # 添加到DataFrame
    df['visual_feature_embedding_cls'] = visual_features_cls_list
    df['visual_feature_embedding_mean'] = visual_features_mean_list

    # === 自检：验证输出特征 ===
    print("\n[自检] 验证视觉特征...")
    valid_cls, valid_mean, avg_cls_norm, avg_mean_norm = validate_visual_features(df, max_frames)
    print(f"  有效CLS特征 (10帧×768维): {valid_cls} 个")
    print(f"  有效Mean特征 (10帧×768维): {valid_mean} 个")
    print(f"  平均CLS范数: {avg_cls_norm:.4f}")
    print(f"  平均Mean范数: {avg_mean_norm:.4f}")

    # 保存npy文件（从临时文件重命名）
    if os.path.exists(temp_cls_path) and os.path.exists(temp_mean_path):
        shutil.move(temp_cls_path, cls_npy_path)
        shutil.move(temp_mean_path, mean_npy_path)
        print(f"\n[保存] CLS特征已保存到: {cls_npy_path}")
        print(f"[保存] Mean特征已保存到: {mean_npy_path}")
    else:
        save_visual_features_to_npy(visual_features_cls_list, visual_features_mean_list, data_path)

    # 保存缓存
    if use_cache:
        save_cached_visual_features(cache_path, visual_features_cls_list, visual_features_mean_list)
        print(f"[保存] 缓存已保存到: {cache_path}")

    # 保存更新后的数据
    df.to_pickle(data_path)
    print(f"[保存] 数据已更新: {data_path}")


def show_sample_visual_features(cls_features_list, mean_features_list, n=2):
    """展示样本视觉特征"""
    print(f"\n[样本展示] 前{n}个视觉特征样本:")
    print("-" * 60)
    for i in range(min(n, len(cls_features_list))):
        cls_feat = cls_features_list[i]
        mean_feat = mean_features_list[i]
        print(f"\n样本 {i+1}:")
        print(f"  CLS特征: {len(cls_feat)} 帧 × {len(cls_feat[0]) if cls_feat else 0} 维")
        if cls_feat and cls_feat[0]:
            print(f"  第0帧CLS范数: {np.linalg.norm(cls_feat[0]):.4f}")
            print(f"  第0帧CLS前10维: {[f'{x:.4f}' for x in cls_feat[0][:10]]}")
        print(f"  Mean特征: {len(mean_feat)} 帧 × {len(mean_feat[0]) if mean_feat else 0} 维")
        if mean_feat and mean_feat[0]:
            print(f"  第0帧Mean范数: {np.linalg.norm(mean_feat[0]):.4f}")
            print(f"  第0帧Mean前10维: {[f'{x:.4f}' for x in mean_feat[0][:10]]}")
    print("-" * 60)


def compute_retrieval_feature(df):
    """
    计算用于检索的特征（视觉和文本特征的融合）

    Args:
        df: 数据DataFrame

    Returns:
        retrieval_features: 检索特征列表
    """
    retrieval_features = []

    for idx, row in df.iterrows():
        # 使用CLS token的平均作为视觉检索特征
        visual_cls = np.array(row['visual_feature_embedding_cls'])
        visual_retrieval = visual_cls.mean(axis=0).tolist()

        # 文本检索特征
        textual = np.array(row['textual_feature_embedding'])
        textual_retrieval = textual.flatten().tolist()

        # 融合 (简单平均)
        if len(visual_retrieval) == len(textual_retrieval):
            retrieval = [(v + t) / 2 for v, t in zip(visual_retrieval, textual_retrieval)]
        else:
            # 如果维度不匹配，使用文本特征
            retrieval = textual_retrieval

        retrieval_features.append(retrieval)

    return retrieval_features


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract visual features from SMP-Video frames')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test', 'all'],
                        help='Which dataset to process')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing data pkl files')
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224-in21k',
                        help='ViT model name')
    parser.add_argument('--num_frames', type=int, default=10,
                        help='Number of frames per video')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu', None],
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable caching')
    parser.add_argument('--local_only', action='store_true',
                        help='Use local cache only (offline mode)')

    args = parser.parse_args()

    # 加载模型
    processor, model, device = load_vit_model(args.model_name, args.device, local_only=args.local_only)

    use_cache = not args.no_cache

    if args.mode == 'all':
        for mode in ['train', 'valid', 'test']:
            data_path = os.path.join(args.data_dir, f'{mode}.pkl')
            if os.path.exists(data_path):
                print(f"\n{'='*50}")
                print(f"Processing {mode} set")
                print(f"{'='*50}")
                extract_features_for_dataset(
                    data_path,
                    processor,
                    model,
                    device,
                    args.num_frames,
                    use_cache=use_cache
                )
    else:
        data_path = os.path.join(args.data_dir, f'{args.mode}.pkl')
        if os.path.exists(data_path):
            extract_features_for_dataset(
                data_path,
                processor,
                model,
                device,
                args.num_frames,
                use_cache=use_cache
            )
        else:
            print(f"Data file not found: {data_path}")
