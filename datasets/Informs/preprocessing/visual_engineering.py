"""
新数据集 (V2) 视觉特征提取
使用ViT提取视频帧的视觉特征
与原有代码完全分离
"""
import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_v2")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)


def get_cache_path(data_path, suffix='_visual_features.npz'):
    """获取缓存文件路径"""
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    return os.path.join(CACHE_DIR, f'{base_name}{suffix}')


def get_feature_npy_path(data_path, feature_type='cls'):
    """获取特征npy文件路径"""
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    return os.path.join(FEATURES_DIR, f'{base_name}_visual_{feature_type}_features.npy')


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


def load_vit_model(model_name='google/vit-base-patch16-224-in21k', device=None, local_only=False):
    """加载ViT模型和处理器"""
    print(f"Loading ViT model: {model_name}")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, using CPU")
        device = 'cpu'

    load_kwargs = {}
    if local_only:
        load_kwargs['local_files_only'] = True
        print("[离线模式] 仅使用本地缓存")

    try:
        processor = ViTImageProcessor.from_pretrained(model_name, **load_kwargs)
        model = ViTModel.from_pretrained(model_name, **load_kwargs)
    except OSError as e:
        if local_only:
            print(f"\n[错误] 离线模式下找不到本地模型: {model_name}")
            raise
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
    """从单帧图像提取视觉特征"""
    try:
        image = Image.open(frame_path).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        cls_token = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        mean_token = outputs.last_hidden_state[0, :, :].mean(dim=0).cpu().numpy()

        return cls_token.tolist(), mean_token.tolist()

    except Exception as e:
        print(f"Error processing {frame_path}: {e}")
        return [0.0] * 768, [0.0] * 768


def extract_video_features(processor, model, frame_paths, device, max_frames=100):
    """从视频的所有帧提取特征"""
    visual_features_cls = []
    visual_features_mean = []

    for frame_path in frame_paths[:max_frames]:
        cls_token, mean_token = extract_frame_features(processor, model, frame_path, device)
        visual_features_cls.append(cls_token)
        visual_features_mean.append(mean_token)

    while len(visual_features_cls) < max_frames:
        visual_features_cls.append([0.0] * 768)
        visual_features_mean.append([0.0] * 768)

    return visual_features_cls, visual_features_mean


def extract_features_for_dataset(data_path, processor, model, device, max_frames=100, use_cache=True):
    """为数据集中的所有视频提取视觉特征"""
    df = pd.read_pickle(data_path)
    cache_path = get_cache_path(data_path)

    if use_cache:
        cached_cls, cached_mean = load_cached_visual_features(cache_path)
        if cached_cls is not None and len(cached_cls) == len(df):
            print("  [缓存] 使用缓存的视觉特征，跳过提取")
            df['visual_feature_embedding_cls'] = cached_cls
            df['visual_feature_embedding_mean'] = cached_mean
            df.to_pickle(data_path)
            print(f"Updated {data_path} with cached visual features")
            return
        elif cached_cls is not None:
            print(f"  [缓存] 缓存样本数不匹配 ({len(cached_cls)} vs {len(df)})，重新提取")

    print(f"\nProcessing {len(df)} videos...")

    visual_features_cls_list = []
    visual_features_mean_list = []

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
            cls_features = [[0.0] * 768 for _ in range(max_frames)]
            mean_features = [[0.0] * 768 for _ in range(max_frames)]

        visual_features_cls_list.append(cls_features)
        visual_features_mean_list.append(mean_features)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
            np.save(temp_cls_path, np.array(visual_features_cls_list, dtype=object))
            np.save(temp_mean_path, np.array(visual_features_mean_list, dtype=object))
            tqdm.write(f"  [保存] 已处理 {idx + 1}/{len(df)} 个视频")

    print(f"\n[样本展示] 前2个视觉特征样本:")
    for i in range(min(2, len(visual_features_cls_list))):
        cls_feat = visual_features_cls_list[i]
        mean_feat = visual_features_mean_list[i]
        print(f"\n样本 {i+1}:")
        print(f"  CLS特征: {len(cls_feat)} 帧 × {len(cls_feat[0]) if cls_feat else 0} 维")
        if cls_feat and cls_feat[0]:
            print(f"  第0帧CLS范数: {np.linalg.norm(cls_feat[0]):.4f}")

    df['visual_feature_embedding_cls'] = visual_features_cls_list
    df['visual_feature_embedding_mean'] = visual_features_mean_list

    if os.path.exists(temp_cls_path) and os.path.exists(temp_mean_path):
        shutil.move(temp_cls_path, cls_npy_path)
        shutil.move(temp_mean_path, mean_npy_path)
        print(f"\n[保存] CLS特征已保存到: {cls_npy_path}")
        print(f"[保存] Mean特征已保存到: {mean_npy_path}")

    if use_cache:
        save_cached_visual_features(cache_path, visual_features_cls_list, visual_features_mean_list)

    df.to_pickle(data_path)
    print(f"[保存] 数据已更新: {data_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract visual features from V2 dataset')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test', 'all'],
                        help='Which dataset to process')
    parser.add_argument('--model_name', type=str, default='google/vit-base-patch16-224-in21k',
                        help='ViT model name')
    parser.add_argument('--num_frames', type=int, default=100,
                        help='Number of frames per video')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu', None],
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable caching')
    parser.add_argument('--local_only', action='store_true',
                        help='Use local cache only (offline mode)')

    args = parser.parse_args()

    processor, model, device = load_vit_model(args.model_name, args.device, local_only=args.local_only)

    use_cache = not args.no_cache

    if args.mode == 'all':
        for mode in ['train', 'valid', 'test']:
            data_path = os.path.join(OUTPUT_DIR, f'{mode}.pkl')
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
        data_path = os.path.join(OUTPUT_DIR, f'{args.mode}.pkl')
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
