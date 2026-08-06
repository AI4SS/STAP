"""
新数据集 (V2) 文本特征提取
使用SentenceTransformer提取文本特征
与原有代码完全分离
"""
import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_v2")
CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)


def get_cache_path(data_path, suffix='_textual_features.npz'):
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    return os.path.join(CACHE_DIR, f'{base_name}{suffix}')


def get_feature_npy_path(data_path):
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    return os.path.join(FEATURES_DIR, f'{base_name}_textual_features.npy')


def load_cached_features(cache_path):
    if os.path.exists(cache_path):
        print(f"  [缓存] 发现缓存文件: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data['features'].tolist(), data['processed_texts'].tolist()
    return None, None


def save_cached_features(cache_path, features, processed_texts):
    np.savez(cache_path, features=np.array(features, dtype=object), processed_texts=np.array(processed_texts, dtype=object))
    print(f"  [缓存] 特征已缓存到: {cache_path}")


def load_text_model(model_name='WhereIsAI/UAE-Large-V1', device=None):
    print(f"Loading SentenceTransformer model: {model_name}")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SentenceTransformer(model_name, device=device, local_files_only=True)
    
    if device == 'cuda' and torch.cuda.is_available():
        print(f"Model loaded on CUDA: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Model loaded on CPU")

    return model, device


def preprocess_text(row):
    video_description = str(row.get('video_description', ''))
    
    if video_description:
        video_description = video_description.replace(',', ' ')
        video_description = ' '.join(video_description.split())
    
    hashtags = row.get('hashtags', [])
    mentions = row.get('mentions', [])
    
    if isinstance(hashtags, str):
        try:
            hashtags = eval(hashtags)
        except:
            hashtags = []
    if isinstance(mentions, str):
        try:
            mentions = eval(mentions)
        except:
            mentions = []
    
    tags_str = ' '.join([f'#{word}' for word in hashtags if word])
    mentions_str = ' '.join([f'@{word}' for word in mentions if word])
    
    parts = [video_description, tags_str, mentions_str]
    parts = [p for p in parts if p]
    processed_text = ' '.join(parts)
    
    if not processed_text or processed_text.strip() == '':
        processed_text = "video content"
    
    return processed_text


def extract_features_for_dataset(data_path, model, device, use_cache=True, batch_size=32):
    df = pd.read_pickle(data_path)
    cache_path = get_cache_path(data_path)
    npy_path = get_feature_npy_path(data_path)

    if use_cache:
        cached_features, cached_texts = load_cached_features(cache_path)
        if cached_features is not None and len(cached_features) == len(df):
            print("  [缓存] 使用缓存的文本特征，跳过提取")
            df['textual_feature_embedding'] = cached_features
            df['processed_text'] = cached_texts
            df.to_pickle(data_path)
            print(f"Updated {data_path} with cached textual features")
            return
        elif cached_features is not None:
            print(f"  [缓存] 缓存样本数不匹配 ({len(cached_features)} vs {len(df)})，重新提取")

    print(f"\nProcessing {len(df)} items...")

    processed_texts = []
    for idx, row in df.iterrows():
        processed_text = preprocess_text(row)
        processed_texts.append(processed_text)

    textual_features_list = []

    temp_npy_path = npy_path.replace('.npy', '_temp.npy')
    start_idx = 0

    if os.path.exists(temp_npy_path):
        print(f"\n[恢复] 发现临时文件，从断点继续...")
        temp_data = np.load(temp_npy_path, allow_pickle=True)
        textual_features_list = temp_data.tolist()
        start_idx = len(textual_features_list)
        print(f"  已处理: {start_idx} 条，剩余: {len(processed_texts) - start_idx} 条")

    for i in tqdm(range(start_idx, len(processed_texts), batch_size), desc="提取文本特征"):
        batch_texts = processed_texts[i:i+batch_size]
        try:
            embeddings = model.encode(batch_texts, convert_to_numpy=True)

            if i == start_idx:
                print(f"\n[调试] 批处理返回形状: {embeddings.shape}")

            for j in range(embeddings.shape[0]):
                textual_features_list.append(embeddings[j].tolist())
        except Exception as e:
            print(f"\n[警告] 批处理出错: {e}，逐个处理")
            for text in batch_texts:
                try:
                    emb = model.encode(text, convert_to_numpy=True)
                    textual_features_list.append(emb.tolist())
                except Exception as e2:
                    print(f"[错误] 单个处理失败: {e2}")
                    textual_features_list.append([0.0] * 384)

        if (i + batch_size) % (batch_size * 10) == 0 or (i + batch_size) >= len(processed_texts):
            np.save(temp_npy_path, np.array(textual_features_list, dtype=object))
            tqdm.write(f"  [保存] 已处理 {len(textual_features_list)}/{len(processed_texts)} 条")

    print(f"\n[样本展示] 前3个文本特征样本:")
    for i in range(min(3, len(textual_features_list))):
        feat = textual_features_list[i]
        text = processed_texts[i]
        print(f"\n样本 {i+1}:")
        print(f"  文本: {text[:100]}{'...' if len(text) > 100 else ''}")
        if feat:
            print(f"  特征维度: {len(feat)}")
            print(f"  特征范数: {np.linalg.norm(feat):.4f}")

    df['textual_feature_embedding'] = textual_features_list
    df['processed_text'] = processed_texts

    if os.path.exists(temp_npy_path):
        shutil.move(temp_npy_path, npy_path)
        print(f"\n[保存] 特征已保存到: {npy_path}")

    if use_cache:
        save_cached_features(cache_path, textual_features_list, processed_texts)

    df.to_pickle(data_path)
    print(f"[保存] 数据已更新: {data_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract textual features from V2 dataset')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test', 'all'],
                        help='Which dataset to process')
    parser.add_argument('--model_name', type=str, default='WhereIsAI/UAE-Large-V1',
                        help='SentenceTransformer model name')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu', None],
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable caching')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')

    args = parser.parse_args()

    model, device = load_text_model(args.model_name, args.device)

    use_cache = not args.no_cache

    if args.mode == 'all':
        for mode in ['train', 'valid', 'test']:
            data_path = os.path.join(OUTPUT_DIR, f'{mode}.pkl')
            if os.path.exists(data_path):
                print(f"\n{'='*50}")
                print(f"Processing {mode} set")
                print(f"{'='*50}")
                extract_features_for_dataset(data_path, model, device, use_cache=use_cache, batch_size=args.batch_size)
    else:
        data_path = os.path.join(OUTPUT_DIR, f'{args.mode}.pkl')
        if os.path.exists(data_path):
            extract_features_for_dataset(data_path, model, device, use_cache=use_cache, batch_size=args.batch_size)
        else:
            print(f"Data file not found: {data_path}")
