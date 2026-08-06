"""
SMP-Video文本特征提取
使用Angle-BERT提取文本特征
遵循MMRA-main的文本处理思想
"""
import os
import shutil
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from angle_emb import AnglE


def get_cache_path(data_path, suffix='_textual_features.npz'):
    """获取缓存文件路径"""
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    cache_dir = os.path.join(os.path.dirname(data_path), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'{base_name}{suffix}')


def get_feature_npy_path(data_path):
    """获取特征npy文件路径"""
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    feature_dir = os.path.join(os.path.dirname(data_path), 'features')
    os.makedirs(feature_dir, exist_ok=True)
    return os.path.join(feature_dir, f'{base_name}_textual_features.npy')


def save_features_to_npy(features, texts, npy_path):
    """保存特征到npy文件"""
    np.save(npy_path, np.array(features, dtype=object))
    # 同时保存文本到txt文件用于检查
    txt_path = npy_path.replace('.npy', '_texts.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        for i, text in enumerate(texts[:10]):  # 只保存前10个样本
            f.write(f"[{i}] {text[:200]}...\n")
    print(f"  [保存] 特征已保存到: {npy_path}")
    print(f"  [保存] 样本文本已保存到: {txt_path}")


def load_cached_features(cache_path):
    """从缓存加载特征"""
    if os.path.exists(cache_path):
        print(f"  [缓存] 发现缓存文件: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data['features'].tolist(), data['processed_texts'].tolist()
    return None, None


def save_cached_features(cache_path, features, processed_texts):
    """保存特征到缓存"""
    np.savez(cache_path, features=np.array(features, dtype=object), processed_texts=np.array(processed_texts, dtype=object))
    print(f"  [缓存] 特征已缓存到: {cache_path}")


def validate_text_data(df):
    """
    验证文本数据完整性

    Args:
        df: 数据DataFrame

    Returns:
        has_content: 有内容的数量
        empty_content: 空内容数量
    """
    has_content = 0
    empty_content = 0
    for _, row in df.iterrows():
        content = row.get('post_content', '')
        suggested = row.get('post_suggested_words', [])
        if content or (suggested and len(suggested) > 0):
            has_content += 1
        else:
            empty_content += 1
    return has_content, empty_content


def validate_textual_features(df):
    """
    验证文本特征提取结果

    Args:
        df: 数据DataFrame

    Returns:
        valid_features: 有效特征数量
        invalid_features: 无效特征数量
        avg_norm: 平均特征范数
    """
    valid_count = 0
    invalid_count = 0
    norms = []

    for feat in df['textual_feature_embedding']:
        if feat and len(feat) == 768:
            valid_count += 1
            norms.append(np.linalg.norm(feat))
        else:
            invalid_count += 1

    avg_norm = np.mean(norms) if norms else 0
    return valid_count, invalid_count, avg_norm


def load_angle_model(model_name='WhereIsAI/UAE-Large-V1', device=None):
    """
    加载Angle-BERT模型

    Args:
        model_name: 预训练模型名称
        device: 指定设备 ('cuda', 'cpu', 或 None 自动检测)

    Returns:
        angle: AnglE模型实例
    """
    print(f"Loading Angle model: {model_name}")

    # 自动检测设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    angle = AnglE.from_pretrained(
        model_name,
        pooling_strategy='cls'
    )

    # 将模型移到指定设备
    if device == 'cuda' and torch.cuda.is_available():
        angle = angle.cuda()
        print(f"Model loaded on CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print(f"Model loaded on CPU")

    return angle, device


def preprocess_text(post_data):
    """
    预处理帖子文本内容

    Args:
        post_data: 帖子数据字典，包含:
            - post_content: 帖子内容
            - post_suggested_words: 建议的标签/关键词
            - post_text_language: 语言

    Returns:
        processed_text: 处理后的文本
    """
    post_content = post_data.get('post_content', '')
    suggested_words = post_data.get('post_suggested_words', [])

    # 【重要】将逗号替换为空格，形成完整句子
    # 原始数据可能是 "word1,word2,word3" 格式
    if post_content:
        post_content = post_content.replace(',', ' ')
        # 去除多余空格
        post_content = ' '.join(post_content.split())

    # 组合内容和建议词
    if suggested_words:
        # 将建议词作为标签添加到内容中
        tags_str = ' '.join([f'#{word}' for word in suggested_words if word])
        if post_content:
            processed_text = f"{post_content} {tags_str}"
        else:
            processed_text = tags_str
    else:
        processed_text = post_content

    # 如果仍然为空，使用默认文本
    if not processed_text or processed_text.strip() == '':
        processed_text = "video content"

    return processed_text


def extract_textual_feature(angle, text):
    """
    提取文本特征

    Args:
        angle: AnglE模型
        text: 输入文本

    Returns:
        embedding: 文本嵌入向量 (768维)
    """
    try:
        embedding = angle.encode(text, to_numpy=True)
        # 确保返回一维列表
        if len(embedding.shape) > 1:
            embedding = embedding[0]
        return embedding.tolist()
    except Exception as e:
        print(f"Error processing text: {e}")
        return [0.0] * 768


def show_sample_features(textual_features_list, processed_texts, n=3):
    """展示样本特征"""
    print(f"\n[样本展示] 前{n}个文本特征样本:")
    print("-" * 60)
    for i in range(min(n, len(textual_features_list))):
        feat = textual_features_list[i]
        text = processed_texts[i]
        print(f"\n样本 {i+1}:")
        print(f"  文本: {text[:100]}{'...' if len(text) > 100 else ''}")
        if feat and len(feat) == 768:
            print(f"  特征维度: {len(feat)}")
            print(f"  特征范数: {np.linalg.norm(feat):.4f}")
            print(f"  特征范围: [{min(feat):.4f}, {max(feat):.4f}]")
            print(f"  前10维: {[f'{x:.4f}' for x in feat[:10]]}")
        else:
            print(f"  特征: 无效或零向量")
    print("-" * 60)


def extract_features_for_dataset(data_path, angle, use_cache=True, batch_size=32):
    """
    为数据集中的所有项目提取文本特征（支持缓存和批处理）

    Args:
        data_path: 数据pkl文件路径
        angle: AnglE模型
        use_cache: 是否使用缓存
        batch_size: 批处理大小
    """
    df = pd.read_pickle(data_path)
    cache_path = get_cache_path(data_path)
    npy_path = get_feature_npy_path(data_path)

    # 检查缓存
    if use_cache:
        cached_features, cached_texts = load_cached_features(cache_path)
        if cached_features is not None and len(cached_features) == len(df):
            print("  [缓存] 使用缓存的文本特征，跳过提取")
            df['textual_feature_embedding'] = cached_features
            df['processed_text'] = cached_texts

            # 验证缓存特征
            print("\n[自检] 验证缓存的文本特征...")
            valid_feat, invalid_feat, avg_norm = validate_textual_features(df)
            print(f"  有效特征 (768维): {valid_feat} 个")
            print(f"  无效特征: {invalid_feat} 个")
            print(f"  平均特征范数: {avg_norm:.4f}")
            show_sample_features(cached_features, cached_texts, n=2)

            # 保存到pkl
            df.to_pickle(data_path)
            print(f"\nUpdated {data_path} with cached textual features")
            return
        elif cached_features is not None:
            print(f"  [缓存] 缓存样本数不匹配 ({len(cached_features)} vs {len(df)})，重新提取")

    # === 自检：验证输入文本数据 ===
    print("\n[自检] 验证输入文本数据...")
    has_content, empty_content = validate_text_data(df)
    print(f"  有文本内容: {has_content} 个")
    print(f"  空文本内容: {empty_content} 个")

    print(f"\nProcessing {len(df)} items...")

    # 预处理所有文本
    processed_texts = []
    for _, row in df.iterrows():
        post_data = {
            'post_content': row.get('post_content', ''),
            'post_suggested_words': row.get('post_suggested_words', []),
            'post_text_language': row.get('post_text_language', 'en')
        }
        processed_text = preprocess_text(post_data)
        processed_texts.append(processed_text)

    # 批处理提取特征 (GPU加速)
    textual_features_list = []

    # 检查是否有临时保存的进度
    temp_npy_path = npy_path.replace('.npy', '_temp.npy')
    start_idx = 0

    if os.path.exists(temp_npy_path):
        print(f"\n[恢复] 发现临时文件，从断点继续...")
        temp_data = np.load(temp_npy_path, allow_pickle=True)
        textual_features_list = temp_data.tolist()
        start_idx = len(textual_features_list)
        print(f"  已处理: {start_idx} 条，剩余: {len(processed_texts) - start_idx} 条")

    total_batches = (len(processed_texts) + batch_size - 1) // batch_size
    current_batch = start_idx // batch_size

    for i in tqdm(range(start_idx, len(processed_texts), batch_size), desc="提取文本特征", initial=current_batch, total=total_batches):
        batch_texts = processed_texts[i:i+batch_size]
        batch_start = i
        try:
            # 批编码
            embeddings = angle.encode(batch_texts, to_numpy=True)

            # 调试：打印第一次批处理的形状
            if i == start_idx:
                print(f"\n[调试] 批处理返回形状: {embeddings.shape}")
                print(f"[调试] 批处理内容前3个: {batch_texts[:3]}")

            # 处理不同的返回格式
            if embeddings.ndim == 2:
                # shape: (batch_size, 768)
                for j in range(embeddings.shape[0]):
                    textual_features_list.append(embeddings[j].tolist())
            elif embeddings.ndim == 1:
                # shape: (768,) - 单个向量
                textual_features_list.append(embeddings.tolist())
            else:
                # 其他格式，逐个处理
                print(f"\n[警告] 意外的embedding形状: {embeddings.shape}")
                for text in batch_texts:
                    try:
                        emb = angle.encode(text, to_numpy=True)
                        if emb.ndim == 1:
                            textual_features_list.append(emb.tolist())
                        elif emb.ndim == 2 and emb.shape[0] == 1:
                            textual_features_list.append(emb[0].tolist())
                        else:
                            textual_features_list.append([0.0] * 768)
                    except:
                        textual_features_list.append([0.0] * 768)
        except Exception as e:
            print(f"\n[警告] 批处理出错: {e}，逐个处理")
            for text in batch_texts:
                try:
                    emb = angle.encode(text, to_numpy=True)
                    if emb.ndim == 1:
                        textual_features_list.append(emb.tolist())
                    elif emb.ndim == 2:
                        if emb.shape[0] == 1:
                            textual_features_list.append(emb[0].tolist())
                        else:
                            # 多个结果，取第一个
                            textual_features_list.append(emb[0].tolist())
                    else:
                        textual_features_list.append([0.0] * 768)
                except Exception as e2:
                    print(f"[错误] 单个处理失败: {e2}")
                    textual_features_list.append([0.0] * 768)

        # 每处理完一个批次就保存一次（支持断点续传）
        if (i + batch_size) % (batch_size * 10) == 0 or (i + batch_size) >= len(processed_texts):
            np.save(temp_npy_path, np.array(textual_features_list, dtype=object))
            tqdm.write(f"  [保存] 已处理 {len(textual_features_list)}/{len(processed_texts)} 条")

    # 展示样本特征
    show_sample_features(textual_features_list, processed_texts, n=3)

    # 添加到DataFrame
    df['textual_feature_embedding'] = textual_features_list
    df['processed_text'] = processed_texts

    # === 自检：验证输出特征 ===
    print("\n[自检] 验证文本特征...")
    valid_feat, invalid_feat, avg_norm = validate_textual_features(df)
    print(f"  有效特征 (768维): {valid_feat} 个")
    print(f"  无效特征: {invalid_feat} 个")
    print(f"  平均特征范数: {avg_norm:.4f}")
    if invalid_feat > 0:
        print(f"  [警告] 发现 {invalid_feat} 个无效特征!")

    # 保存npy文件（从临时文件重命名）
    if os.path.exists(temp_npy_path):
        shutil.move(temp_npy_path, npy_path)
        print(f"\n[保存] 特征已保存到: {npy_path}")
    else:
        save_features_to_npy(textual_features_list, processed_texts, npy_path)

    # 保存缓存
    if use_cache:
        save_cached_features(cache_path, textual_features_list, processed_texts)
        print(f"[保存] 缓存已保存到: {cache_path}")

    # 保存更新后的数据
    df.to_pickle(data_path)
    print(f"[保存] 数据已更新: {data_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract textual features from SMP-Video')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'valid', 'test', 'all'],
                        help='Which dataset to process')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Directory containing data pkl files')
    parser.add_argument('--model_name', type=str, default='WhereIsAI/UAE-Large-V1',
                        help='Angle model name')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu', None],
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable caching')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for feature extraction')

    args = parser.parse_args()

    # 加载模型
    angle, device = load_angle_model(args.model_name, args.device)

    use_cache = not args.no_cache

    if args.mode == 'all':
        for mode in ['train', 'valid', 'test']:
            data_path = os.path.join(args.data_dir, f'{mode}.pkl')
            if os.path.exists(data_path):
                print(f"\n{'='*50}")
                print(f"Processing {mode} set")
                print(f"{'='*50}")
                extract_features_for_dataset(data_path, angle, use_cache=use_cache, batch_size=args.batch_size)
    else:
        data_path = os.path.join(args.data_dir, f'{args.mode}.pkl')
        if os.path.exists(data_path):
            extract_features_for_dataset(data_path, angle, use_cache=use_cache, batch_size=args.batch_size)
        else:
            print(f"Data file not found: {data_path}")
