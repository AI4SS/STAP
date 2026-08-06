"""
MicroLens特征提取脚本
使用AnglE提取文本特征
参考ICPF-main的特征提取方法
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def extract_text_features_angle(
    titles_file: str,
    output_path: str,
    device: str = 'cuda:0',
    batch_size: int = 32,
):
    """
    使用AnglE提取文本特征
    """
    print("\n" + "=" * 60)
    print("使用AnglE提取文本特征")
    print("=" * 60)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    print("加载AnglE模型...")
    angle_model = None
    use_angle = False
    
    try:
        from angle_emb import AnglE
        local_model_path = '/root/.cache/huggingface/hub/models--WhereIsAI--UAE-Large-V1'
        if os.path.exists(local_model_path):
            snapshots_dir = os.path.join(local_model_path, 'snapshots')
            if os.path.exists(snapshots_dir):
                snapshot_id = os.listdir(snapshots_dir)[0]
                local_model_path = os.path.join(snapshots_dir, snapshot_id)
            angle_model = AnglE.from_pretrained(local_model_path, local_files_only=True)
            print(f"使用本地缓存的UAE-Large-V1模型: {local_model_path}")
        else:
            angle_model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', local_files_only=True)
            print("使用本地缓存的UAE-Large-V1模型")
        angle_model = angle_model.to(device)
        angle_model.eval()
        use_angle = True
    except Exception as e:
        print(f"AnglE加载失败: {e}, 使用BERT替代...")
        return extract_text_features_bert(titles_file, output_path, device, batch_size)
    
    titles_df = pd.read_csv(titles_file, header=None, names=['video_id', 'title'])
    print(f"标题数量: {len(titles_df)}")
    
    video_ids = titles_df['video_id'].tolist()
    titles = titles_df['title'].fillna('').tolist()
    
    all_features = []
    
    for i in tqdm(range(0, len(titles), batch_size), desc="提取文本特征"):
        batch_titles = titles[i:i+batch_size]
        with torch.no_grad():
            embeddings = angle_model.encode(batch_titles)
        all_features.extend(embeddings)
    
    all_features = np.array(all_features)
    
    print(f"文本特征形状: {all_features.shape}")
    
    np.save(output_path, all_features)
    print(f"已保存到: {output_path}")
    
    return all_features, video_ids


def extract_text_features_bert(
    titles_file: str,
    output_path: str,
    device: str = 'cuda:0',
    batch_size: int = 32,
):
    """使用BERT提取文本特征作为备选"""
    from transformers import AutoModel, AutoTokenizer
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    print("加载BERT模型...")
    local_bert_path = '/root/.cache/huggingface/hub/models--bert-base-uncased'
    if os.path.exists(local_bert_path):
        snapshots_dir = os.path.join(local_bert_path, 'snapshots')
        if os.path.exists(snapshots_dir):
            snapshot_id = os.listdir(snapshots_dir)[0]
            local_bert_path = os.path.join(snapshots_dir, snapshot_id)
        tokenizer = AutoTokenizer.from_pretrained(local_bert_path, local_files_only=True)
        model = AutoModel.from_pretrained(local_bert_path, local_files_only=True)
        print(f"使用本地缓存的BERT模型: {local_bert_path}")
    else:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
        model = AutoModel.from_pretrained('bert-base-uncased', local_files_only=True)
        print("使用本地缓存的BERT模型")
    model = model.to(device)
    model.eval()
    
    titles_df = pd.read_csv(titles_file, header=None, names=['video_id', 'title'])
    print(f"标题数量: {len(titles_df)}")
    
    video_ids = titles_df['video_id'].tolist()
    titles = titles_df['title'].fillna('').tolist()
    
    all_features = []
    
    for i in tqdm(range(0, len(titles), batch_size), desc="提取文本特征"):
        batch_titles = titles[i:i+batch_size]
        inputs = tokenizer(batch_titles, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_features.extend(embeddings)
    
    all_features = np.array(all_features)
    
    print(f"文本特征形状: {all_features.shape}")
    
    np.save(output_path, all_features)
    print(f"已保存到: {output_path}")
    
    return all_features, video_ids


def main():
    parser = argparse.ArgumentParser(description='MicroLens文本特征提取')
    parser.add_argument('--titles_file', type=str,
                        default='/root/gsar_jamba_topo/data_MicroLens/MicroLens-100k_title_en.csv')
    parser.add_argument('--output_dir', type=str,
                        default='/root/gsar_jamba_topo/data_MicroLens/extracted_features_6frames')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    text_features, video_ids = extract_text_features_angle(
        args.titles_file,
        os.path.join(args.output_dir, 'text_features_angle.npy'),
        args.device,
        args.batch_size,
    )
    
    import json
    meta = {
        'video_ids': video_ids,
        'text_features_shape': list(text_features.shape),
    }
    with open(os.path.join(args.output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    print("\n" + "=" * 60)
    print("特征提取完成!")
    print("=" * 60)
    print(f"输出目录: {args.output_dir}")
    print(f"文本特征: {text_features.shape}")


if __name__ == '__main__':
    main()
