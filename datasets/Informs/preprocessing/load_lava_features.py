"""
加载Lava-BERT文本特征并替换原有文本特征
"""
import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_v2")
LAVA_TRAIN_DIR = os.path.join(OUTPUT_DIR, "features", "lava_bert_training")
LAVA_TEST_DIR = os.path.join(OUTPUT_DIR, "features", "lava_bert_testing")


def load_lava_features():
    """加载所有lava特征"""
    features = {}
    
    print("加载训练集lava特征...")
    for f in tqdm(os.listdir(LAVA_TRAIN_DIR)):
        if f.endswith('.pt'):
            video_id = f.replace('_repaired.pt', '')
            path = os.path.join(LAVA_TRAIN_DIR, f)
            data = torch.load(path, map_location='cpu')
            if isinstance(data, torch.Tensor):
                features[video_id] = data.squeeze().numpy().tolist()
    
    print("加载测试集lava特征...")
    for f in tqdm(os.listdir(LAVA_TEST_DIR)):
        if f.endswith('.pt'):
            video_id = f.replace('_repaired.pt', '')
            path = os.path.join(LAVA_TEST_DIR, f)
            data = torch.load(path, map_location='cpu')
            if isinstance(data, torch.Tensor):
                features[video_id] = data.squeeze().numpy().tolist()
    
    return features


def update_data_with_lava():
    """更新数据文件，替换文本特征为lava_bert特征"""
    print("=" * 60)
    print("加载Lava-BERT特征并更新数据")
    print("=" * 60)
    
    lava_features = load_lava_features()
    print(f"\n共加载 {len(lava_features)} 个lava特征")
    
    for mode in ['train', 'valid', 'test']:
        data_path = os.path.join(OUTPUT_DIR, f'{mode}.pkl')
        if not os.path.exists(data_path):
            print(f"数据文件不存在: {data_path}")
            continue
        
        print(f"\n处理 {mode} 集...")
        df = pd.read_pickle(data_path)
        print(f"  样本数: {len(df)}")
        
        textual_features = []
        missing_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"更新{mode}"):
            video_id = str(row['video_id'])
            if video_id in lava_features:
                textual_features.append(lava_features[video_id])
            else:
                textual_features.append([0.0] * 768)
                missing_count += 1
        
        df['textual_feature_embedding'] = textual_features
        df.to_pickle(data_path)
        
        print(f"  缺失特征: {missing_count}")
        print(f"  已更新: {data_path}")
        
        if len(textual_features) > 0:
            sample_feat = textual_features[0]
            print(f"  特征维度: {len(sample_feat)}")
            print(f"  特征范数: {np.linalg.norm(sample_feat):.4f}")
    
    print("\n" + "=" * 60)
    print("更新完成!")
    print("=" * 60)


if __name__ == '__main__':
    update_data_with_lava()
