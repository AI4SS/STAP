"""
新数据集 (V2) 最终数据准备
将所有特征整合成训练所需的格式
支持多目标预测: video_comment_count, video_heart_count, video_play_count, video_share_count
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_v2")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data_v2", "processed_features")

os.makedirs(PROCESSED_DIR, exist_ok=True)

TARGET_COLUMNS = ['video_comment_count', 'video_heart_count', 'video_play_count', 'video_share_count']


def prepare_final_data():
    """准备最终训练数据"""
    print("=" * 60)
    print("新数据集 (V2) 最终数据准备")
    print("=" * 60)

    for mode in ['train', 'valid', 'test']:
        data_path = os.path.join(OUTPUT_DIR, f'{mode}.pkl')
        if not os.path.exists(data_path):
            print(f"数据文件不存在: {data_path}")
            continue

        print(f"\n处理 {mode} 集...")
        df = pd.read_pickle(data_path)
        print(f"  样本数: {len(df)}")

        visual_features = []
        text_features = []
        user_features = []
        labels = {col: [] for col in TARGET_COLUMNS}

        missing_visual = 0
        missing_text = 0
        missing_user = 0

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            vis_feat = row.get('visual_feature_embedding_cls', None)
            if vis_feat is None or len(vis_feat) == 0:
                vis_feat = [[0.0] * 768 for _ in range(100)]
                missing_visual += 1
            visual_features.append(vis_feat)

            txt_feat = row.get('textual_feature_embedding', None)
            if txt_feat is None or len(txt_feat) == 0:
                txt_feat = [0.0] * 1024
                missing_text += 1
            text_features.append(txt_feat)

            usr_feat = row.get('user_feature_vector', None)
            if usr_feat is None or len(usr_feat) == 0:
                usr_feat = [0.0] * 6
                missing_user += 1
            user_features.append(usr_feat)

            for col in TARGET_COLUMNS:
                label = row.get(col, 0)
                labels[col].append(label)

        print(f"  缺失视觉特征: {missing_visual}")
        print(f"  缺失文本特征: {missing_text}")
        print(f"  缺失用户特征: {missing_user}")

        data = {
            'visual_features': visual_features,
            'text_features': text_features,
            'user_features': user_features,
            'labels': labels,
        }

        output_path = os.path.join(PROCESSED_DIR, f'{mode}.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  已保存: {output_path}")

        print(f"\n  统计信息:")
        print(f"    视觉特征: {len(visual_features)} 个, 形状: {np.array(visual_features[0]).shape}")
        print(f"    文本特征: {len(text_features)} 个, 维度: {len(text_features[0])}")
        print(f"    用户特征: {len(user_features)} 个, 维度: {len(user_features[0])}")
        for col in TARGET_COLUMNS:
            print(f"    {col}: min={min(labels[col])}, max={max(labels[col])}, mean={np.mean(labels[col]):.2f}")

    print("\n" + "=" * 60)
    print("数据准备完成!")
    print("=" * 60)


if __name__ == '__main__':
    prepare_final_data()
