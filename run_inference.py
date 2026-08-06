"""
推理脚本 - 对 saved_models 中的模型进行推理评估
"""
import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.gsar_jamba_v2 import create_gsar_jamba_encoder_v2
from models.topo_pmb_multimodal import TopologyAwarePMBMultimodal, create_topo_pmb_multimodal
from models.topo_pmb import TopologyAwarePMB, create_topo_pmb
from core.load_balance import create_load_balancing
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List


class DPPO:
    def __init__(self, pool_size=1000, margin=0.5, beta=0.5, gamma=1.0,
                 middle_sample_ratio=0.2, eps=1e-8):
        self.pool_size = pool_size
        self.margin = margin
        self.beta = beta
        self.gamma = gamma
        self.middle_sample_ratio = middle_sample_ratio
        self.eps = eps
        self.sample_pool = []


class GSARJambaTOPOV2Model(nn.Module):
    def __init__(self, d_input=768, d_model=768, d_text=1024, d_user=6,
                 max_frames=1, m_heat=6, n_topic=32, top_k=5, n_blocks=2,
                 d_state=16, d_conv=4, expand=2, n_heads=12,
                 base_window_size=10, window_alpha=0.5, dt_rank=48,
                 delta_min=0.01, delta_max=1.0, temperature=1.0,
                 learnable_temperature=True, fixed_pmb_path=None,
                 use_load_balance=True, load_balance_alpha=0.01,
                 load_balance_beta=0.01, long_tail_exponent=1.5,
                 use_dppo=True, dppo_lambda=0.1, dppo_beta=0.5,
                 dppo_pool_size=1000, dppo_margin=0.5, dppo_gamma=1.0,
                 dropout=0.1):
        super().__init__()
        self.use_dppo = use_dppo
        self.dppo_lambda = dppo_lambda
        self.d_model = d_model

        self.encoder = create_gsar_jamba_encoder_v2(
            num_frames=max_frames, d_input=d_input, d_model=d_model,
            n_blocks=n_blocks, d_state=d_state, d_conv=d_conv, expand=expand,
            n_heads=n_heads, base_window_size=base_window_size,
            window_alpha=window_alpha, dt_rank=dt_rank,
            delta_min=delta_min, delta_max=delta_max, dropout=dropout)

        self.topo_pmb = create_topo_pmb_multimodal(
            d_model=d_model, m_heat=m_heat, n_topic=n_topic, top_k=top_k,
            temperature=temperature, learnable_temperature=learnable_temperature,
            fixed_pmb_path=fixed_pmb_path, dropout=dropout)

        if use_load_balance:
            self.load_balance = create_load_balancing(
                m_heat=m_heat, n_topic=n_topic, gamma=load_balance_alpha,
                alpha=load_balance_alpha, beta=load_balance_beta,
                long_tail_exponent=long_tail_exponent)
        else:
            self.load_balance = None

        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.user_proj = nn.Sequential(
            nn.Linear(d_user, 64), nn.ReLU(), nn.Linear(64, d_model))

        self.visual_embedding = nn.Linear(d_model, d_model)
        self.textual_embedding = nn.Linear(d_model, d_model)
        self.retrieval_embedding = nn.Linear(d_model, d_model)

        self.dual_attention_linear_1 = nn.Linear(d_model * 2, d_model)
        self.dual_attention_linear_2 = nn.Linear(d_model * 2, d_model)
        self.cross_modal_linear_1 = nn.Linear(d_model * 2, d_model)
        self.cross_modal_linear_2 = nn.Linear(d_model * 2, d_model)
        self.uni_modal_linear_1 = nn.Linear(d_model, 1)
        self.uni_modal_linear_2 = nn.Linear(d_model, 1)

        self.retrieval_dual_1 = nn.Linear(d_model * 2, d_model)
        self.retrieval_dual_2 = nn.Linear(d_model * 2, d_model)
        self.retrieval_cross_1 = nn.Linear(d_model * 2, d_model)
        self.retrieval_cross_2 = nn.Linear(d_model * 2, d_model)
        self.retrieval_uni_1 = nn.Linear(d_model, 1)
        self.retrieval_uni_2 = nn.Linear(d_model, 1)

        self.tanh = nn.Tanh()
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 6, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))

        if use_dppo:
            self.dppo = DPPO(pool_size=dppo_pool_size, margin=dppo_margin,
                             beta=dppo_beta, gamma=dppo_gamma,
                             middle_sample_ratio=0.2, eps=1e-8)

    def cross_modal_attention(self, visual_feature, textual_feature, visual_emb, textual_emb):
        d_model = visual_feature.shape[-1]
        S = torch.matmul(visual_emb, textual_emb.transpose(1, 2)) / d_model
        T_p = torch.matmul(F.softmax(S, dim=1), textual_feature)
        V_p = torch.matmul(F.softmax(S.transpose(1, 2), dim=1), visual_feature)
        T_n = torch.matmul((-0.5) * F.softmax(S, dim=1), textual_feature)
        V_n = torch.matmul((-0.5) * F.softmax(S.transpose(1, 2), dim=1), visual_feature)
        T_star = self.tanh(self.dual_attention_linear_1(torch.cat([T_p, T_n], dim=2)))
        V_star = self.tanh(self.dual_attention_linear_2(torch.cat([V_p, V_n], dim=2)))
        V_f = self.tanh(self.cross_modal_linear_1(torch.cat([visual_feature, T_star], dim=2)))
        T_f = self.tanh(self.cross_modal_linear_2(torch.cat([textual_feature, V_star], dim=2)))
        return V_f, T_f

    def retrieval_attention(self, retrieved_feature, retrieved_emb):
        d_model = retrieved_feature.shape[-1]
        S = torch.matmul(retrieved_emb, retrieved_emb.transpose(1, 2)) / d_model
        T_p = torch.matmul(F.softmax(S, dim=1), retrieved_feature)
        V_p = torch.matmul(F.softmax(S.transpose(1, 2), dim=1), retrieved_feature)
        T_n = torch.matmul((-0.5) * F.softmax(S, dim=1), retrieved_feature)
        V_n = torch.matmul((-0.5) * F.softmax(S.transpose(1, 2), dim=1), retrieved_feature)
        T_star = self.tanh(self.retrieval_dual_1(torch.cat([T_p, T_n], dim=2)))
        V_star = self.tanh(self.retrieval_dual_2(torch.cat([V_p, V_n], dim=2)))
        V_f = self.tanh(self.retrieval_cross_1(torch.cat([retrieved_feature, T_star], dim=2)))
        T_f = self.tanh(self.retrieval_cross_2(torch.cat([retrieved_feature, V_star], dim=2)))
        return V_f, T_f

    def uni_modal_attention(self, V_f, T_f):
        d_model = V_f.shape[-1]
        alpha_v = F.softmax(self.uni_modal_linear_1(V_f) / d_model, dim=1)
        V_f_star = torch.matmul(alpha_v.transpose(1, 2), V_f)
        alpha_t = F.softmax(self.uni_modal_linear_2(T_f) / d_model, dim=1)
        T_f_star = torch.matmul(alpha_t.transpose(1, 2), T_f)
        return V_f_star, T_f_star

    def retrieval_uni_attention(self, V_f, T_f):
        d_model = V_f.shape[-1]
        alpha_v = F.softmax(self.retrieval_uni_1(V_f) / d_model, dim=1)
        V_f_star = torch.matmul(alpha_v.transpose(1, 2), V_f)
        alpha_t = F.softmax(self.retrieval_uni_2(T_f) / d_model, dim=1)
        T_f_star = torch.matmul(alpha_t.transpose(1, 2), T_f)
        return V_f_star, T_f_star

    def forward(self, visual_features, text_features, user_features, labels=None):
        B = visual_features.shape[0]
        device = visual_features.device
        text_proj = self.text_proj(text_features)
        user_proj = self.user_proj(user_features)
        visual_global = self.encoder(visual_features)
        multimodal_query = torch.cat([visual_global, text_proj, user_proj], dim=-1)
        pmb_output = self.topo_pmb(multimodal_query)

        visual_feature = visual_global.unsqueeze(1)
        textual_feature = text_proj.unsqueeze(1)
        visual_emb = self.tanh(self.visual_embedding(visual_feature))
        textual_emb = self.tanh(self.textual_embedding(textual_feature))

        V_f, T_f = self.cross_modal_attention(visual_feature, textual_feature, visual_emb, textual_emb)
        V_f_star, T_f_star = self.uni_modal_attention(V_f, T_f)

        retrieved_feature = pmb_output['z_aug'].unsqueeze(1)
        retrieved_emb = self.tanh(self.retrieval_embedding(retrieved_feature))
        V_f_, T_f_ = self.retrieval_attention(retrieved_feature, retrieved_emb)
        V_f_star_, T_f_star_ = self.retrieval_uni_attention(V_f_, T_f_)

        similarity_weights = F.softmax(pmb_output['top_k_weights'], dim=1)
        retrieved_agg = (V_f_star_ * similarity_weights.unsqueeze(2)).sum(dim=1, keepdim=True)
        r_v_v = torch.mul(V_f_star, retrieved_agg)
        r_t_v = torch.mul(T_f_star, retrieved_agg)
        user_emb = user_proj.unsqueeze(1)

        pred_input = torch.cat([T_f_star, V_f_star, retrieved_agg, r_v_v, r_t_v, user_emb], dim=2).squeeze(1)
        output = self.prediction_head(pred_input).squeeze(-1)
        return {'output': output}


class GSARJambaTOPOV2ModelSMP(nn.Module):
    """SMP 模型使用旧版 TopologyAwarePMB（非多模态版本）"""
    def __init__(self, d_input=768, d_model=768, d_text=1024, d_user=9,
                 max_frames=100, m_heat=6, n_topic=32, top_k=5, n_blocks=2,
                 d_state=16, d_conv=4, expand=2, n_heads=12,
                 base_window_size=20, window_alpha=0.5, dt_rank=48,
                 delta_min=0.01, delta_max=1.0, temperature=1.0,
                 learnable_temperature=True, fixed_pmb_path=None,
                 use_load_balance=True, load_balance_alpha=0.01,
                 load_balance_beta=0.01, long_tail_exponent=1.5,
                 use_dppo=True, dppo_lambda=0.1, dppo_beta=0.5,
                 dppo_pool_size=1000, dppo_margin=0.5, dppo_gamma=1.0,
                 dropout=0.1):
        super().__init__()
        self.use_dppo = use_dppo
        self.dppo_lambda = dppo_lambda
        self.d_model = d_model

        self.encoder = create_gsar_jamba_encoder_v2(
            num_frames=max_frames, d_input=d_input, d_model=d_model,
            n_blocks=n_blocks, d_state=d_state, d_conv=d_conv, expand=expand,
            n_heads=n_heads, base_window_size=base_window_size,
            window_alpha=window_alpha, dt_rank=dt_rank,
            delta_min=delta_min, delta_max=delta_max, dropout=dropout)

        # 使用旧版 TopologyAwarePMB
        self.topo_pmb = create_topo_pmb(
            d_model=d_model, m_heat=m_heat, n_topic=n_topic, top_k=top_k,
            temperature=temperature, learnable_temperature=learnable_temperature,
            fixed_pmb_path=fixed_pmb_path, dropout=dropout)

        if use_load_balance:
            self.load_balance = create_load_balancing(
                m_heat=m_heat, n_topic=n_topic, gamma=load_balance_alpha,
                alpha=load_balance_alpha, beta=load_balance_beta,
                long_tail_exponent=long_tail_exponent)
        else:
            self.load_balance = None

        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))
        self.user_proj = nn.Sequential(
            nn.Linear(d_user, 64), nn.ReLU(), nn.Linear(64, d_model))

        self.visual_embedding = nn.Linear(d_model, d_model)
        self.textual_embedding = nn.Linear(d_model, d_model)
        self.retrieval_embedding = nn.Linear(d_model, d_model)

        self.dual_attention_linear_1 = nn.Linear(d_model * 2, d_model)
        self.dual_attention_linear_2 = nn.Linear(d_model * 2, d_model)
        self.cross_modal_linear_1 = nn.Linear(d_model * 2, d_model)
        self.cross_modal_linear_2 = nn.Linear(d_model * 2, d_model)
        self.uni_modal_linear_1 = nn.Linear(d_model, 1)
        self.uni_modal_linear_2 = nn.Linear(d_model, 1)

        self.retrieval_dual_1 = nn.Linear(d_model * 2, d_model)
        self.retrieval_dual_2 = nn.Linear(d_model * 2, d_model)
        self.retrieval_cross_1 = nn.Linear(d_model * 2, d_model)
        self.retrieval_cross_2 = nn.Linear(d_model * 2, d_model)
        self.retrieval_uni_1 = nn.Linear(d_model, 1)
        self.retrieval_uni_2 = nn.Linear(d_model, 1)

        self.tanh = nn.Tanh()
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 6, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))

        if use_dppo:
            self.dppo = DPPO(pool_size=dppo_pool_size, margin=dppo_margin,
                             beta=dppo_beta, gamma=dppo_gamma,
                             middle_sample_ratio=0.2, eps=1e-8)

    def cross_modal_attention(self, visual_feature, textual_feature, visual_emb, textual_emb):
        d_model = visual_feature.shape[-1]
        S = torch.matmul(visual_emb, textual_emb.transpose(1, 2)) / d_model
        T_p = torch.matmul(F.softmax(S, dim=1), textual_feature)
        V_p = torch.matmul(F.softmax(S.transpose(1, 2), dim=1), visual_feature)
        T_n = torch.matmul((-0.5) * F.softmax(S, dim=1), textual_feature)
        V_n = torch.matmul((-0.5) * F.softmax(S.transpose(1, 2), dim=1), visual_feature)
        T_star = self.tanh(self.dual_attention_linear_1(torch.cat([T_p, T_n], dim=2)))
        V_star = self.tanh(self.dual_attention_linear_2(torch.cat([V_p, V_n], dim=2)))
        V_f = self.tanh(self.cross_modal_linear_1(torch.cat([visual_feature, T_star], dim=2)))
        T_f = self.tanh(self.cross_modal_linear_2(torch.cat([textual_feature, V_star], dim=2)))
        return V_f, T_f

    def retrieval_attention(self, retrieved_feature, retrieved_emb):
        d_model = retrieved_feature.shape[-1]
        S = torch.matmul(retrieved_emb, retrieved_emb.transpose(1, 2)) / d_model
        T_p = torch.matmul(F.softmax(S, dim=1), retrieved_feature)
        V_p = torch.matmul(F.softmax(S.transpose(1, 2), dim=1), retrieved_feature)
        T_n = torch.matmul((-0.5) * F.softmax(S, dim=1), retrieved_feature)
        V_n = torch.matmul((-0.5) * F.softmax(S.transpose(1, 2), dim=1), retrieved_feature)
        T_star = self.tanh(self.retrieval_dual_1(torch.cat([T_p, T_n], dim=2)))
        V_star = self.tanh(self.retrieval_dual_2(torch.cat([V_p, V_n], dim=2)))
        V_f = self.tanh(self.retrieval_cross_1(torch.cat([retrieved_feature, T_star], dim=2)))
        T_f = self.tanh(self.retrieval_cross_2(torch.cat([retrieved_feature, V_star], dim=2)))
        return V_f, T_f

    def uni_modal_attention(self, V_f, T_f):
        d_model = V_f.shape[-1]
        alpha_v = F.softmax(self.uni_modal_linear_1(V_f) / d_model, dim=1)
        V_f_star = torch.matmul(alpha_v.transpose(1, 2), V_f)
        alpha_t = F.softmax(self.uni_modal_linear_2(T_f) / d_model, dim=1)
        T_f_star = torch.matmul(alpha_t.transpose(1, 2), T_f)
        return V_f_star, T_f_star

    def retrieval_uni_attention(self, V_f, T_f):
        d_model = V_f.shape[-1]
        alpha_v = F.softmax(self.retrieval_uni_1(V_f) / d_model, dim=1)
        V_f_star = torch.matmul(alpha_v.transpose(1, 2), V_f)
        alpha_t = F.softmax(self.retrieval_uni_2(T_f) / d_model, dim=1)
        T_f_star = torch.matmul(alpha_t.transpose(1, 2), T_f)
        return V_f_star, T_f_star

    def forward(self, visual_features, text_features, user_features, labels=None):
        B = visual_features.shape[0]
        device = visual_features.device
        text_proj = self.text_proj(text_features)
        user_proj = self.user_proj(user_features)
        visual_global = self.encoder(visual_features)
        # 旧版 topo_pmb 只接收 visual_global（768维）
        pmb_output = self.topo_pmb(visual_global)

        visual_feature = visual_global.unsqueeze(1)
        textual_feature = text_proj.unsqueeze(1)
        visual_emb = self.tanh(self.visual_embedding(visual_feature))
        textual_emb = self.tanh(self.textual_embedding(textual_feature))

        V_f, T_f = self.cross_modal_attention(visual_feature, textual_feature, visual_emb, textual_emb)
        V_f_star, T_f_star = self.uni_modal_attention(V_f, T_f)

        retrieved_feature = pmb_output['z_aug'].unsqueeze(1)
        retrieved_emb = self.tanh(self.retrieval_embedding(retrieved_feature))
        V_f_, T_f_ = self.retrieval_attention(retrieved_feature, retrieved_emb)
        V_f_star_, T_f_star_ = self.retrieval_uni_attention(V_f_, T_f_)

        similarity_weights = F.softmax(pmb_output['top_k_weights'], dim=1)
        retrieved_agg = (V_f_star_ * similarity_weights.unsqueeze(2)).sum(dim=1, keepdim=True)
        r_v_v = torch.mul(V_f_star, retrieved_agg)
        r_t_v = torch.mul(T_f_star, retrieved_agg)
        user_emb = user_proj.unsqueeze(1)

        pred_input = torch.cat([T_f_star, V_f_star, retrieved_agg, r_v_v, r_t_v, user_emb], dim=2).squeeze(1)
        output = self.prediction_head(pred_input).squeeze(-1)
        return {'output': output}


class GenericDataset(Dataset):
    def __init__(self, data_path, labels=None, max_frames=100):
        self.max_frames = max_frames
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.visual_features = data['visual_features']
        self.text_features = data['text_features']
        self.user_features = data['user_features']
        self.labels = labels if labels is not None else data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        visual = np.array(self.visual_features[idx])
        if visual.ndim == 1:
            visual = visual.reshape(1, -1)
        if len(visual) > self.max_frames:
            indices = np.linspace(0, len(visual)-1, self.max_frames, dtype=int)
            visual = visual[indices]
        return {
            'visual_features': torch.tensor(visual, dtype=torch.float32),
            'text_features': torch.tensor(self.text_features[idx], dtype=torch.float32),
            'user_features': torch.tensor(self.user_features[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def collate_fn(batch):
    visual_features = [item['visual_features'] for item in batch]
    max_len = max(f.shape[0] for f in visual_features)
    padded_visual = []
    for f in visual_features:
        if f.shape[0] < max_len:
            pad = torch.zeros(max_len - f.shape[0], f.shape[1])
            f = torch.cat([f, pad], dim=0)
        padded_visual.append(f)
    return {
        'visual_features': torch.stack(padded_visual),
        'text_features': torch.stack([item['text_features'] for item in batch]),
        'user_features': torch.stack([item['user_features'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch]),
    }


def compute_metrics(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    mae = np.mean(np.abs(predictions - labels))
    mse = np.mean((predictions - labels) ** 2)
    rmse = np.sqrt(mse)
    mask = labels != 0
    mape = np.mean(np.abs((labels[mask] - predictions[mask]) / labels[mask])) * 100 if mask.sum() > 0 else 0.0
    if labels.std() > 1e-8:
        nmse = mse / (labels.std() ** 2)
    else:
        nmse = 0.0
    ss_res = np.sum((labels - predictions) ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    src, _ = spearmanr(predictions, labels) if len(predictions) > 1 else (0.0, None)
    return {
        'mae': mae, 'mse': mse, 'rmse': rmse, 'nmse': nmse,
        'mape': mape, 'r2': r2, 'src': src if not np.isnan(src) else 0.0
    }


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            visual = batch['visual_features'].to(device)
            text = batch['text_features'].to(device)
            user = batch['user_features'].to(device)
            labels = batch['labels'].to(device)
            output = model(visual, text, user)
            all_preds.extend(output['output'].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return compute_metrics(torch.tensor(all_preds), torch.tensor(all_labels))


def compute_composite_popularity(data):
    TARGET_COLUMNS = ['video_comment_count', 'video_heart_count', 'video_play_count', 'video_share_count']
    all_labels = {target: [] for target in TARGET_COLUMNS}
    for split in ['train', 'valid', 'test']:
        for target in TARGET_COLUMNS:
            all_labels[target].extend(data[split]['labels'][target])
    min_max = {}
    for target in TARGET_COLUMNS:
        all_labels[target] = np.array(all_labels[target])
        min_max[target] = {'min': all_labels[target].min(), 'max': all_labels[target].max()}
    result = {}
    for split in ['train', 'valid', 'test']:
        split_data = data[split]
        normalized = {}
        for target in TARGET_COLUMNS:
            values = np.array(split_data['labels'][target])
            normalized[target] = (values - min_max[target]['min']) / (min_max[target]['max'] - min_max[target]['min'] + 1e-8)
        popularity = (normalized['video_play_count'] + normalized['video_heart_count'] +
                      normalized['video_share_count'] + normalized['video_comment_count']) / 4.0
        result[split] = {'popularity': popularity.astype(np.float32), 'normalized': normalized}
    return result, min_max


def find_pmb_path(pmb_name):
    """查找 PMB 文件路径"""
    paths = [
        f"/root/GSAR-Jamba-TOPO-Public/.fixed_assets/{pmb_name}",
        f"/root/gsar_jamba_topo/.fixed_assets/{pmb_name}",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def find_data_path(dataset_type):
    """查找数据路径"""
    if dataset_type == 'microlens_31frames':
        paths = [
            '/root/GSAR-Jamba-TOPO-Public/data_MicroLens/processed_31frames',
            '/root/gsar_jamba_topo/data_MicroLens/processed_31frames',
        ]
    elif dataset_type == 'microlens_100frames':
        paths = [
            '/root/gsar_jamba_topo/data_MicroLens/processed_100frames',
        ]
    elif dataset_type == 'microlens_processed':
        paths = [
            '/root/GSAR-Jamba-TOPO-Public/data_MicroLens_processed/processed_features',
            '/root/gsar_jamba_topo/data_MicroLens_processed/processed_features',
        ]
    elif dataset_type == 'smp':
        paths = ['/root/GSAR-Jamba-TOPO-Public/data/processed_features']
    elif dataset_type == 'informs':
        paths = ['/root/GSAR-Jamba-TOPO-Public/data_v2/processed_features']
    else:
        return None
    for p in paths:
        if os.path.exists(f"{p}/test.pkl"):
            return p
    return None


def try_load_model(ckpt, model_kwargs, device):
    """尝试加载模型，自动检测 max_frames"""
    state_dict = ckpt['model_state_dict']
    # 从 checkpoint 的 temporal_encoding 推断 max_frames
    pe_key = 'encoder.temporal_encoding.pe'
    if pe_key in state_dict:
        max_frames = state_dict[pe_key].shape[1]
        print(f"  Detected max_frames={max_frames} from checkpoint")
        model_kwargs['max_frames'] = max_frames

    model = GSARJambaTOPOV2Model(**model_kwargs).to(device)
    model.load_state_dict(state_dict, strict=True)
    return model, model_kwargs['max_frames']


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    results = {}

    # ============================================================
    # 1. multimodal_memory_bank 模型
    # ============================================================
    mm_dir = PROJECT_ROOT / "saved_models" / "multimodal_memory_bank"
    mm_models = sorted(mm_dir.glob("*.pt"))

    for model_path in mm_models:
        print(f"\n{'='*60}")
        print(f"Loading: {model_path.name}")
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        args = ckpt.get('args', {})
        epoch = ckpt.get('epoch', '?')

        dataset_path_orig = args.get('dataset_path', '')
        max_frames = args.get('max_frames', 31)
        fixed_pmb_path = args.get('fixed_pmb_path', '')
        d_text = args.get('d_text', 1024)
        d_user = args.get('d_user', 6)

        # 确定数据类型
        if 'processed_31frames' in dataset_path_orig:
            data_type = 'microlens_31frames'
        elif 'processed_100frames' in dataset_path_orig:
            data_type = 'microlens_100frames'
        elif 'processed_features' in dataset_path_orig and 'MicroLens' in dataset_path_orig:
            data_type = 'microlens_processed'
        else:
            data_type = 'microlens_31frames'

        dataset_path = find_data_path(data_type)
        if dataset_path is None:
            print(f"  SKIP: data not found for {data_type}")
            continue

        pmb_name = os.path.basename(fixed_pmb_path)
        pmb_path = find_pmb_path(pmb_name)
        if pmb_path is None:
            print(f"  SKIP: PMB not found: {pmb_name}")
            continue

        print(f"  Epoch: {epoch}, Data: {dataset_path}, PMB: {pmb_name}")

        test_dataset = GenericDataset(f"{dataset_path}/test.pkl", max_frames=max_frames)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        print(f"  Test samples: {len(test_dataset)}")

        model_kwargs = dict(
            d_input=768, d_model=768, d_text=d_text, d_user=d_user,
            max_frames=max_frames, m_heat=6, n_topic=32, top_k=5,
            n_blocks=args.get('n_blocks', 2), d_state=16, d_conv=4, expand=2,
            n_heads=12, base_window_size=args.get('base_window_size', 10),
            window_alpha=args.get('window_alpha', 0.5), dt_rank=48,
            delta_min=args.get('delta_min', 0.01), delta_max=args.get('delta_max', 1.0),
            temperature=args.get('temperature', 1.0),
            learnable_temperature=args.get('learnable_temperature', True),
            fixed_pmb_path=pmb_path,
            use_load_balance=args.get('use_load_balance', True),
            load_balance_alpha=args.get('load_balance_alpha', 0.01),
            load_balance_beta=args.get('load_balance_beta', 0.01),
            long_tail_exponent=args.get('long_tail_exponent', 1.5),
            use_dppo=args.get('use_dppo', True),
            dppo_lambda=args.get('dppo_lambda', 0.1),
            dropout=0.1,
        )

        try:
            model, detected_mf = try_load_model(ckpt, model_kwargs, device)
            # 如果检测到的 max_frames 和数据集不同，需要重建数据集
            if detected_mf != max_frames:
                test_dataset = GenericDataset(f"{dataset_path}/test.pkl", max_frames=detected_mf)
                test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

            metrics = evaluate(model, test_loader, device)
            key = f"MM/{model_path.name[:55]}"
            results[key] = metrics
            print(f"  MAE={metrics['mae']:.4f}  MSE={metrics['mse']:.4f}  RMSE={metrics['rmse']:.4f}  "
                  f"NMSE={metrics['nmse']:.4f}  MAPE={metrics['mape']:.2f}%  R2={metrics['r2']:.4f}  SRC={metrics['src']:.4f}")
            del model
        except Exception as e:
            print(f"  Error: {e}")

        torch.cuda.empty_cache()

    # ============================================================
    # 2. full_retrain 模型
    # ============================================================
    fr_dir = PROJECT_ROOT / "saved_models" / "full_retrain"

    # --- best_model_microlens.pt ---
    microlens_path = fr_dir / "best_model_microlens.pt"
    if microlens_path.exists():
        print(f"\n{'='*60}")
        print(f"Loading: best_model_microlens.pt")
        ckpt = torch.load(microlens_path, map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', '?')
        print(f"  Epoch: {epoch}")

        # 使用31帧数据
        dataset_path = find_data_path('microlens_31frames')
        if dataset_path is None:
            print("  SKIP: MicroLens 31frames data not found")
        else:
            with open(f"{dataset_path}/test.pkl", 'rb') as f:
                test_data = pickle.load(f)
            text_dim = len(test_data['text_features'][0])
            user_dim = len(test_data['user_features'][0])
            print(f"  Data: {dataset_path}, text_dim={text_dim}, user_dim={user_dim}")

            pmb_path = find_pmb_path('master_grid_M6_N32_microlens_31frames_multimodal.pkl')
            if pmb_path is None:
                print("  SKIP: PMB not found")
            else:
                model_kwargs = dict(
                    d_input=768, d_model=768, d_text=text_dim, d_user=user_dim,
                    max_frames=31, m_heat=6, n_topic=32, top_k=5,
                    n_blocks=2, d_state=16, d_conv=4, expand=2, n_heads=12,
                    base_window_size=10, window_alpha=0.5, dt_rank=48,
                    delta_min=0.001, delta_max=0.1,
                    temperature=1.0, learnable_temperature=True,
                    fixed_pmb_path=pmb_path,
                    use_load_balance=True, load_balance_alpha=0.01,
                    load_balance_beta=0.01, long_tail_exponent=1.5,
                    use_dppo=True, dppo_lambda=0.1, dropout=0.1,
                )

                try:
                    model, detected_mf = try_load_model(ckpt, model_kwargs, device)
                    test_dataset = GenericDataset(f"{dataset_path}/test.pkl", max_frames=detected_mf)
                    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
                    print(f"  Test samples: {len(test_dataset)}, max_frames={detected_mf}")

                    metrics = evaluate(model, test_loader, device)
                    results["FR/microlens"] = metrics
                    print(f"  MAE={metrics['mae']:.4f}  MSE={metrics['mse']:.4f}  RMSE={metrics['rmse']:.4f}  "
                          f"NMSE={metrics['nmse']:.4f}  MAPE={metrics['mape']:.2f}%  R2={metrics['r2']:.4f}  SRC={metrics['src']:.4f}")
                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()

                try:
                    del model
                except:
                    pass
                torch.cuda.empty_cache()

    # --- best_model_smp_mra.pt ---
    smp_path = fr_dir / "best_model_smp_mra.pt"
    if smp_path.exists():
        print(f"\n{'='*60}")
        print(f"Loading: best_model_smp_mra.pt")
        ckpt = torch.load(smp_path, map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', '?')
        print(f"  Epoch: {epoch}")

        dataset_path = find_data_path('smp')
        if dataset_path is None:
            print("  SKIP: SMP data not found")
        else:
            with open(f"{dataset_path}/test.pkl", 'rb') as f:
                test_data = pickle.load(f)
            text_dim = len(test_data['text_features'][0])
            user_dim = len(test_data['user_features'][0])
            print(f"  Data: {dataset_path}, text_dim={text_dim}, user_dim={user_dim}")

            pmb_path = find_pmb_path('master_grid_M6_N32_multimodal.pkl')
            if pmb_path is None:
                print("  SKIP: PMB not found")
            else:
                # SMP 使用旧版 TopologyAwarePMB
                model_kwargs = dict(
                    d_input=768, d_model=768, d_text=text_dim, d_user=user_dim,
                    max_frames=100, m_heat=6, n_topic=32, top_k=5,
                    n_blocks=2, d_state=16, d_conv=4, expand=2, n_heads=12,
                    base_window_size=20, window_alpha=0.5, dt_rank=48,
                    delta_min=0.01, delta_max=1.0,
                    temperature=1.0, learnable_temperature=True,
                    fixed_pmb_path=pmb_path,
                    use_load_balance=True, load_balance_alpha=0.01,
                    load_balance_beta=0.01, long_tail_exponent=1.5,
                    use_dppo=True, dppo_lambda=0.1, dropout=0.1,
                )

                try:
                    # 使用 SMP 专用模型类
                    state_dict = ckpt['model_state_dict']
                    pe_key = 'encoder.temporal_encoding.pe'
                    if pe_key in state_dict:
                        detected_mf = state_dict[pe_key].shape[1]
                        model_kwargs['max_frames'] = detected_mf
                        print(f"  Detected max_frames={detected_mf} from checkpoint")

                    model = GSARJambaTOPOV2ModelSMP(**model_kwargs).to(device)
                    model.load_state_dict(state_dict, strict=True)
                    test_dataset = GenericDataset(f"{dataset_path}/test.pkl", max_frames=model_kwargs['max_frames'])
                    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
                    print(f"  Test samples: {len(test_dataset)}, max_frames={model_kwargs['max_frames']}")

                    metrics = evaluate(model, test_loader, device)
                    results["FR/smp_mra"] = metrics
                    print(f"  MAE={metrics['mae']:.4f}  MSE={metrics['mse']:.4f}  RMSE={metrics['rmse']:.4f}  "
                          f"NMSE={metrics['nmse']:.4f}  MAPE={metrics['mape']:.2f}%  R2={metrics['r2']:.4f}  SRC={metrics['src']:.4f}")
                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()

                try:
                    del model
                except:
                    pass
                torch.cuda.empty_cache()

    # --- best_model_informs.pt ---
    informs_path = fr_dir / "best_model_informs.pt"
    if informs_path.exists():
        print(f"\n{'='*60}")
        print(f"Loading: best_model_informs.pt")
        ckpt = torch.load(informs_path, map_location='cpu', weights_only=False)
        epoch = ckpt.get('epoch', '?')
        print(f"  Epoch: {epoch}")

        dataset_path = find_data_path('informs')
        if dataset_path is None:
            print("  SKIP: Informs data not found")
        else:
            # 计算复合热度
            train_data = pickle.load(open(f"{dataset_path}/train.pkl", 'rb'))
            valid_data = pickle.load(open(f"{dataset_path}/valid.pkl", 'rb'))
            test_data = pickle.load(open(f"{dataset_path}/test.pkl", 'rb'))
            data = {'train': train_data, 'valid': valid_data, 'test': test_data}
            popularity_data, _ = compute_composite_popularity(data)

            text_dim = len(train_data['text_features'][0])
            user_dim = len(train_data['user_features'][0])
            print(f"  Data: {dataset_path}, text_dim={text_dim}, user_dim={user_dim}")

            pmb_path = find_pmb_path('master_grid_M6_N32_informs_multimodal.pkl')
            if pmb_path is None:
                print("  SKIP: PMB not found")
            else:
                model_kwargs = dict(
                    d_input=768, d_model=768, d_text=text_dim, d_user=user_dim,
                    max_frames=100, m_heat=6, n_topic=32, top_k=5,
                    n_blocks=2, d_state=16, d_conv=4, expand=2, n_heads=12,
                    base_window_size=20, window_alpha=0.5, dt_rank=48,
                    delta_min=0.01, delta_max=1.0,
                    temperature=1.0, learnable_temperature=True,
                    fixed_pmb_path=pmb_path,
                    use_load_balance=True, load_balance_alpha=0.01,
                    load_balance_beta=0.01, long_tail_exponent=1.5,
                    use_dppo=True, dppo_lambda=0.05, dppo_beta=0.3, dropout=0.1,
                )

                try:
                    model, detected_mf = try_load_model(ckpt, model_kwargs, device)
                    test_dataset = GenericDataset(f"{dataset_path}/test.pkl",
                                                  labels=popularity_data['test']['popularity'],
                                                  max_frames=detected_mf)
                    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
                    print(f"  Test samples: {len(test_dataset)}, max_frames={detected_mf}")

                    metrics = evaluate(model, test_loader, device)
                    results["FR/informs"] = metrics
                    print(f"  MAE={metrics['mae']:.4f}  MSE={metrics['mse']:.4f}  RMSE={metrics['rmse']:.4f}  "
                          f"NMSE={metrics['nmse']:.4f}  MAPE={metrics['mape']:.2f}%  R2={metrics['r2']:.4f}  SRC={metrics['src']:.4f}")
                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()

                try:
                    del model
                except:
                    pass
                torch.cuda.empty_cache()

    # ============================================================
    # 汇总结果
    # ============================================================
    print(f"\n\n{'='*100}")
    print("INFERENCE RESULTS SUMMARY")
    print(f"{'='*100}")
    print(f"{'Model':<58} {'MAE':>8} {'MSE':>8} {'RMSE':>8} {'NMSE':>8} {'MAPE':>8} {'R2':>8} {'SRC':>8}")
    print("-" * 114)
    for key, m in results.items():
        print(f"{key:<58} {m['mae']:>8.4f} {m['mse']:>8.4f} {m['rmse']:>8.4f} {m['nmse']:>8.4f} {m['mape']:>7.2f}% {m['r2']:>8.4f} {m['src']:>8.4f}")


if __name__ == '__main__':
    main()
