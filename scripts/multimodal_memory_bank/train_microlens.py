"""
GSAR-Jamba-TOPO V2 MicroLens 31帧版本训练脚本 - 多模态记忆库版本
使用 visual + text + user 多模态融合记忆库
"""

import os
import sys
import argparse
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.stats import spearmanr
import random

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.gsar_jamba_v2 import GSARJambaEncoderV2, create_gsar_jamba_encoder_v2
from models.topo_pmb_multimodal import TopologyAwarePMBMultimodal, create_topo_pmb_multimodal
from core.load_balance import CombinedLoadBalancing, create_load_balancing

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = PROJECT_ROOT / "logs" / "multimodal_memory_bank"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"train_microlens_multimodal_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger('MicroLens_31Frames_Multimodal')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True)
    except:
        pass


class MicroLensDataset(Dataset):
    def __init__(self, data_path: str, max_frames: int = 1):
        self.max_frames = max_frames
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.visual_features = data['visual_features']
        self.text_features = data['text_features']
        self.user_features = data['user_features']
        self.labels = data['labels']
    
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
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'nmse': nmse,
        'mape': mape,
        'r2': r2,
        'src': src if not np.isnan(src) else 0.0
    }


class DPPO:
    def __init__(
        self,
        pool_size: int = 1000,
        margin: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
        middle_sample_ratio: float = 0.2,
        eps: float = 1e-8,
    ):
        self.pool_size = pool_size
        self.margin = margin
        self.beta = beta
        self.gamma = gamma
        self.middle_sample_ratio = middle_sample_ratio
        self.eps = eps
        self.sample_pool: List[Dict] = []
    
    def update_pool(
        self,
        visual_global: torch.Tensor,
        text_proj: torch.Tensor,
        user_proj: torch.Tensor,
        routing_probs: torch.Tensor,
        labels: torch.Tensor,
    ):
        B = visual_global.shape[0]
        for i in range(B):
            self.sample_pool.append({
                'visual_global': visual_global[i].detach().cpu(),
                'text_proj': text_proj[i].detach().cpu(),
                'user_proj': user_proj[i].detach().cpu(),
                'routing_probs': routing_probs[i].detach().cpu(),
                'label': labels[i].detach().cpu(),
            })
        if len(self.sample_pool) > self.pool_size:
            self.sample_pool = self.sample_pool[-self.pool_size:]
    
    def sample_pairs(self, batch_size: int) -> List:
        if len(self.sample_pool) < 10:
            return []
        
        pairs = []
        sorted_pool = sorted(self.sample_pool, key=lambda x: x['label'].item())
        n = len(sorted_pool)
        
        num_pairs = min(4, batch_size)
        num_extreme = int(num_pairs * (1 - self.middle_sample_ratio))
        num_middle = num_pairs - num_extreme
        
        for _ in range(num_extreme):
            high_idx = np.random.randint(int(n * 0.6), n)
            low_idx = np.random.randint(0, int(n * 0.4))
            win = sorted_pool[high_idx]
            lose = sorted_pool[low_idx]
            label_diff = (win['label'] - lose['label']).item()
            if label_diff > self.margin:
                pairs.append((win, lose, label_diff, 'extreme'))
        
        for _ in range(num_middle):
            sample_type = np.random.choice(['high_mid', 'mid_low', 'mid_mid'], p=[0.4, 0.4, 0.2])
            
            if sample_type == 'high_mid':
                high_idx = np.random.randint(int(n * 0.6), n)
                mid_idx = np.random.randint(int(n * 0.3), int(n * 0.7))
                win = sorted_pool[high_idx]
                lose = sorted_pool[mid_idx]
            elif sample_type == 'mid_low':
                mid_idx = np.random.randint(int(n * 0.3), int(n * 0.7))
                low_idx = np.random.randint(0, int(n * 0.4))
                win = sorted_pool[mid_idx]
                lose = sorted_pool[low_idx]
            else:
                mid_idx1 = np.random.randint(int(n * 0.3), int(n * 0.7))
                mid_idx2 = np.random.randint(int(n * 0.3), int(n * 0.7))
                if mid_idx1 > mid_idx2:
                    win = sorted_pool[mid_idx1]
                    lose = sorted_pool[mid_idx2]
                else:
                    win = sorted_pool[mid_idx2]
                    lose = sorted_pool[mid_idx1]
            
            label_diff = (win['label'] - lose['label']).item()
            if label_diff > 0.1:
                pairs.append((win, lose, label_diff, 'middle'))
        
        return pairs
    
    def compute_loss(
        self,
        win_sample: Dict,
        lose_sample: Dict,
        label_diff: float,
        model: nn.Module,
        device: torch.device,
        sample_type: str = 'extreme',
    ) -> torch.Tensor:
        win_visual = win_sample['visual_global'].unsqueeze(0).to(device)
        lose_visual = lose_sample['visual_global'].unsqueeze(0).to(device)
        
        win_query = win_visual
        lose_query = lose_visual
        
        pmb = model.topo_pmb
        
        routing_win = pmb.forward(win_query, use_top_k=False)
        routing_lose = pmb.forward(lose_query, use_top_k=False)
        
        routing_win = routing_win['joint_weights'].view(-1)
        routing_lose = routing_lose['joint_weights'].view(-1)
        
        log_ratio = torch.log(routing_win + self.eps) - torch.log(routing_lose + self.eps)
        scaled_delta = self.beta * log_ratio.sum()
        
        if sample_type == 'middle':
            weight = 0.5
        else:
            weight = 1.0
        
        loss = -weight * torch.log(torch.sigmoid(scaled_delta) + self.eps)
        
        return loss


class GSARJambaTOPOV2Model(nn.Module):
    def __init__(
        self,
        d_input: int = 768,
        d_model: int = 768,
        d_text: int = 1024,
        d_user: int = 6,
        max_frames: int = 1,
        m_heat: int = 6,
        n_topic: int = 32,
        top_k: int = 5,
        n_blocks: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_heads: int = 12,
        base_window_size: int = 10,
        window_alpha: float = 0.5,
        dt_rank: int = 48,
        delta_min: float = 0.01,
        delta_max: float = 1.0,
        temperature: float = 1.0,
        learnable_temperature: bool = True,
        fixed_pmb_path: Optional[str] = None,
        use_load_balance: bool = True,
        load_balance_alpha: float = 0.01,
        load_balance_beta: float = 0.01,
        long_tail_exponent: float = 1.5,
        use_dppo: bool = True,
        dppo_lambda: float = 0.1,
        dppo_beta: float = 0.5,
        dppo_pool_size: int = 1000,
        dppo_margin: float = 0.5,
        dppo_gamma: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.use_dppo = use_dppo
        self.dppo_lambda = dppo_lambda
        self.d_model = d_model
        
        self.encoder = create_gsar_jamba_encoder_v2(
            num_frames=max_frames,
            d_input=d_input,
            d_model=d_model,
            n_blocks=n_blocks,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_heads=n_heads,
            base_window_size=base_window_size,
            window_alpha=window_alpha,
            dt_rank=dt_rank,
            delta_min=delta_min,
            delta_max=delta_max,
            dropout=dropout,
        )
        
        self.topo_pmb = create_topo_pmb_multimodal(
            d_model=d_model,
            m_heat=m_heat,
            n_topic=n_topic,
            top_k=top_k,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            fixed_pmb_path=fixed_pmb_path,
            dropout=dropout,
        )
        
        if use_load_balance:
            self.load_balance = create_load_balancing(
                m_heat=m_heat,
                n_topic=n_topic,
                gamma=load_balance_alpha,
                alpha=load_balance_alpha,
                beta=load_balance_beta,
                long_tail_exponent=long_tail_exponent,
            )
        else:
            self.load_balance = None
        
        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        
        self.user_proj = nn.Sequential(
            nn.Linear(d_user, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )
        
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
            nn.Linear(d_model * 6, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        if use_dppo:
            self.dppo = DPPO(
                pool_size=dppo_pool_size,
                margin=dppo_margin,
                beta=dppo_beta,
                gamma=dppo_gamma,
                middle_sample_ratio=0.2,
                eps=1e-8,
            )
    
    def cross_modal_attention(
        self,
        visual_feature: torch.Tensor,
        textual_feature: torch.Tensor,
        visual_emb: torch.Tensor,
        textual_emb: torch.Tensor,
    ):
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
    
    def retrieval_attention(
        self,
        retrieved_feature: torch.Tensor,
        retrieved_emb: torch.Tensor,
    ):
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
    
    def uni_modal_attention(self, V_f: torch.Tensor, T_f: torch.Tensor):
        d_model = V_f.shape[-1]
        alpha_v = F.softmax(self.uni_modal_linear_1(V_f) / d_model, dim=1)
        V_f_star = torch.matmul(alpha_v.transpose(1, 2), V_f)
        
        alpha_t = F.softmax(self.uni_modal_linear_2(T_f) / d_model, dim=1)
        T_f_star = torch.matmul(alpha_t.transpose(1, 2), T_f)
        
        return V_f_star, T_f_star
    
    def retrieval_uni_attention(self, V_f: torch.Tensor, T_f: torch.Tensor):
        d_model = V_f.shape[-1]
        alpha_v = F.softmax(self.retrieval_uni_1(V_f) / d_model, dim=1)
        V_f_star = torch.matmul(alpha_v.transpose(1, 2), V_f)
        
        alpha_t = F.softmax(self.retrieval_uni_2(T_f) / d_model, dim=1)
        T_f_star = torch.matmul(alpha_t.transpose(1, 2), T_f)
        
        return V_f_star, T_f_star
    
    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        user_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = visual_features.shape[0]
        device = visual_features.device
        
        text_proj = self.text_proj(text_features)
        user_proj = self.user_proj(user_features)
        
        visual_global = self.encoder(visual_features)
        
        # 使用多模态查询向量
        multimodal_query = torch.cat([visual_global, text_proj, user_proj], dim=-1)
        pmb_output = self.topo_pmb(multimodal_query)
        
        load_balance_loss = torch.tensor(0.0, device=device)
        if self.load_balance is not None:
            lb_output = self.load_balance(
                pmb_output['joint_weights'],
                pmb_output['heat_marginal'],
                pmb_output['topic_marginal'],
            )
            load_balance_loss = lb_output['loss']
        
        visual_feature = visual_global.unsqueeze(1)
        textual_feature = text_proj.unsqueeze(1)
        
        visual_emb = self.tanh(self.visual_embedding(visual_feature))
        textual_emb = self.tanh(self.textual_embedding(textual_feature))
        
        V_f, T_f = self.cross_modal_attention(
            visual_feature, textual_feature,
            visual_emb, textual_emb
        )
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
        
        pred_input = torch.cat([
            T_f_star,
            V_f_star,
            retrieved_agg,
            r_v_v,
            r_t_v,
            user_emb
        ], dim=2).squeeze(1)
        
        output = self.prediction_head(pred_input).squeeze(-1)
        
        mse_loss = torch.tensor(0.0, device=device)
        huber_loss = torch.tensor(0.0, device=device)
        dppo_loss = torch.tensor(0.0, device=device)
        total_loss = torch.tensor(0.0, device=device)
        
        if labels is not None:
            mse_loss = F.mse_loss(output, labels)
            huber_loss = F.huber_loss(output, labels, reduction='mean', delta=1.0)
            
            if self.use_dppo:
                self.dppo.update_pool(
                    visual_global, text_proj, user_proj,
                    pmb_output['joint_weights'].view(B, -1),
                    labels
                )
                
                dppo_pairs = self.dppo.sample_pairs(B)
                if dppo_pairs:
                    total_dppo = torch.tensor(0.0, device=device)
                    for win, lose, label_diff, sample_type in dppo_pairs:
                        pair_loss = self.dppo.compute_loss(win, lose, label_diff, self, device, sample_type)
                        total_dppo = total_dppo + pair_loss
                    dppo_loss = total_dppo / len(dppo_pairs)
            
            total_loss = huber_loss + self.dppo_lambda * dppo_loss + load_balance_loss
        
        return {
            'output': output,
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'huber_loss': huber_loss,
            'dppo_loss': dppo_loss,
            'load_balance_loss': load_balance_loss,
            'pmb_output': pmb_output,
            'temperature': pmb_output['temperature'],
        }


def train_epoch(model, dataloader, optimizer, device, epoch_seed, logger=None):
    set_seed(epoch_seed)
    model.train()
    total_loss_sum = 0
    mse_loss_sum = 0
    huber_loss_sum = 0
    dppo_loss_sum = 0
    lb_loss_sum = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, batch in enumerate(dataloader):
        visual = batch['visual_features'].to(device)
        text = batch['text_features'].to(device)
        user = batch['user_features'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        output = model(visual, text, user, labels)
        loss = output['total_loss']
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss_sum += loss.item()
        mse_loss_sum += output['mse_loss'].item()
        huber_loss_sum += output['huber_loss'].item()
        dppo_loss_sum += output['dppo_loss'].item()
        lb_loss_sum += output['load_balance_loss'].item()
        all_preds.extend(output['output'].detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    
    n_batches = len(dataloader)
    metrics = compute_metrics(torch.tensor(all_preds), torch.tensor(all_labels))
    metrics['total_loss'] = total_loss_sum / n_batches
    metrics['mse_loss'] = mse_loss_sum / n_batches
    metrics['huber_loss'] = huber_loss_sum / n_batches
    metrics['dppo_loss'] = dppo_loss_sum / n_batches
    metrics['load_balance_loss'] = lb_loss_sum / n_batches
    
    return metrics


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


def main():
    parser = argparse.ArgumentParser(description='MicroLens 31帧训练 - 多模态记忆库版本')
    parser.add_argument('--dataset_path', type=str, default='/root/GSAR-Jamba-TOPO-Public/data_MicroLens_processed/processed_features')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--epochs', type=int, default=350)
    parser.add_argument('--early_stop', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.00035)
    parser.add_argument('--encoder_lr_ratio', type=float, default=0.04)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--max_frames', type=int, default=31)
    parser.add_argument('--n_blocks', type=int, default=2)
    parser.add_argument('--base_window_size', type=int, default=10)
    parser.add_argument('--window_alpha', type=float, default=0.5)
    parser.add_argument('--delta_min', type=float, default=0.001)
    parser.add_argument('--delta_max', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--learnable_temperature', action='store_true', default=True)
    parser.add_argument('--load_balance_alpha', type=float, default=0.01)
    parser.add_argument('--load_balance_beta', type=float, default=0.01)
    parser.add_argument('--long_tail_exponent', type=float, default=1.5)
    parser.add_argument('--use_load_balance', action='store_true', default=True)
    parser.add_argument('--use_dppo', action='store_true', default=True)
    parser.add_argument('--dppo_lambda', type=float, default=0.1)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--fixed_pmb_path', type=str, default='/root/gsar_jamba_topo/.fixed_assets/master_grid_M6_N32_microlens_multimodal.pkl')

    args = parser.parse_args()
    
    set_seed(args.seed)
    logger.info(f"Random seed: {args.seed}")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"日志文件: {log_file}")
    
    logger.info(f"Loading training dataset: {args.dataset_path}/train.pkl")
    train_dataset = MicroLensDataset(f"{args.dataset_path}/train.pkl", args.max_frames)
    logger.info(f"Training dataset loaded: {len(train_dataset)} samples")
    
    logger.info(f"Loading validation dataset: {args.dataset_path}/valid.pkl")
    valid_dataset = MicroLensDataset(f"{args.dataset_path}/valid.pkl", args.max_frames)
    logger.info(f"Validation dataset loaded: {len(valid_dataset)} samples")
    
    logger.info(f"Loading test dataset: {args.dataset_path}/test.pkl")
    test_dataset = MicroLensDataset(f"{args.dataset_path}/test.pkl", args.max_frames)
    logger.info(f"Test dataset loaded: {len(test_dataset)} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 使用多模态记忆库
    fixed_pmb_path = args.fixed_pmb_path
    
    model = GSARJambaTOPOV2Model(
        d_input=768,
        d_model=768,
        d_text=1024,
        d_user=6,
        max_frames=args.max_frames,
        m_heat=6,
        n_topic=32,
        top_k=args.top_k,
        n_blocks=args.n_blocks,
        d_state=16,
        d_conv=4,
        expand=2,
        n_heads=12,
        base_window_size=args.base_window_size,
        window_alpha=args.window_alpha,
        dt_rank=48,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        temperature=args.temperature,
        learnable_temperature=args.learnable_temperature,
        fixed_pmb_path=fixed_pmb_path,
        use_load_balance=args.use_load_balance,
        load_balance_alpha=args.load_balance_alpha,
        load_balance_beta=args.load_balance_beta,
        long_tail_exponent=args.long_tail_exponent,
        use_dppo=args.use_dppo,
        dppo_lambda=args.dppo_lambda,
        dropout=0.1,
    ).to(device)
    
    encoder_params = list(model.encoder.parameters())
    pmb_params = list(model.topo_pmb.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith('encoder') and not n.startswith('topo_pmb')]
    
    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': args.lr * args.encoder_lr_ratio, 'weight_decay': 1e-5},
        {'params': pmb_params, 'lr': args.lr, 'weight_decay': 1e-5},
        {'params': other_params, 'lr': args.lr, 'weight_decay': 1e-5},
    ])
    
    logger.info("=" * 60)
    logger.info("GSAR-Jamba-TOPO Training Configuration (Multimodal Memory Bank)")
    logger.info("=" * 60)
    logger.info(f"Random Seed: {args.seed}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Model: GSAR-Jamba-TOPO (AA-SSM) with Multimodal Memory Bank")
    logger.info(f"Dataset: MicroLens-31Frames")
    logger.info(f"Memory Bank: {fixed_pmb_path}")
    logger.info(f"Optimizer: Adam(encoder_lr={args.lr*args.encoder_lr_ratio}, other_lr={args.lr})")
    logger.info(f"Total Epoch: {args.epochs}")
    logger.info(f"Early Stop: {args.early_stop}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Max frames: {args.max_frames}")
    logger.info(f"N blocks: {args.n_blocks}")
    logger.info(f"Delta range: [{args.delta_min}, {args.delta_max}]")
    logger.info(f"Temperature: {args.temperature} (learnable: {args.learnable_temperature})")
    logger.info(f"Load Balance: alpha={args.load_balance_alpha}, beta={args.load_balance_beta}")
    logger.info(f"Long Tail Exponent: {args.long_tail_exponent}")
    logger.info(f"DPPO: {'ENABLED' if args.use_dppo else 'DISABLED'}")
    logger.info(f"Training Starts!")
    logger.info("=" * 60)
    
    best_valid_mae = float('inf')
    best_valid_src = -float('inf')
    best_test_metrics = None
    early_stop_count = 0
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"-----------------------------------Epoch {epoch} Start!-----------------------------------")
        
        epoch_seed = args.seed + epoch
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch_seed, logger)
        logger.info(f"[ Epoch {epoch} (train) ]: total_loss = {train_metrics['total_loss']:.6f}, huber = {train_metrics['huber_loss']:.6f}, dppo = {train_metrics['dppo_loss']:.6f}, lb = {train_metrics['load_balance_loss']:.6f}")
        
        valid_metrics = evaluate(model, valid_loader, device)
        logger.info("=" * 60)
        logger.info("Valid Metrics:")
        logger.info(f"  MAE:  {valid_metrics['mae']:.6f}")
        logger.info(f"  MSE:  {valid_metrics['mse']:.6f}")
        logger.info(f"  RMSE: {valid_metrics['rmse']:.6f}")
        logger.info(f"  NMSE: {valid_metrics['nmse']:.6f}")
        logger.info(f"  MAPE: {valid_metrics['mape']:.4f}%")
        logger.info(f"  R2:   {valid_metrics['r2']:.6f}")
        logger.info(f"  SRC:  {valid_metrics['src']:.6f}")
        logger.info("=" * 60)
        
        test_metrics = evaluate(model, test_loader, device)
        logger.info("Test Metrics:")
        logger.info(f"  MAE:  {test_metrics['mae']:.6f}")
        logger.info(f"  MSE:  {test_metrics['mse']:.6f}")
        logger.info(f"  RMSE: {test_metrics['rmse']:.6f}")
        logger.info(f"  NMSE: {test_metrics['nmse']:.6f}")
        logger.info(f"  MAPE: {test_metrics['mape']:.4f}%")
        logger.info(f"  R2:   {test_metrics['r2']:.6f}")
        logger.info(f"  SRC:  {test_metrics['src']:.6f}")
        logger.info("=" * 60)
        
        improved = False
        if valid_metrics['mae'] < best_valid_mae:
            improvement_mae = (best_valid_mae - valid_metrics['mae']) / best_valid_mae * 100 if best_valid_mae != float('inf') else 100
            improvement_src = (valid_metrics['src'] - best_valid_src) / abs(best_valid_src) * 100 if best_valid_src != -float('inf') else 100
            
            if best_valid_mae == float('inf'):
                logger.critical(f"=== Epoch {epoch}: Initializing best metrics ===")
                improved = True
            elif improvement_mae > 0.1 or improvement_src > 0.1:
                logger.critical(f"Improvement: MAE=+{improvement_mae:.2f}%, SRC=+{improvement_src:.2f}%")
                logger.critical(f"*** New Best Model at Epoch {epoch} ***")
                improved = True
        
        if improved:
            best_valid_mae = valid_metrics['mae']
            best_valid_src = valid_metrics['src']
            best_test_metrics = test_metrics.copy()
            early_stop_count = 0

            # 保存 best model
            save_dir = Path("/root/GSAR-Jamba-TOPO-Public/saved_models/multimodal_memory_bank")
            save_dir.mkdir(parents=True, exist_ok=True)
            model_name = f"best_model_microlens_{os.path.basename(fixed_pmb_path).replace('.pkl','')}_{timestamp}.pt"
            save_path = save_dir / model_name
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_test_metrics': best_test_metrics,
                'fixed_pmb_path': fixed_pmb_path,
                'args': vars(args),
            }, save_path)
            logger.info(f"Best model saved to: {save_path}")
        else:
            early_stop_count += 1
            logger.critical(f"No improvement. Early stop count: {early_stop_count}/{args.early_stop}")
        
        if early_stop_count >= args.early_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info("Best Test Results:")
    logger.info(f"  MAE:  {best_test_metrics['mae']:.6f}")
    logger.info(f"  MSE:  {best_test_metrics['mse']:.6f}")
    logger.info(f"  NMSE: {best_test_metrics['nmse']:.6f}")
    logger.info(f"  MAPE: {best_test_metrics['mape']:.4f}%")
    logger.info(f"  R2:   {best_test_metrics['r2']:.6f}")
    logger.info(f"  SRC:  {best_test_metrics['src']:.6f}")


if __name__ == '__main__':
    main()
