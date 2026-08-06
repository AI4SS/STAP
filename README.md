# Seeing Further and Wider: Joint Spatio-Temporal Enlargement for Micro-Video Popularity Prediction

[![arXiv](https://img.shields.io/badge/arXiv-2604.20311-b31b1b?style=flat-square)](https://arxiv.org/abs/2604.20311)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*Accepted by ACM Multimedia (MM) 2026*

This repository contains the official implementation of the paper **"Seeing Further and Wider: Joint Spatio-Temporal Enlargement for Micro-Video Popularity Prediction"**.

> **Abstract**: Micro-video popularity prediction (MVPP) aims to forecast the future popularity of videos on online media, which is essential for applications such as content recommendation and traffic allocation. Existing approaches suffer from limitations in both dimensions: temporally, they rely on sparse short-range sampling that restricts content perception; spatially, they depend on flat retrieval memory with limited capacity and low efficiency, hindering scalable knowledge utilization. To overcome these limitations, we propose a unified framework that achieves joint spatio-temporal enlargement, enabling precise perception of extremely long video sequences while supporting a scalable memory bank that can infinitely expand to incorporate all relevant historical videos.

---

## ✨ Highlight

- **Temporal Enlargement**: A frame scoring module extracts highlight cues from video frames through two complementary pathways — sparse sampling and dense perception — which are adaptively fused for robust long-sequence content understanding.
- **Spatial Enlargement**: A **Topology-Aware Memory Bank** hierarchically clusters historically relevant content based on topological relationships. Instead of directly expanding memory capacity, encoder features of the corresponding clusters are updated when incorporating new videos, enabling unbounded historical association without unbounded storage growth.
- **State-of-the-art results**: Consistently outperforms **11 strong baselines** across three widely used MVPP benchmarks (SMP, MicroLens, and Informs), with robust improvements in both prediction accuracy and ranking consistency.

---

## 🗓️ Timeline

- **[2026-04-22]** Paper released on **arXiv** ([2604.20311](https://arxiv.org/abs/2604.20311)).
- **[2026]** Paper accepted by **ACM Multimedia (MM) 2026**.
- **[2026-08-06]** Initial code release.

---

## 📂 Project Structure

```
STAP/
├── models/                          # Core model implementations
│   ├── gsar_jamba_v2.py             # Temporal Enlargement encoder (GSAR-Jamba V2)
│   ├── topo_pmb.py                  # Topology-Aware PMB (visual-only)
│   ├── topo_pmb_multimodal.py       # Topology-Aware PMB (multi-modal)
│   ├── parametric_memory_bank.py    # Parameterized memory bank
│   └── load_balance.py              # Load balancing
├── core/                            # Shared components
│   ├── topo_pmb.py                  # Topology-Aware PMB core
│   └── load_balance.py              # Asymmetric marginal load balancing
├── memory_bank_construction/        # Memory bank construction for each dataset
├── datasets/
│   ├── SMP/preprocessing/           # SMP data preprocessing
│   ├── MicroLens/preprocessing/     # MicroLens data preprocessing
│   └── Informs/preprocessing/       # Informs data preprocessing
├── scripts/multimodal_memory_bank/  # Training scripts
├── run_inference.py                 # Inference / evaluation script
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

---

## 🧪 Usage

### 1. Preparation

```bash
conda create -n stap python=3.10
conda activate stap
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Data Preprocessing

Each dataset ships with a preprocessing pipeline under `datasets/<dataset>/preprocessing/`:

- **SMP**: `dataset_split.py` → `video_frame_capture.py` → `visual_engineering.py` → `textual_engineering.py` → `user_feature_engineering.py`
- **MicroLens**: `extract_microlens_frames_30.py` → `extract_microlens_features.py` → `preprocess_microlens_31frames_v3.py`
- **Informs**: `run_pipeline.py` (end-to-end, supports `--skip_visual`, `--skip_textual`, `--skip_user`, etc.)

### 3. Memory Bank Construction

Build the topology-aware memory bank for a given dataset (saved to `.fixed_assets/`):

```bash
python memory_bank_construction/build_smp_multimodal_memory_bank.py
python memory_bank_construction/build_microlens_31frames_multimodal_memory_bank.py
python memory_bank_construction/build_informs_multimodal_memory_bank.py
```

### 4. Training

```bash
# SMP
python scripts/multimodal_memory_bank/train_topo_smpd.py --dataset_path /path/to/smp_data --max_frames 100

# MicroLens (31 frames)
python scripts/multimodal_memory_bank/train_microlens_31frames.py --dataset_path /path/to/processed_31frames

# Informs
python scripts/multimodal_memory_bank/train_informs.py --dataset_path /path/to/informs_data
```

Key options include `--lr`, `--epochs`, `--batch_size`, `--n_blocks`, `--top_k`, `--use_dppo`, `--use_load_balance`, and `--fixed_pmb_path`.

### 5. Evaluation

Run inference on all saved checkpoints (MAE / MSE / RMSE / NMSE / MAPE / R² / SRC):

```bash
python run_inference.py
```

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@misc{wang2026seeing,
  title={Seeing Further and Wider: Joint Spatio-Temporal Enlargement for Micro-Video Popularity Prediction},
  author={Wang, Dali and Zhang, Yunyao and Yu, Junqing and Chen, Yi-Ping Phoebe and Xu, Chen and Song, Zikai},
  year={2026},
  eprint={2604.20311},
  archivePrefix={arXiv},
  primaryClass={cs.MM},
  url={https://arxiv.org/abs/2604.20311}
}
```

---

## 📜 License

This project is licensed under the MIT License.
