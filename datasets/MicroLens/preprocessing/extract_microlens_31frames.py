"""
MicroLens 31帧特征提取脚本 - 支持断点续传
使用CLIP ViT-L/14模型 (768维)
每个视频提取1封面 + 30帧 = 31帧
"""
import os
import sys
import argparse
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from multiprocessing import Process

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def get_video_ids(covers_dir):
    covers_files = sorted([f for f in os.listdir(covers_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    video_ids = []
    for f in covers_files:
        try:
            vid = int(f.split('.')[0])
            video_ids.append(vid)
        except:
            pass
    return sorted(video_ids)


def load_checkpoint(output_dir, process_id):
    checkpoint_file = os.path.join(output_dir, f'checkpoint_process_{process_id}.json')
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {'processed_ids': [], 'last_index': 0}


def save_checkpoint(output_dir, process_id, processed_ids, last_index):
    checkpoint_file = os.path.join(output_dir, f'checkpoint_process_{process_id}.json')
    with open(checkpoint_file, 'w') as f:
        json.dump({'processed_ids': processed_ids, 'last_index': last_index}, f)


def save_features_batch(output_dir, process_id, video_ids_batch, features_batch):
    if len(video_ids_batch) == 0:
        return
    
    features_file = os.path.join(output_dir, f'features_process_{process_id}.npz')
    
    features_array = np.array(features_batch)
    
    if os.path.exists(features_file):
        existing = np.load(features_file)
        existing_features = existing['features']
        existing_ids = existing['video_ids'].tolist()
        
        features_array = np.vstack([existing_features, features_array])
        video_ids_batch = existing_ids + video_ids_batch
    
    np.savez(features_file, features=features_array, video_ids=np.array(video_ids_batch))


def load_clip_model(device):
    print(f"加载CLIP ViT-L/14模型到 {device}...")
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    model = model.to(device)
    model.eval()
    feature_dim = 768
    print("CLIP ViT-L/14模型加载完成")
    return model, preprocess, feature_dim


def extract_features_for_process(
    process_id: int,
    video_ids: list,
    covers_dir: str,
    frames_dir: str,
    output_dir: str,
    device: str,
    num_frames: int = 31,
    save_interval: int = 50,
):
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(device)
    model, transform, feature_dim = load_clip_model(device)
    
    checkpoint = load_checkpoint(output_dir, process_id)
    processed_ids = set(checkpoint['processed_ids'])
    start_index = checkpoint['last_index']
    
    print(f"进程 {process_id}: 从索引 {start_index} 开始，已处理 {len(processed_ids)} 个视频")
    
    batch_video_ids = []
    batch_features = []
    
    for idx in tqdm(range(start_index, len(video_ids)), desc=f"进程{process_id}", position=process_id):
        vid = video_ids[idx]
        
        if vid in processed_ids:
            continue
        
        frame_features = []
        
        # 封面图
        cover_path = os.path.join(covers_dir, f"{vid}.jpg")
        if not os.path.exists(cover_path):
            cover_path = os.path.join(covers_dir, f"{vid}.png")
        
        if os.path.exists(cover_path):
            try:
                cover_img = Image.open(cover_path).convert('RGB')
                img_tensor = transform(cover_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = model.encode_image(img_tensor).squeeze().cpu().numpy()
                frame_features.append(feat)
            except:
                continue
        
        # 30帧 (从子目录中读取)
        for i in range(num_frames - 1):
            # 尝试多种路径格式
            frame_path = os.path.join(frames_dir, str(vid), f"frame_{i:03d}.jpg")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(frames_dir, str(vid), f"frame_{i:03d}.png")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(frames_dir, f"{vid}_{i}.jpg")
            if not os.path.exists(frame_path):
                frame_path = os.path.join(frames_dir, f"{vid}-{i}.jpg")
            
            if os.path.exists(frame_path):
                try:
                    frame_img = Image.open(frame_path).convert('RGB')
                    img_tensor = transform(frame_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        feat = model.encode_image(img_tensor).squeeze().cpu().numpy()
                    frame_features.append(feat)
                except:
                    pass
        
        # 填充到num_frames
        while len(frame_features) < num_frames:
            if len(frame_features) > 0:
                frame_features.append(frame_features[-1].copy())
            else:
                break
        
        if len(frame_features) == num_frames:
            batch_video_ids.append(vid)
            batch_features.append(np.array(frame_features))
            processed_ids.add(vid)
        
        if len(batch_video_ids) >= save_interval:
            save_features_batch(output_dir, process_id, batch_video_ids, batch_features)
            save_checkpoint(output_dir, process_id, list(processed_ids), idx + 1)
            batch_video_ids = []
            batch_features = []
    
    if len(batch_video_ids) > 0:
        save_features_batch(output_dir, process_id, batch_video_ids, batch_features)
        save_checkpoint(output_dir, process_id, list(processed_ids), len(video_ids))
    
    print(f"进程 {process_id}: 完成，共处理 {len(processed_ids)} 个视频")


def merge_features(output_dir, num_processes):
    print("\n合并各进程特征...")
    
    all_video_ids = []
    all_features = []
    
    for pid in range(num_processes):
        features_file = os.path.join(output_dir, f'features_process_{pid}.npz')
        if os.path.exists(features_file):
            data = np.load(features_file)
            features = data['features']
            video_ids = data['video_ids'].tolist()
            
            all_features.append(features)
            all_video_ids.extend(video_ids)
            print(f"进程 {pid}: {len(video_ids)} 个视频, 特征形状 {features.shape}")
    
    if len(all_features) > 0:
        all_features = np.vstack(all_features)
        
        visual_features_path = os.path.join(output_dir, 'visual_features_clip.npy')
        np.save(visual_features_path, all_features)
        
        video_ids_path = os.path.join(output_dir, 'video_ids.json')
        with open(video_ids_path, 'w') as f:
            json.dump(all_video_ids, f)
        
        print(f"\n合并完成:")
        print(f"  视觉特征: {all_features.shape}")
        print(f"  视频数量: {len(all_video_ids)}")
        print(f"  保存路径: {visual_features_path}")
        
        return all_features, all_video_ids
    
    return None, None


def main():
    parser = argparse.ArgumentParser(description='MicroLens 31帧特征提取 - 支持断点续传')
    parser.add_argument('--covers_dir', type=str,
                        default='/root/gsar_jamba_topo/data_MicroLens/MicroLens-100k_covers')
    parser.add_argument('--frames_dir', type=str,
                        default='/root/gsar_jamba_topo/data_MicroLens/MicroLens-100k_frames_30')
    parser.add_argument('--output_dir', type=str,
                        default='/root/gsar_jamba_topo/data_MicroLens/extracted_features_31frames')
    parser.add_argument('--num_processes', type=int, default=4)
    parser.add_argument('--num_frames', type=int, default=31)
    parser.add_argument('--save_interval', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda:0')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_video_ids = get_video_ids(args.covers_dir)
    print(f"总视频数: {len(all_video_ids)}")
    
    chunk_size = len(all_video_ids) // args.num_processes
    chunks = []
    for i in range(args.num_processes):
        start = i * chunk_size
        if i == args.num_processes - 1:
            chunks.append(all_video_ids[start:])
        else:
            chunks.append(all_video_ids[start:start + chunk_size])
    
    processes = []
    for pid in range(args.num_processes):
        p = Process(
            target=extract_features_for_process,
            args=(
                pid,
                chunks[pid],
                args.covers_dir,
                args.frames_dir,
                args.output_dir,
                args.device,
                args.num_frames,
                args.save_interval,
            )
        )
        processes.append(p)
        p.start()
        print(f"启动进程 {pid}")
    
    for p in processes:
        p.join()
    
    visual_features, visual_video_ids = merge_features(args.output_dir, args.num_processes)
    
    if visual_features is not None:
        meta = {
            'video_ids': visual_video_ids,
            'visual_features_shape': list(visual_features.shape),
            'num_frames': args.num_frames,
            'feature_dim': visual_features.shape[-1],
            'model': 'CLIP ViT-L/14',
        }
        with open(os.path.join(args.output_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        
        print("\n" + "=" * 60)
        print("特征提取完成!")
        print("=" * 60)


if __name__ == '__main__':
    main()
