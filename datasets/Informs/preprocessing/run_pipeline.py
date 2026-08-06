"""
新数据集 (V2) 完整数据处理流水线
一键执行所有数据处理步骤
"""
import os
import sys
import subprocess
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "data_preprocess_v2")


def run_step(script_name, args=None):
    """运行单个处理步骤"""
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    cmd = ['python', script_path]
    if args:
        cmd.extend(args)
    
    print(f"\n{'='*60}")
    print(f"执行: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"[错误] {script_name} 执行失败!")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='V2 数据处理流水线')
    parser.add_argument('--skip_preprocess', action='store_true', help='跳过数据预处理')
    parser.add_argument('--skip_visual', action='store_true', help='跳过视觉特征提取')
    parser.add_argument('--skip_textual', action='store_true', help='跳过文本特征提取')
    parser.add_argument('--skip_user', action='store_true', help='跳过用户特征提取')
    parser.add_argument('--skip_final', action='store_true', help='跳过最终数据准备')
    parser.add_argument('--visual_device', type=str, default='cuda', help='视觉特征提取设备')
    parser.add_argument('--textual_device', type=str, default='cuda', help='文本特征提取设备')
    parser.add_argument('--local_only', action='store_true', help='仅使用本地模型缓存')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("新数据集 (V2) 完整数据处理流水线")
    print("=" * 60)
    
    if not args.skip_preprocess:
        print("\n[Step 1] 数据预处理...")
        if not run_step('preprocess_data.py'):
            return
    
    if not args.skip_visual:
        print("\n[Step 2] 视觉特征提取...")
        visual_args = ['--mode', 'all', '--device', args.visual_device]
        if args.local_only:
            visual_args.append('--local_only')
        if not run_step('visual_engineering.py', visual_args):
            return
    
    if not args.skip_textual:
        print("\n[Step 3] 文本特征提取...")
        textual_args = ['--mode', 'all', '--device', args.textual_device]
        if not run_step('textual_engineering.py', textual_args):
            return
    
    if not args.skip_user:
        print("\n[Step 4] 用户特征提取...")
        if not run_step('user_feature_engineering.py', ['--mode', 'all']):
            return
    
    if not args.skip_final:
        print("\n[Step 5] 最终数据准备...")
        if not run_step('prepare_final_data.py'):
            return
    
    print("\n" + "=" * 60)
    print("数据处理流水线完成!")
    print("=" * 60)
    print(f"\n输出目录: {os.path.join(PROJECT_ROOT, 'data_v2')}")
    print(f"最终数据: {os.path.join(PROJECT_ROOT, 'data_v2', 'processed_features')}")


if __name__ == '__main__':
    main()
