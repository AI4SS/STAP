"""
新数据集 (V2) 数据预处理 - 归一化热度版本
对四个指标分别做min-max归一化，然后等权平均得到最终热度标签
"""
import os
import sys
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from datetime import datetime
import holidays
import re
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

BASE_DIR = "/root/dataset"
TRAIN_EXCEL = os.path.join(BASE_DIR, "training.xlsx")
TEST_EXCEL = os.path.join(BASE_DIR, "testing.xlsx")
TEST_LABELS_EXCEL = os.path.join(BASE_DIR, "testing_set_with_labels.xlsx")

TRAIN_VIDEO_DIR = os.path.join(BASE_DIR, "train_converted_mp4")
TEST_VIDEO_DIR = os.path.join(BASE_DIR, "test_converted_mp4")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data_v2", "normalized_heat")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "raw_frames")

TARGET_COLUMNS = ['video_comment_count', 'video_heart_count', 'video_play_count', 'video_share_count']


def extract_tags(description):
    description = str(description)
    hashtags = re.findall(r'#(\w+)', description)
    mentions = re.findall(r'@(\w+)', description)
    return hashtags, mentions


def total_tag_frequency(tags_list, frequency_dict):
    return sum(frequency_dict[tag] for tag in tags_list if tag in frequency_dict)


def extract_date_components_and_check_holidays(row):
    try:
        date = datetime.utcfromtimestamp(int(row['video_create_date']))
    except:
        date = datetime.now()

    is_holiday = date in holidays.UnitedStates()

    if 9 <= date.hour < 18:
        time_period = 'Work Time'
    elif 18 <= date.hour < 23:
        time_period = 'Leisure Time'
    else:
        time_period = 'Sleep Time'

    return pd.Series({
        'year': date.year,
        'month': date.month,
        'day': date.day,
        'hour': date.hour,
        'is_holiday': is_holiday,
        'time_period': time_period
    })


def get_video_details(video_path):
    try:
        if not os.path.exists(video_path):
            return None, None, None, None, None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, None, None, None

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if frame_rate > 0:
            duration = total_frames / frame_rate
        else:
            duration = 0
        
        cap.release()

        return duration, total_frames, frame_rate, width, height
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return None, None, None, None, None


def extract_frames_from_video(video_path, output_dir, num_frames=10, target_size=(224, 224)):
    if not os.path.exists(video_path):
        return []

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <