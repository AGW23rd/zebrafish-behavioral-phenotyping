'''
Benchmark tracker performance on different motion types and number of objects.

Motion types:
- linear
- random_walk
- curve
- levy_flight
- brownian_bridge
- correlated_random_walk
- perlin_noise


'''
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, deque
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict
import cv2
import colorsys

try:
    from metric_evaluator import MetricEvaluator
except ImportError:
    print("MetricEvaluator not found.")
    exit()

try:
    from yolo_tracker_v2 import YOLOTracker, SORTTracker, ByteTracker, OCSORTTracker
except ImportError:
    print("YOLOTracker modules not found.")
    exit()

motion_types=['linear', 'random_walk', 'curve', 'levy_flight', 'brownian_bridge', 'correlated_random_walk', 'perlin_noise', 'spiral']

# ==========================================
# 1. 虛擬場景與運動模擬器 (Motion Simulator)
# ==========================================

class MotionSimulator:
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.objects = {}  # ground truth objects
        self.frame_id = 0

    def add_object(self, obj_id, start_pos, motion_type='linear', speed=10.0):
        """
        motion_type: 'linear', 'random_walk', 'curve', 'levy_flight', 
                     'brownian_bridge', 'correlated_random_walk', 'perlin_noise'
        """
        self.objects[obj_id] = {
            'bbox': np.array(start_pos),
            'velocity': np.random.uniform(-speed, speed, 2),
            'type': motion_type,
            'class_id': motion_types.index(motion_type),
            'embedding': np.random.randn(128),
            'active': True,
            'bounce_cooldown': 0,  # 防止邊界重複觸發
            'target_direction': np.random.uniform(0, 2*np.pi),  # 用於某些運動模式
            'step_counter': 0,  # 用於週期性運動
            'levy_step_remaining': 0,  # Levy flight 用
        }
        self.objects[obj_id]['embedding'] /= np.linalg.norm(self.objects[obj_id]['embedding'])


    def step(self, noise_std=0.0, drop_prob=0.0):
        """
        更新所有物體位置並生成帶有雜訊的偵測結果
        """
        self.frame_id += 1
        detections = []
        ground_truth = []

        for obj_id, obj in self.objects.items():
            if not obj['active']:
                continue

            w = obj['bbox'][2] - obj['bbox'][0]
            h = obj['bbox'][3] - obj['bbox'][1]
            cx = (obj['bbox'][0] + obj['bbox'][2]) / 2
            cy = (obj['bbox'][1] + obj['bbox'][3]) / 2

            # 減少 bounce cooldown
            if obj['bounce_cooldown'] > 0:
                obj['bounce_cooldown'] -= 1

            # 1. Update Position based on motion type
            if obj['type'] == 'linear':
                # 慣性運動加上小擾動
                obj['velocity'] += np.random.normal(0, 1.0, 2)

            elif obj['type'] == 'random_walk':
                # 改進：使用動量保持，避免速度突然衰減
                momentum = 0.7
                new_random = np.random.uniform(-8, 8, 2)
                obj['velocity'] = obj['velocity'] * momentum + new_random * (1 - momentum)

            elif obj['type'] == 'curve':
                # 平滑曲線運動
                angle = np.deg2rad(np.random.uniform(-5, 5))
                obj['velocity'] += np.random.normal(0, 1.0, 2)
                vx, vy = obj['velocity']
                obj['velocity'][0] = vx * np.cos(angle) - vy * np.sin(angle)
                obj['velocity'][1] = vx * np.sin(angle) + vy * np.cos(angle)

            elif obj['type'] == 'levy_flight':
                # Levy Flight: 長距離跳躍 + 短距離遊走
                if obj['levy_step_remaining'] <= 0:
                    # 決定新的移動模式
                    if np.random.random() < 0.1:  # 10% 機率長跳
                        obj['levy_step_remaining'] = np.random.randint(20, 50)
                        direction = np.random.uniform(0, 2*np.pi)
                        speed = np.random.uniform(10, 15)
                        obj['velocity'] = np.array([
                            speed * np.cos(direction),
                            speed * np.sin(direction)
                        ])
                    else:  # 短距離遊走
                        obj['levy_step_remaining'] = np.random.randint(5, 15)
                        obj['velocity'] += np.random.uniform(-3, 3, 2)
                else:
                    obj['levy_step_remaining'] -= 1
                    # 維持當前方向，加入小擾動
                    obj['velocity'] += np.random.normal(0, 0.5, 2)

            elif obj['type'] == 'brownian_bridge':
                # Brownian Bridge: 朝向目標點移動，但帶有隨機性
                obj['step_counter'] += 1
                
                # 每 100 steps 設定新目標
                if obj['step_counter'] % 100 == 0:
                    obj['target_pos'] = np.array([
                        np.random.uniform(100, self.width-100),
                        np.random.uniform(100, self.height-100)
                    ])
                
                if 'target_pos' in obj:
                    target_vec = obj['target_pos'] - np.array([cx, cy])
                    distance = np.linalg.norm(target_vec)
                    if distance > 10:
                        # 朝向目標 + 隨機擾動
                        direction = target_vec / distance
                        obj['velocity'] = direction * 5 + np.random.normal(0, 2, 2)
                    else:
                        # 到達目標，設定新目標
                        obj['target_pos'] = np.array([
                            np.random.uniform(100, self.width-100),
                            np.random.uniform(100, self.height-100)
                        ])

            elif obj['type'] == 'correlated_random_walk':
                # Correlated Random Walk: 傾向維持前進方向
                obj['step_counter'] += 1
                
                # 每 20 步調整目標方向
                if obj['step_counter'] % 20 == 0:
                    obj['target_direction'] += np.random.uniform(-np.pi/4, np.pi/4)
                
                # 朝目標方向移動，帶有隨機性
                current_speed = np.linalg.norm(obj['velocity'])
                if current_speed < 1:
                    current_speed = 5
                
                target_vx = current_speed * np.cos(obj['target_direction'])
                target_vy = current_speed * np.sin(obj['target_direction'])
                
                # 平滑過渡到目標方向
                obj['velocity'][0] = obj['velocity'][0] * 0.8 + target_vx * 0.2
                obj['velocity'][1] = obj['velocity'][1] * 0.8 + target_vy * 0.2
                
                # 加入小擾動
                obj['velocity'] += np.random.normal(0, 1, 2)

            elif obj['type'] == 'perlin_noise':
                # 使用簡化的 Perlin-like 運動（平滑隨機）
                obj['step_counter'] += 1
                
                # 使用 sin/cos 組合產生平滑變化
                t = obj['step_counter'] * 0.05
                noise_x = np.sin(t * 1.5) * 3 + np.cos(t * 0.7) * 2
                noise_y = np.cos(t * 1.3) * 3 + np.sin(t * 0.9) * 2
                
                obj['velocity'] = obj['velocity'] * 0.9 + np.array([noise_x, noise_y]) * 0.1
                obj['velocity'] += np.random.normal(0, 0.5, 2)

            elif obj['type'] == 'spiral':
                # 螺旋運動
                obj['step_counter'] += 1
                t = obj['step_counter'] * 0.1
                
                # 中心點
                center_x = self.width / 2
                center_y = self.height / 2
                
                # 當前位置到中心的向量
                to_center = np.array([center_x - cx, center_y - cy])
                distance = np.linalg.norm(to_center)
                
                if distance > 50:
                    # 切線方向（垂直於半徑）
                    tangent = np.array([-to_center[1], to_center[0]])
                    tangent = tangent / np.linalg.norm(tangent)
                    
                    # 結合向心力和切線運動
                    radial_component = to_center / distance * 0.5
                    obj['velocity'] = tangent * 8 + radial_component + np.random.normal(0, 0.5, 2)
                else:
                    # 離開中心
                    obj['velocity'] = -to_center / distance * 10

            # Limit max speed
            speed = np.linalg.norm(obj['velocity'])
            if speed > 15:
                obj['velocity'] = obj['velocity'] / speed * 15

            # Apply movement
            cx += obj['velocity'][0]
            cy += obj['velocity'][1]

            # 改進的邊界處理
            margin = 50
            bounced = False
            
            # X-axis boundary
            if cx < margin:
                cx = margin
                if obj['bounce_cooldown'] == 0:
                    obj['velocity'][0] = abs(obj['velocity'][0]) + np.random.uniform(5, 10)
                    obj['bounce_cooldown'] = 5
                    bounced = True
                    
            elif cx > self.width - margin:
                cx = self.width - margin
                if obj['bounce_cooldown'] == 0:
                    obj['velocity'][0] = -abs(obj['velocity'][0]) - np.random.uniform(5, 10)
                    obj['bounce_cooldown'] = 5
                    bounced = True
                
            # Y-axis boundary
            if cy < margin:
                cy = margin
                if obj['bounce_cooldown'] == 0:
                    obj['velocity'][1] = abs(obj['velocity'][1]) + np.random.uniform(5, 10)
                    obj['bounce_cooldown'] = 5
                    bounced = True
                    
            elif cy > self.height - margin:
                cy = self.height - margin
                if obj['bounce_cooldown'] == 0:
                    obj['velocity'][1] = -abs(obj['velocity'][1]) - np.random.uniform(5, 10)
                    obj['bounce_cooldown'] = 5
                    bounced = True

            # 如果碰撞且是 random_walk，給予額外推力
            if bounced and obj['type'] in ['random_walk', 'levy_flight']:
                # 給予遠離邊界的額外推力
                push_x = 0
                push_y = 0
                if cx < self.width / 2:
                    push_x = 8
                else:
                    push_x = -8
                    
                if cy < self.height / 2:
                    push_y = 8
                else:
                    push_y = -8
                    
                obj['velocity'] += np.array([push_x, push_y])

            # Update bbox
            new_bbox = np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
            obj['bbox'] = new_bbox

            # 2. Generate Ground Truth Record
            ground_truth.append({
                'frame_id': self.frame_id,
                'track_id': obj_id,
                'bbox': new_bbox.copy(),
                'class_id': obj['class_id'],
                'embedding': obj['embedding'],
                'motion_type': obj['type']
            })

            # 3. Generate Detection (with Noise and Dropouts)
            if np.random.random() > drop_prob:
                # Add position noise
                noise = np.random.normal(0, noise_std, 4)
                det_bbox = new_bbox + noise
                
                # Simulate embedding noise
                det_emb = obj['embedding'] + np.random.normal(0, 0.1, 128)
                det_emb /= np.linalg.norm(det_emb)

                detections.append({
                    'bbox': det_bbox.tolist(),
                    'confidence': np.random.uniform(0.7, 0.99),
                    'class_id': obj['class_id'],
                    'embedding': det_emb
                })

        return ground_truth, detections


def get_color(motion_type, obj_id, motion_types=motion_types):
    """
    根據 motion_type 決定 Hue
    根據 obj_id 決定 Saturation / Value
    """
    if motion_type not in motion_types:
        hue = 0.0
    else:
        hue = motion_types.index(motion_type) / len(motion_types)
    
    
    # Generate variations
    sat = 0.5 + 0.5 * ((obj_id * 37) % 10) / 10.0
    val = 0.6 + 0.4 * ((obj_id * 19) % 10) / 10.0
    
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(b*255), int(g*255), int(r*255)) # BGR for OpenCV

def draw_simulation_frame(img, ground_truth):
    for obj in ground_truth:
        bbox = obj['bbox'].astype(int)
        color = get_color(obj['motion_type'], obj['track_id'])
        
        # Calculate center and radius for circle
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        radius = min(w, h) // 2
        
        cv2.circle(img, (cx, cy), radius, color, -1)
        cv2.putText(img, f"ID:{obj['track_id']} {obj['motion_type']}", 
                    (bbox[0], bbox[1]-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
    return img

def draw_tracking_frame(img, tracked_objects, ground_truth=None):
    # Draw ground truth first (simulated objects) as circles
    if ground_truth:
        for obj in ground_truth:
            bbox = obj['bbox'].astype(int)
            color = get_color(obj['motion_type'], obj['track_id'])
            
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            radius = min(w, h) // 2
            
            cv2.circle(img, (cx, cy), radius, color, -1)

    for obj in tracked_objects:
        bbox = obj.bbox.astype(int)
        # Random color for tracking ID
        np.random.seed(obj.id)
        color = np.random.randint(0, 255, 3).tolist()
        
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(img, f"ID:{obj.id}", (bbox[0], bbox[1]-7), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
        
        # Draw trail
        if len(obj.trail) > 1:
            points = np.array(obj.trail, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [points], False, color, 1)
    return img


def run_single_test(tracker_name, tracker_config, n_obj, frames, motion_types=motion_types, save_video=True):
    """
    運行單一tracker的測試
    
    Args:
        tracker_name: tracker名稱 ('UKF', 'SORT', 'BYTE')
        tracker_config: tracker設定
        n_obj: 物件數量
        frames: 幀數
    
    Returns:
        測試結果字典
    """
    print(f"  [{tracker_name}] Testing...")
    
    # 1. Setup Simulator
    sim = MotionSimulator()
    # Mix motion types
    for i in range(n_obj):
        m_type = np.random.choice(motion_types)  #p=[0.4, 0.3, 0.3] if want
        
        # Fix: Ensure valid bbox (x2 > x1, y2 > y1)
        w = np.random.randint(30, 90)
        h = np.random.randint(30, 90)
        x1 = np.random.randint(50, sim.width - w - 50)
        y1 = np.random.randint(50, sim.height - h - 50)
        pos = [x1, y1, x1+w, y1+h]
        
        sim.add_object(i+1, pos, motion_type=m_type)

    # 2. Setup Tracker based on type
    if tracker_name == 'UKF':
        tracker = YOLOTracker(**tracker_config)
    elif tracker_name == 'SORT':
        tracker = SORTTracker(**tracker_config)
    elif tracker_name == 'BYTE':
        tracker = ByteTracker(**tracker_config)
    elif tracker_name == 'OCSORT':
        tracker = OCSORTTracker(**tracker_config)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_name}")
    
    # 3. Setup Evaluator
    evaluator = MetricEvaluator(iou_threshold=0.5)

    # 4. Setup Video Writers
    combined_writer = None
    dashboard_height = 200
    separator_width = 5
    
    if save_video:
        os.makedirs("runs/benchmark_videos", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        combined_out_path = f"runs/benchmark_videos/{tracker_name}_N{n_obj}_combined.mp4"
        
        # Width * 2 + separator, Height + dashboard
        out_w = sim.width * 2 + separator_width
        out_h = sim.height + dashboard_height
        
        combined_writer = cv2.VideoWriter(combined_out_path, fourcc, 20.0, (out_w, out_h))
        print(f"  [Video] Saving to {combined_out_path}")

    # 5. Run Loop
    start_time = time.time()
    
    for f in range(frames):
        # A. Generate Data (with noise and dropouts)
        gt_list, det_list = sim.step(noise_std=2.0, drop_prob=0.1)
        
        # B. Run Tracker
        tracked_objects = tracker.update(det_list, frame_id=f, dt=1.0)
        
        # C. Update Evaluator (with min_hits parameter)
        evaluator.update(gt_list, tracked_objects, min_hits=tracker.min_hits)
        
        # D. Write Video
        if save_video:
            # 1. Simulation Frame
            sim_img = np.zeros((sim.height, sim.width, 3), dtype=np.uint8)
            sim_img = draw_simulation_frame(sim_img, gt_list)
            cv2.putText(sim_img, f"Sim Frame: {f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 2. Tracking Frame
            track_img = np.zeros((sim.height, sim.width, 3), dtype=np.uint8)
            track_img = draw_tracking_frame(track_img, tracked_objects, gt_list)
            cv2.putText(track_img, f"Track Frame: {f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 3. Separator
            separator = np.full((sim.height, separator_width, 3), 255, dtype=np.uint8)
            
            # 4. Dashboard
            dashboard = np.zeros((dashboard_height, out_w, 3), dtype=np.uint8)
            
            # Calculate Metrics for Dashboard
            # Class Counts
            class_counts = defaultdict(int)
            for obj in gt_list:
                class_counts[obj['class_id']] += 1
            
            # Path Stats
            active_paths = len(tracked_objects)
            avg_path_len = np.mean([len(obj.trail) for obj in tracked_objects]) if tracked_objects else 0
            total_paths = max([obj.id for obj in tracked_objects]) if tracked_objects else 0
            
            # Cumulative Stats
            total_gt = evaluator.total_gt
            total_fn = sum(f['fn'] for f in evaluator.frame_stats)
            total_fp = sum(f['fp'] for f in evaluator.frame_stats)
            
            fn_rate = (total_fn / total_gt * 100) if total_gt > 0 else 0
            fp_rate = (total_fp / total_gt * 100) if total_gt > 0 else 0
            
            # Advanced Metrics (Frag, IDSW) - Compute every frame for display
            curr_metrics = evaluator.compute_metrics()
            frag = curr_metrics.get('Frag', 0)
            idsw = curr_metrics.get('IDSW', 0)
            
            # Draw Text on Dashboard
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            line_height = 50
            col1_x = 20
            col2_x = 500
            col3_x = 1000   #not used, save for class
            col4_x = 1500
            col5_x = 2000
            col6_x = 2500
            
            # Column 1: General Info & Classes
            cv2.putText(dashboard, f"Tracker: {tracker_name}", (col1_x, 40), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(dashboard, f"Objects: {n_obj}", (col1_x, 40 + line_height), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(dashboard, f"Frames: {frames}", (col1_x, 40 + line_height*2), font, font_scale, (255, 255, 255), thickness)
            
            # Class Breakdown (RGB)
            col_count = 0
            for i in range(len(motion_types)):
                if class_counts[i] > 0:
                    color = get_color(motion_types[i], i)
                    cv2.putText(dashboard, f"Class {i} ({motion_types[i]}): {class_counts[i]}", (col2_x + 500*(col_count%2), 40 + line_height*(col_count//2)*3//4), font, font_scale*0.75, color, thickness)
                    col_count += 1

            # Column 2: Tracking Performance
            cv2.putText(dashboard, f"Active Paths: {active_paths}", (col4_x, 40), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(dashboard, f"Avg Path Len: {avg_path_len:.1f}", (col4_x, 40 + line_height), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(dashboard, f"Total Paths: {total_paths}", (col4_x, 40 + line_height*2), font, font_scale, (255, 255, 255), thickness)
            
            cv2.putText(dashboard, f"Fragmentation: {frag}", (col5_x, 40), font, font_scale, (255, 255, 0), thickness)
            cv2.putText(dashboard, f"ID Switches: {idsw}", (col5_x, 40 + line_height), font, font_scale, (0, 255, 255), thickness)

            # Column 3: Errors
            cv2.putText(dashboard, f"False Neg (FN): {fn_rate:.1f}%", (col6_x, 40), font, font_scale, (100, 100, 255), thickness)
            cv2.putText(dashboard, f"False Pos (FP): {fp_rate:.1f}%", (col6_x, 40 + line_height), font, font_scale, (100, 100, 255), thickness)
            
            # Combine All
            top_row = np.hstack((sim_img, separator, track_img))
            final_frame = np.vstack((top_row, dashboard))
            
            combined_writer.write(final_frame)

    if save_video:
        combined_writer.release()

    end_time = time.time()
    fps = frames / (end_time - start_time)
    
    # 6. Calculate Metrics
    metrics = evaluator.compute_metrics()
    
    print(f"  [{tracker_name}] Performance: {fps:.1f} FPS")
    print(f"  [{tracker_name}] HOTA: {metrics['HOTA']:.3f} | IDF1: {metrics['IDF1']:.3f} | MOTA: {metrics['MOTA']:.3f}")
    print(f"  [{tracker_name}] IDSW: {metrics['IDSW']} | Frag: {metrics['Frag']}")
    
    return {
        "Tracker": tracker_name,
        "N_Objects": n_obj,
        "Frames": frames,
        "HOTA": metrics['HOTA'],
        "IDF1": metrics['IDF1'],
        "MOTA": metrics['MOTA'],
        "DetA": metrics['DetA'],
        "AssA": metrics['AssA'],
        "IDSW": metrics['IDSW'],
        "IDFP": metrics['IDFP'],
        "IDFN": metrics['IDFN'],
        "Frag": metrics['Frag'],
        "FP": metrics['FP'],
        "FN": metrics['FN'],
        "FPS": fps
    }


def run_benchmark(num_objects_list=[5, 20, 50], frames=200, random_seed=42):
    """
    運行benchmark，比較UKF、SORT、BYTE三種tracker
    """
    print(f"{'='*70}")
    print(f"Multi-Tracker Benchmark: UKF vs SORT vs ByteTrack vs OC-SORT")
    print(f"{'='*70}")
    
    # 定義三種tracker的配置
    tracker_configs = {
        'UKF': {
            'max_age': 30,
            'max_trail_len': frames,
            'min_hits': 3,
            'iou_threshold': 0.3,
            'gating_threshold': 300,
            'appearance_weight': 0.2,
        },
        'SORT': {
            'max_age': 30,
            'max_trail_len': frames,
            'min_hits': 3,
            'iou_threshold': 0.3,
        },
        'BYTE': {
            'max_age': 30,
            'max_trail_len': frames,
            'min_hits': 3,
            'iou_threshold': 0.3,
            'high_thresh': 0.6,
            'low_thresh': 0.1,
        },
        'OCSORT': {
            'max_age': 30,
            'max_trail_len': frames,
            'min_hits': 3,
            'iou_threshold': 0.3,
            'inertia': 0.2,
        }
    }
    
    results_summary = []

    for n_obj in num_objects_list:
        print(f"\n{'='*70}")
        print(f"[Scenario] {n_obj} Objects, {frames} Frames")
        print(f"{'='*70}")
        
        # 使用相同的random seed確保公平比較
        np.random.seed(random_seed)
        
        # 測試每一種tracker
        for tracker_name in ['UKF', 'SORT', 'BYTE', 'OCSORT']:
            # Reset seed for each tracker to ensure same scenario
            np.random.seed(random_seed)
            
            result = run_single_test(
                tracker_name=tracker_name,
                tracker_config=tracker_configs[tracker_name],
                n_obj=n_obj,
                frames=frames
            )
            results_summary.append(result)

    print(f"\n{'='*70}")
    print("Final Benchmark Summary")
    print(f"{'='*70}")
    df = pd.DataFrame(results_summary)
    
    # 重新排列欄位順序，將Tracker放在前面
    cols = ['Tracker', 'N_Objects', 'Frames', 'HOTA', 'IDF1', 'MOTA', 
            'DetA', 'AssA', 'IDSW', 'IDFP', 'IDFN', 'Frag', 'FP', 'FN', 'FPS']
    df = df[cols]
    
    print(df.to_string(index=False))
    
    # Export to Excel
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    excel_filename = f"benchmark_results_{timestamp}.xlsx"
    
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Main results sheet
        df.to_excel(writer, sheet_name='Benchmark Results', index=False)
        
        # Per-tracker summary (pivot table style)
        summary_by_tracker = df.groupby('Tracker')[['HOTA', 'IDF1', 'MOTA', 'IDSW', 'Frag', 'FPS']].mean()
        summary_by_tracker.to_excel(writer, sheet_name='Tracker Summary')
        
        # Metrics description sheet
        metrics_info = pd.DataFrame([
            {"Metric": "HOTA", "Description": "Higher Order Tracking Accuracy (combines DetA and AssA)"},
            {"Metric": "IDF1", "Description": "ID F1 Score (identification precision and recall)"},
            {"Metric": "MOTA", "Description": "Multiple Object Tracking Accuracy"},
            {"Metric": "DetA", "Description": "Detection Accuracy"},
            {"Metric": "AssA", "Description": "Association Accuracy"},
            {"Metric": "IDSW", "Description": "ID Switches (number of times ID changed)"},
            {"Metric": "IDFP", "Description": "ID False Positives"},
            {"Metric": "IDFN", "Description": "ID False Negatives"},
            {"Metric": "Frag", "Description": "Fragmentation count (track interruptions)"},
            {"Metric": "FP", "Description": "False Positives (frame-level)"},
            {"Metric": "FN", "Description": "False Negatives (frame-level)"},
            {"Metric": "FPS", "Description": "Processing speed (frames per second)"}
        ])
        metrics_info.to_excel(writer, sheet_name='Metrics Description', index=False)
        
        # Tracker info sheet
        tracker_info = pd.DataFrame([
            {"Tracker": "UKF", "Description": "Unscented Kalman Filter - Uses non-linear state estimation with appearance features"},
            {"Tracker": "SORT", "Description": "Simple Online Realtime Tracking - Linear Kalman filter with IOU matching"},
            {"Tracker": "BYTE", "Description": "ByteTrack - Multi-stage matching with high/low confidence detections"},
            {"Tracker": "OCSORT", "Description": "Observation-Centric SORT - Incorporates Object-Centric Momentum for robust tracking in non-linear motion"}
        ])
        tracker_info.to_excel(writer, sheet_name='Tracker Info', index=False)
    
    print(f"\n✅ Results exported to: {excel_filename}")
    
    # Print comparison summary
    print(f"\n{'='*70}")
    print("Performance Comparison (Average across all scenarios)")
    print(f"{'='*70}")
    print(summary_by_tracker.to_string())
    
    return excel_filename

if __name__ == "__main__":
    run_benchmark(num_objects_list=[15, 30, 60, 120], frames=400, random_seed=123)