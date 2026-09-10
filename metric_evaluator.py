import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

class MetricEvaluator:
    def __init__(self, iou_threshold=0.5):
        self.iou_thresh = iou_threshold
        self.reset()

    def reset(self):
        # 紀錄每幀的匹配結果
        self.frame_stats = [] 
        # 全局 ID 映射 (for IDF1)
        self.gt_ids = set()
        self.pred_ids = set()
        self.total_gt = 0
        self.total_pred = 0
        
        # For HOTA
        # matches[gt_id][pred_id] = count of frames they matched
        self.global_matches = defaultdict(lambda: defaultdict(int))
        self.gt_lifetimes = defaultdict(int)    # Total frames gt_id existed
        self.pred_lifetimes = defaultdict(int)  # Total frames pred_id existed

    @staticmethod
    def calculate_iou(box1, box2):
        # box: [x1, y1, x2, y2]
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
        return iou

    def update(self, gt_objects, tracker_objects, min_hits=3):
        """
        Args:
            gt_objects: list of dicts {'track_id', 'bbox', ...}
            tracker_objects: list of TrackedObject
            min_hits: minimum hits required to consider a track as "confirmed"
        """
        # Filter for confirmed tracks only
        confirmed_tracks = [obj for obj in tracker_objects if obj.hits >= min_hits]
        
        self.total_gt += len(gt_objects)
        self.total_pred += len(confirmed_tracks)  # Count only confirmed tracks
        
        # Update lifetimes
        for gt in gt_objects: self.gt_lifetimes[gt['track_id']] += 1
        for pr in confirmed_tracks: self.pred_lifetimes[pr.id] += 1
        
        # 1. 計算 IOU Matrix (only with confirmed tracks)
        n_gt = len(gt_objects)
        n_pred = len(confirmed_tracks)
        
        matches = [] # List of (gt_idx, pred_idx)
        
        if n_gt > 0 and n_pred > 0:
            iou_matrix = np.zeros((n_gt, n_pred))
            for i, gt in enumerate(gt_objects):
                for j, pred in enumerate(confirmed_tracks):
                    iou_matrix[i, j] = self.calculate_iou(gt['bbox'], pred.bbox)

            # Hungarian matching for evaluation
            cost_matrix = 1 - iou_matrix
            cost_matrix[iou_matrix < self.iou_thresh] = 1.0
            
            # Safeguard: replace any NaN or Inf values
            cost_matrix = np.nan_to_num(cost_matrix, nan=1.0, posinf=1.0, neginf=1.0)
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_thresh:
                    matches.append((r, c))
                    
                    gt_id = gt_objects[r]['track_id']
                    pred_id = confirmed_tracks[c].id
                    
                    self.global_matches[gt_id][pred_id] += 1
                    self.gt_ids.add(gt_id)
                    self.pred_ids.add(pred_id)

        # Stats for this frame
        tp = len(matches)
        fn = n_gt - tp
        fp = n_pred - tp
        
        # Track current matches for ID switches and fragmentation
        current_matches = {}
        for r, c in matches:
            gt_id = gt_objects[r]['track_id']
            pred_id = confirmed_tracks[c].id
            current_matches[gt_id] = pred_id
        
        self.frame_stats.append({
            'tp': tp, 'fn': fn, 'fp': fp,
            'matches': list(current_matches.items()),
            'gt_ids': [gt['track_id'] for gt in gt_objects],
        })

    def compute_metrics(self):
        if self.total_gt == 0: return {}

        # --- 1. Compute MOTA ---
        # MOTA = 1 - (FN + FP + IDSW) / Num_GT
        total_fn = sum(f['fn'] for f in self.frame_stats)
        total_fp = sum(f['fp'] for f in self.frame_stats)
        
        # Calculate ID Switches
        id_switches = 0
        last_matches = {} # gt_id -> pred_id
        
        for frame in self.frame_stats:
            current_matches = dict(frame['matches'])
            for gt_id, pred_id in current_matches.items():
                if gt_id in last_matches:
                    if last_matches[gt_id] != pred_id:
                        id_switches += 1
            last_matches = current_matches

        mota = 1.0 - (total_fn + total_fp + id_switches) / self.total_gt
        mota = max(0.0, mota) # Clip at 0

        # --- 2. Compute IDF1 ---
        # Build bipartite graph between all GT IDs and all Pred IDs
        # Weight = number of frames they matched
        all_gt_ids = list(self.gt_lifetimes.keys())
        all_pred_ids = list(self.pred_lifetimes.keys())
        
        if not all_gt_ids or not all_pred_ids:
            idf1 = 0.0
            id_tp = 0
            id_fn = self.total_gt
            id_fp = self.total_pred
        else:
            weight_matrix = np.zeros((len(all_gt_ids), len(all_pred_ids)))
            for i, g_id in enumerate(all_gt_ids):
                for j, p_id in enumerate(all_pred_ids):
                    weight_matrix[i, j] = self.global_matches[g_id][p_id]
            
            # Hungarian assignment to find best global ID mapping
            row_ind, col_ind = linear_sum_assignment(weight_matrix, maximize=True)
            
            id_tp = sum(weight_matrix[r, c] for r, c in zip(row_ind, col_ind))
            id_fn = self.total_gt - id_tp
            id_fp = self.total_pred - id_tp
            
            idf1 = 2 * id_tp / (2 * id_tp + id_fp + id_fn) if (2 * id_tp + id_fp + id_fn) > 0 else 0.0

        # --- 3. Compute HOTA (Simplified HOTA_alpha) ---
        # HOTA = sqrt(DetA * AssA)
        
        # DetA = TP / (TP + FN + FP)
        total_tp = sum(f['tp'] for f in self.frame_stats)
        det_a = total_tp / (total_tp + total_fn + total_fp + 1e-6)
        
        # AssA (Association Accuracy)
        ass_a_sum = 0
        total_matches_count = 0
        
        for gt_id, preds in self.global_matches.items():
            for pred_id, count in preds.items():
                if count == 0: continue
                
                # TPA(c): frames where gt_id and pred_id matched
                tpa = count
                
                # FNA(c): frames where gt_id existed but matched to different pred OR missed
                fna = self.gt_lifetimes[gt_id] - tpa
                
                # FPA(c): frames where pred_id existed but matched to different gt OR false positive
                fpa = self.pred_lifetimes[pred_id] - tpa
                
                ass_iou = tpa / (tpa + fna + fpa)
                
                # We sum AssIoU for every matched frame (weighted by match duration)
                ass_a_sum += ass_iou * tpa
                total_matches_count += tpa
        
        ass_a = ass_a_sum / (total_matches_count + 1e-6)
        hota = np.sqrt(det_a * ass_a)

        # --- 4. Compute Fragmentation (Frag) ---
        # Fragmentation: number of times a GT track is interrupted (matched, then not matched, then matched again)
        frag_count = 0
        for gt_id in self.gt_lifetimes.keys():
            was_matched = False
            was_interrupted = False
            
            for frame in self.frame_stats:
                frame_matches = dict(frame['matches'])
                is_matched = gt_id in frame_matches
                gt_exists = gt_id in frame.get('gt_ids', [])
                
                if not gt_exists:
                    continue
                    
                if is_matched:
                    if was_interrupted:
                        # Was interrupted, now matched again -> fragmentation!
                        frag_count += 1
                        was_interrupted = False
                    was_matched = True
                else:
                    # Not matched in this frame
                    if was_matched:
                        # Was matched before, now interrupted
                        was_interrupted = True

        return {
            "MOTA": mota,
            "IDF1": idf1,
            "HOTA": hota,
            "DetA": det_a,
            "AssA": ass_a,
            "IDSW": id_switches,
            "IDFP": int(id_fp),  # ID False Positives
            "IDFN": int(id_fn),  # ID False Negatives
            "Frag": frag_count,  # Fragmentation count
            "FP": total_fp,
            "FN": total_fn,
            "GT_Count": self.total_gt
        }
