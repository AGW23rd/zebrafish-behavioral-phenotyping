

import pandas as pd
import numpy as np
import os
import glob

def score_larvae():
    INPUT_PATTERN = 'runs/locomotion_raw_data*.xlsx'
    
    input_files = glob.glob(INPUT_PATTERN)
    
    if not input_files:
        print(f"Error: No files matching {INPUT_PATTERN} found.")
        return

    for INPUT_FILE in input_files:
        # Extract base name to create a corresponding output file
        # Example: locomotion_raw_data_February(WT50).xlsx -> larvae_scoring_results_February(WT50).xlsx
        base_name = os.path.basename(INPUT_FILE)
        if base_name.startswith('locomotion_raw_data_'):
            suffix = base_name.replace('locomotion_raw_data_', '')
            OUTPUT_FILE = f'runs/larvae_scoring_results_{suffix}'
        else:
            OUTPUT_FILE = 'runs/larvae_scoring_results.xlsx'

        print(f"\nLoading {INPUT_FILE}...")
        try:
            df = pd.read_excel(INPUT_FILE)
        except Exception as e:
            print(f"Failed to load {INPUT_FILE}: {e}")
            continue
        
        # --- 1. Select Top 30 Tracks ---
        # Metric: Duration (frame count)
        obj_counts = df['object_id'].value_counts()
        
        # Filter for minimum viable duration (e.g., > 50 frames)
        valid_objs = obj_counts[obj_counts > 50]
        
        if len(valid_objs) < 30:
            print(f"Warning: Only found {len(valid_objs)} valid tracks (>50 frames) in {base_name}. Scoring all of them.")
            top_30_ids = valid_objs.index.tolist()
        else:
            top_30_ids = valid_objs.head(30).index.tolist()
            print(f"Selected Top {len(top_30_ids)} larvae by duration for {base_name}.")
            
        # Ensure centroids are calculated
        if 'cx' not in df.columns:
            df['cx'] = (df['x1'] + df['x2']) / 2
        if 'cy' not in df.columns:
            df['cy'] = (df['y1'] + df['y2']) / 2
            
        # Dynamic Coordinate & Well Grid Scaling
        # Fallback global coordinates
        x_max_global = df['x2'].max()
        y_max_global = df['y2'].max()
        x_min_global = df['x1'].min()
        y_min_global = df['y1'].min()
        
        WIDTH = max(2048.0, x_max_global) if pd.notna(x_max_global) else 2048.0
        HEIGHT = max(1536.0, y_max_global) if pd.notna(y_max_global) else 1536.0
        
        arena_width = WIDTH - min(0.0, x_min_global)
        arena_height = HEIGHT - min(0.0, y_min_global)
        
        # Calculate maximum span of any single track to classify the format (micro-wells vs. large open-well)
        max_track_span_x = 0.0
        max_track_span_y = 0.0
        for obj_id, track_g in df.groupby('object_id'):
            if len(track_g) > 50:
                span_x = track_g['cx'].max() - track_g['cx'].min()
                span_y = track_g['cy'].max() - track_g['cy'].min()
                if span_x > max_track_span_x:
                    max_track_span_x = span_x
                if span_y > max_track_span_y:
                    max_track_span_y = span_y
                    
        # If any single track spans > 40% of global width/height, it's a large open-well tracking format (1x1 grid).
        # Otherwise, default to micro-well 2x3 grid as defined in analyze_wells.py.
        if max_track_span_x > 0.4 * arena_width or max_track_span_y > 0.4 * arena_height:
            cols = 1
            rows = 1
        else:
            cols = 3
            rows = 2
            
        w_step = WIDTH / cols
        h_step = HEIGHT / rows
        
        # Assign well ID to each coordinate dynamically
        c_idx = (df['cx'] // w_step).astype(int).clip(0, cols - 1)
        r_idx = (df['cy'] // h_step).astype(int).clip(0, rows - 1)
        df['well_id'] = r_idx * cols + c_idx + 1
    
        # --- 2. Calculate Scoring Metrics ---
        results = []
        
        for obj_id in top_30_ids:
            track = df[df['object_id'] == obj_id].copy().sort_values('frame_id')
            
            # Calculate derived columns if missing (safety)
            if 'speed' not in track.columns:
                dx = track['x1'].diff().fillna(0)
                dy = track['y1'].diff().fillna(0)
                track['speed'] = np.sqrt(dx**2 + dy**2)
                
            if 'acceleration' not in track.columns:
                track['acceleration'] = track['speed'].diff().fillna(0)
                
            # 1. Kinematic Vigor (S.D. Velocity)
            sd_velocity = track['speed'].std()
            if pd.isna(sd_velocity):
                sd_velocity = 0.0
                
            # 2. Kinematic Force (S.D. Acceleration)
            sd_acceleration = track['acceleration'].std()
            if pd.isna(sd_acceleration):
                sd_acceleration = 0.0
                
            # 3. Spatial Intelligence (Thigmotaxis Out-Zone Ratio)
            # Find dominant well of the track
            well_id = track['well_id'].mode().iloc[0] if not track['well_id'].empty else 1
            w_row = (well_id - 1) // cols
            w_col = (well_id - 1) % cols
            
            x_min = w_col * w_step
            x_max = (w_col + 1) * w_step
            y_min = w_row * h_step
            y_max = (w_row + 1) * h_step
            
            w_width = x_max - x_min
            w_height = y_max - y_min
            
            # Outer 50% boundary area corresponds to a concentric scaling factor of sqrt(0.5)
            s = np.sqrt(0.5)
            margin_x = (1 - s) / 2 * w_width
            margin_y = (1 - s) / 2 * w_height
            
            cx_vals = track['cx']
            cy_vals = track['cy']
            
            in_outer_zone = (
                (cx_vals < x_min + margin_x) | 
                (cx_vals > x_max - margin_x) | 
                (cy_vals < y_min + margin_y) | 
                (cy_vals > y_max - margin_y)
            )
            thigmotaxis_index = in_outer_zone.mean() if len(track) > 0 else 0.0
            
            # 4. Peak Physical Capacity (V_max)
            v_max = track['speed'].max()
            if pd.isna(v_max):
                v_max = 0.0
                
            # 5. Peak Acceleration (A_max)
            a_max = track['acceleration'].abs().max()
            if pd.isna(a_max):
                a_max = 0.0
                
            # 6. Bout Frequency (Bout_Freq_per_min)
            # A "bout" is a contiguous run of frames where speed exceeds the median speed.
            # Frequency is normalized to bouts per minute assuming 30 fps.
            median_speed = track['speed'].median()
            if pd.notna(median_speed) and median_speed > 0:
                above_threshold = (track['speed'] > median_speed).astype(int)
                bout_starts = (above_threshold.diff() == 1).sum()
            else:
                bout_starts = 0
            duration_minutes = len(track) / (30.0 * 60.0)  # assuming 30 fps
            bout_freq = bout_starts / duration_minutes if duration_minutes > 0 else 0.0
            
            # 7. Asymmetry Index
            # Measures turning bias via mean signed angular change of heading.
            # |mean(dθ)| / mean(|dθ|): 0 = perfectly symmetric, 1 = always turning one direction.
            dx = track['cx'].diff().fillna(0)
            dy = track['cy'].diff().fillna(0)
            heading = np.arctan2(dy, dx)
            dtheta = heading.diff().fillna(0)
            # Wrap angle differences to [-pi, pi]
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
            mean_abs_dtheta = dtheta.abs().mean()
            if mean_abs_dtheta > 0:
                asymmetry_index = abs(dtheta.mean()) / mean_abs_dtheta
            else:
                asymmetry_index = 0.0
                
            results.append({
                'Larva_ID': obj_id,
                'Well_ID': well_id,
                'Velocity_SD_Raw': sd_velocity,
                'Acceleration_SD_Raw': sd_acceleration,
                'Thigmotaxis_Index': thigmotaxis_index,
                'V_max_Raw': v_max,
                'A_max_Raw': a_max,
                'Bout_Freq_per_min': bout_freq,
                'Asymmetry_Index': asymmetry_index
            })
            
        metrics_df = pd.DataFrame(results)
        
        # Guard against empty metrics_df (if no valid tracks at all)
        if metrics_df.empty:
            print(f"No valid tracks scored for {base_name}. Skipping save.")
            continue
            
        # --- 3. Normalization & Scoring ---
        # Rank Normalize (0.0 to 1.0)
        metrics_df['Velocity_SD_Rank'] = metrics_df['Velocity_SD_Raw'].rank(pct=True)
        metrics_df['Acceleration_SD_Rank'] = metrics_df['Acceleration_SD_Raw'].rank(pct=True)
        metrics_df['V_max_Rank'] = metrics_df['V_max_Raw'].rank(pct=True)
        metrics_df['A_max_Rank'] = metrics_df['A_max_Raw'].rank(pct=True)
        metrics_df['Bout_Freq_Rank'] = metrics_df['Bout_Freq_per_min'].rank(pct=True)
            
        # Composite Score Calculation (7 metrics):
        # Higher is better for all components.
        # Thigmotaxis & Asymmetry are inverted (1 - value) since lower raw values indicate healthier larvae.
        metrics_df['Score_Composite'] = (
            0.20 * metrics_df['Velocity_SD_Rank'] + 
            0.15 * metrics_df['Acceleration_SD_Rank'] + 
            0.15 * (1 - metrics_df['Thigmotaxis_Index']) +
            0.15 * metrics_df['V_max_Rank'] +
            0.10 * metrics_df['A_max_Rank'] +
            0.15 * metrics_df['Bout_Freq_Rank'] +
            0.10 * (1 - metrics_df['Asymmetry_Index'])
        )
        
        # Scale to 1-10 (Legacy Quality Score)
        metrics_df['Quality_Score'] = (1 + 9 * metrics_df['Score_Composite']).round().astype(int)
        
        # --- Precision Grading System (A, B, C, D) ---
        # Grade A: High-viability (Composite >= 0.65)
        # Grade B: Acceptable viability (Composite >= 0.35)
        # Grade C: Sub-optimal viability (Composite >= 0.20)
        # Grade D: Low quality embryo (Composite < 0.20)
        def assign_grade(score):
            if score >= 0.65:
                return 'A'
            elif score >= 0.35:
                return 'B'
            elif score >= 0.20:
                return 'C'
            else:
                return 'D'
                
        metrics_df['Precision_Grade'] = metrics_df['Score_Composite'].apply(assign_grade)
        
        # Sort by Score
        final_df = metrics_df.sort_values('Score_Composite', ascending=False)
        
        print(f"Top Larvae for {base_name} (Grade A):")
        print(final_df[final_df['Precision_Grade'] == 'A'][['Larva_ID', 'Precision_Grade', 'Score_Composite', 'Velocity_SD_Raw']].head())
        
        # Export
        final_df.to_excel(OUTPUT_FILE, index=False)
        print(f"Scoring complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    score_larvae()
