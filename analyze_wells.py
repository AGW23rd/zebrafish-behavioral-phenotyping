
import pandas as pd
import numpy as np
import os
import glob

def safe_series(df, col_name):
    if col_name not in df.columns:
        return pd.Series(0.0, index=df.index)
    col = df[col_name]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col

def analyze_wells():
    INPUT_PATTERN = 'runs/locomotion_raw_data*.xlsx'
    
    input_files = glob.glob(INPUT_PATTERN)
    
    if not input_files:
        print(f"Error: No files matching {INPUT_PATTERN} found. Please run the tracker first.")
        return

    for INPUT_FILE in input_files:
        # Extract base name to create a corresponding output file
        # Example: locomotion_raw_data_February(WT50).xlsx -> locomotion_summary_February(WT50).xlsx
        base_name = os.path.basename(INPUT_FILE)
        if base_name.startswith('locomotion_raw_data_'):
            suffix = base_name.replace('locomotion_raw_data_', '')
            OUTPUT_FILE = f'runs/locomotion_summary_{suffix}'
        else:
            OUTPUT_FILE = 'runs/locomotion_summary.xlsx'
            
        print(f"\nProcessing {INPUT_FILE}...")
        try:
            df = pd.read_excel(INPUT_FILE)
        except Exception as e:
            print(f"Failed to load {INPUT_FILE}: {e}")
            continue
            
        # --- 1. Define Wells ---
        # Assumptions: 2048x1536 resolution, 2x3 grid
        # Well 1: Top-Left, Well 2: Top-Center, Well 3: Top-Right
        # Well 4: Bot-Left, Well 5: Bot-Center, Well 6: Bot-Right
        
        WIDTH = 2048
        HEIGHT = 1536
        
        # Calculate Centroids if not present
        if 'cx' not in df.columns:
            df['cx'] = (df['x1'] + df['x2']) / 2
        if 'cy' not in df.columns:
            df['cy'] = (df['y1'] + df['y2']) / 2
            
        cols = 3
        rows = 2
        w_step = WIDTH / cols
        h_step = HEIGHT / rows
        
        c = (df['cx'] // w_step).astype(int)
        r = (df['cy'] // h_step).astype(int)
        
        # Clamp bounds
        c = np.clip(c, 0, cols - 1)
        r = np.clip(r, 0, rows - 1)
        
        # Assign well for each frame, or per object based on first frame? 
        # Let's assign per frame and take mode for the object to be robust.
        df['well_id'] = r * cols + c + 1
        
        # Determine dominant well for each object
        # Mode can be empty if the series is empty or all NaN, so guard it
        def get_mode(x):
            m = x.mode()
            return m.iloc[0] if not m.empty else np.nan
        obj_wells = df.groupby('object_id')['well_id'].agg(get_mode)
        
        # --- 2. Select Best Track per Well ---
        # Metric: Duration (count of frames)
        well_best_tracks = {} # well_id -> object_id
        
        for well in range(1, 7):
            # Objects in this well
            ids_in_well = obj_wells[obj_wells == well].index
            if len(ids_in_well) == 0:
                print(f"Well {well}: No tracks found.")
                continue
                
            # Filter dataframe for these objects
            well_df = df[df['object_id'].isin(ids_in_well)]
            
            # Count frames per object
            counts = well_df['object_id'].value_counts()
            best_id = counts.idxmax()
            count = counts.max()
            
            well_best_tracks[well] = best_id
            print(f"Well {well}: Best Track ID={best_id} ({count} frames)")

        # --- 3. Calculate Parameters ---
        summary_rows = []
        
        for well in range(1, 7):
            if well not in well_best_tracks:
                # Empty row for missing well
                summary_rows.append({'Well': well, 'TrackID': None, 'Notes': 'No data'})
                continue
                
            obj_id = well_best_tracks[well]
            track_data = df[df['object_id'] == obj_id].copy()
            
            # Sort by frame
            track_data = track_data.sort_values('frame_id')
            
            # --- Metrics Calculation ---
            # 1. Average velocity (Speed)
            speed_col = safe_series(track_data, 'speed')
            avg_vel = speed_col.mean()
            # 2. S.D. of velocity
            sd_vel = speed_col.std()
            
            # 3. Average acceleration
            accel_col = safe_series(track_data, 'acceleration')
            avg_acc = accel_col.abs().mean()
            # 4. S.D. of acceleration
            sd_acc = accel_col.std()
            
            # 5. Average track angle
            angle_col = safe_series(track_data, 'track_angle')
            avg_angle = angle_col.mean()
            # 6. S.D. of track angle
            sd_angle = angle_col.std()
            
            # 7. Average angular velocity
            ang_vel_col = safe_series(track_data, 'angular_velocity')
            avg_ang_vel = ang_vel_col.abs().mean()
            # 8. S.D. of angular velocity
            sd_ang_vel = ang_vel_col.std()
            
            # 9. Average angular acceleration
            ang_acc_col = safe_series(track_data, 'angular_acceleration')
            avg_ang_acc = ang_acc_col.abs().mean()
            # 10. S.D. of angular acceleration
            sd_ang_acc = ang_acc_col.std()
            
            # 11. Dispersion of path
            disp_col = safe_series(track_data, 'dispersion')
            dispersion = disp_col.max()
            
            # 12. Average angle of particle
            part_angle_col = safe_series(track_data, 'particle_angle')
            avg_part_angle = part_angle_col.mean() if 'particle_angle' in track_data.columns else None
            # 13. S.D. of angle of particle
            sd_part_angle = part_angle_col.std() if 'particle_angle' in track_data.columns else None
            
            # 14. Average meander of path
            meander_col = safe_series(track_data, 'meander_index')
            avg_meander = meander_col.mean()
            # 15. S.D. of meander
            sd_meander = meander_col.std()
            
            # 16. Average time of no movement
            is_stopped = safe_series(track_data, 'is_stopped').astype(bool)
            bouts = []
            current_bout = 0
            for stopped in is_stopped:
                if stopped:
                    current_bout += 1
                else:
                    if current_bout > 0:
                        bouts.append(current_bout)
                    current_bout = 0
            if current_bout > 0: bouts.append(current_bout)
            
            if not bouts:
                avg_stop_time = 0
                sd_stop_time = 0
            else:
                avg_stop_time = np.mean(bouts)
                sd_stop_time = np.std(bouts)
                
            # 18. Average angle between path and particle
            rel_angle_col = safe_series(track_data, 'path_particle_angle_diff')
            avg_rel_angle = rel_angle_col.mean() if 'path_particle_angle_diff' in track_data.columns else None
            sd_rel_angle = rel_angle_col.std() if 'path_particle_angle_diff' in track_data.columns else None
    
            row = {
                'Well': well,
                'TrackID': obj_id,
                'Average velocity': avg_vel,
                'S.D. velocity': sd_vel,
                'Average acceleration': avg_acc,
                'S.D. acceleration': sd_acc,
                'Average track angle': avg_angle,
                'S.D. track angle': sd_angle,
                'Average angular velocity': avg_ang_vel,
                'S.D. angular velocity': sd_ang_vel,
                'Average angular acceleration': avg_ang_acc,
                'S.D. angular acceleration': sd_ang_acc,
                'Dispersion of path': dispersion,
                'Average angle of particle': avg_part_angle,
                'S.D. angle of particle': sd_part_angle,
                'Average meander': avg_meander,
                'S.D. meander': sd_meander,
                'Average time of no movement': avg_stop_time,
                'S.D. time of no movement': sd_stop_time,
                'Average angle path-particle': avg_rel_angle,
                'S.D. angle path-particle': sd_rel_angle
            }
            summary_rows.append(row)
    
        print(f"Generating summary for {base_name}...")
        summary_df = pd.DataFrame(summary_rows)
        print(summary_df)
        
        try:
            summary_df.to_excel(OUTPUT_FILE, index=False)
            print(f"Summary saved to {OUTPUT_FILE}")
        except PermissionError:
            print(f"\n[Warning] Permission denied: Could not write to {OUTPUT_FILE}.")
            print("Please close the file if it is open in Excel and run the script again.")

if __name__ == "__main__":
    analyze_wells()
