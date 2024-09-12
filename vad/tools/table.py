import os
import pandas as pd
from natsort import natsorted


def read_metrics(metrics_folder):
    detection, map, motion, plan = 0, 0, 0, 0, 
    try:
        detection_csv = pd.read_csv(os.path.join(metrics_folder, "detection.csv"))
        detection = detection_csv.iloc[0, 1]
    except:
        pass
    try:
        motion_csv = pd.read_csv(os.path.join(metrics_folder, "motion.csv"))
        motion = motion_csv.iloc[0, 1]
    except:
        pass
    try:
        map_csv = pd.read_csv(os.path.join(metrics_folder, "map.csv"))
        map = float(map_csv.iloc[3, 1])
    except:
        pass
    try:
        plan_csv = pd.read_csv(os.path.join(metrics_folder, "planning.csv"))
        plan = plan_csv.iloc[2, 5]
    except:
        pass
    return detection, map, motion, plan


def fill_table_black():
    save_folder = './tifscsv/black'
    folders = {
        'vanilla': [
            'test0822black/VAD_base_e2e_mini/attack@pgdloo_0.2_5/metrics',
            'test0904transfer/VAD_base_e2e_mini/attack@vadpgdlooat_0.2_5/metrics',
            'test0904transfer/VAD_base_e2e_mini/attack@uniad_0.2_5/metrics',
        ],
        'fgsmat': [
            'test0904_tradat/VAD_base_e2e_mini/defense@black_pgdloo_0.2_5@0.1_1_fgsmat/metrics',
            'test0904transfer/VAD_base_e2e_mini/trad_defense@vadpgdlooat_0.2_5@0.1_1_10_fgsmat/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@uniad_pgdloo_0.2_5@0.1_1_fgsmat/metrics',
        ],
        'p1at': [
            'test0904_tradat/VAD_base_e2e_mini/defense@black_pgdloo_0.2_5@0.1_5_pgdl1at/metrics',
            'test0904transfer/VAD_base_e2e_mini/trad_defense@vadpgdlooat_0.2_5@0.1_5_10_pgdl1at/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@uniad_pgdloo_0.2_5@0.1_5_pgdl1at/metrics',
        ],
        'p2at': [
            'test0904_tradat/VAD_base_e2e_mini/defense@black_pgdloo_0.2_5@0.1_5_pgdl2at/metrics',
            'test0904transfer/VAD_base_e2e_mini/trad_defense@vadpgdlooat_0.2_5@0.1_5_10_pgdl2at/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@uniad_pgdloo_0.2_5@0.1_5_pgdl2at/metrics',
        ],
        'plooat': [
            'test0904_tradat/VAD_base_e2e_mini/defense@black_pgdloo_0.2_5@0.1_5_pgdlooat/metrics',
            'test0904transfer/VAD_base_e2e_mini/trad_defense@vadpgdlooat_0.2_5@0.1_5_10_pgdlooat/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@uniad_pgdloo_0.2_5@0.1_5_pgdlooat/metrics'
        ],
        'mat': [
            'test0822black/VAD_base_e2e_mini/defense@pgdloo_0.2_5@0.1_0.1_0.1_5_10_at/metrics',
            'test0904transfer/VAD_base_e2e_mini/defense@vadpgdlooat_0.2_5@0.1_0.1_0.1_5_10_at/metrics',
            'test0904transfer/VAD_base_e2e_mini/defense@uniad_0.2_5@0.1_0.1_0.1_5_10_at/metrics'
        ]
    }
    header = ['attack', 'vanilla', 'fgsmat', 'p1at', 'p2at', 'plooat', 'atavg', 'mat']
    attack_types = ['origin', 'tradat', 'uniad']

        # Create empty DataFrames for each metric
    detection_df = pd.DataFrame(columns=header)
    motion_df = pd.DataFrame(columns=header)
    map_df = pd.DataFrame(columns=header)
    plan_df = pd.DataFrame(columns=header)

    # Populate the DataFrames
    for i, attack in enumerate(attack_types):
        # Create rows for each metric type
        detection_row = {'attack': attack}
        motion_row = {'attack': attack}
        map_row = {'attack': attack}
        plan_row = {'attack': attack}

        for folder_type, paths in folders.items():
            detection, motion, map, plan = read_metrics(paths[i])
            
            # Fill rows for each metric type
            detection_row[folder_type] = detection
            motion_row[folder_type] = motion
            map_row[folder_type] = map
            plan_row[folder_type] = plan
            

        # Calculate atavg as the mean of fgsmat, p1at, p2at, and plooat
        for row in [detection_row, motion_row,  map_row, plan_row]:
            row['atavg'] = (row.get('fgsmat', 0) + row.get('p1at', 0) + row.get('p2at', 0) + row.get('plooat', 0)) / 4

        # Append data to each DataFrame
        detection_df = pd.concat([detection_df, pd.DataFrame([detection_row])], ignore_index=True)
        motion_df = pd.concat([motion_df, pd.DataFrame([motion_row])], ignore_index=True)
        map_df = pd.concat([map_df, pd.DataFrame([map_row])], ignore_index=True)
        plan_df = pd.concat([plan_df, pd.DataFrame([plan_row])], ignore_index=True)   
    
    # Round all numerical values to two decimal places
    detection_df = detection_df.round(2)
    motion_df = motion_df.round(2)
    map_df = map_df.round(2)
    plan_df = plan_df.round(2)

    os.makedirs(save_folder, exist_ok=True)

    # Save each DataFrame to a CSV file
    detection_df.to_csv(os.path.join(save_folder, 'detection_metrics.csv'), index=False)
    motion_df.to_csv(os.path.join(save_folder, 'motion_metrics.csv'), index=False)
    map_df.to_csv(os.path.join(save_folder, 'map_metrics.csv'), index=False)
    plan_df.to_csv(os.path.join(save_folder, 'plan_metrics.csv'), index=False)

def fill_table_white():
    save_folder = './tifscsv/white'
    folders = {
        'vanilla': [
            'test0819white/VAD_base_e2e_mini/attack@fgsm_0.2_1/metrics',
            'test0819white/VAD_base_e2e_mini/attack@mifgsm_0.2_5/metrics',
            'test0819white/VAD_base_e2e_mini/attack@pgdl1_288000.0_5/metrics',
            'test0819white/VAD_base_e2e_mini/attack@pgdl2_240.0_5/metrics',
            'test0819white/VAD_base_e2e_mini/attack@pgdloo_0.2_5/metrics',
            'test0819white/VAD_base_e2e_mini/ori/metrics',
        ],
        'fgsmat': [
            'test0904_tradat/VAD_base_e2e_mini/defense@fgsm_0.2_1@0.1_1_fgsmat/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@mifgsm_0.2_5@0.1_1_fgsmat/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@pgdl1_288000.0_5@0.1_1_fgsmat/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@pgdl2_240.0_5@0.1_5_pgdl2at/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@pgdloo_0.2_5@0.1_1_fgsmat/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@no_attack@0.1_1_fgsmat/metrics',
        ],
        'p1at': [
            'test0904_tradat/VAD_base_e2e_mini/defense@fgsm_0.2_1@0.1_5_pgdl1at/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@mifgsm_0.2_5@0.1_5_pgdl1at/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@pgdl1_288000.0_5@0.1_5_pgdl1at/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@pgdl2_240.0_5@0.1_5_pgdl1at/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@pgdloo_0.2_5@0.1_5_pgdl1at/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@no_attack@0.1_5_pgdl1at/metrics',
        ],
        'p2at': [
            'test0904_tradat/VAD_base_e2e_mini/defense@fgsm_0.2_1@0.1_5_pgdl2at/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@mifgsm_0.2_5@0.1_5_pgdl2at/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@pgdl1_288000.0_5@0.1_5_pgdl2at/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@pgdl2_240.0_5@0.1_5_pgdl2at/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@pgdloo_0.2_5@0.1_5_pgdl2at/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@no_attack@0.1_5_pgdl2at/metrics',
        ],
        'plooat': [
            'test0904_tradat/VAD_base_e2e_mini/defense@fgsm_0.2_1@0.1_5_pgdlooat/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@mifgsm_0.2_5@0.1_5_pgdlooat/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@pgdl1_288000.0_5@0.1_5_pgdlooat/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@pgdl2_240.0_5@0.1_5_pgdlooat/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@pgdloo_0.2_5@0.1_5_pgdlooat/metrics',
            'test0904_tradat/VAD_base_e2e_mini/defense@no_attack@0.1_5_pgdlooat/metrics',
        ],
        'mat': [
            'test0819white/VAD_base_e2e_mini/defense@fgsm_0.2_1@0.1_0.1_0.1_5_10_at/metrics',
            'test0819white/VAD_base_e2e_mini/defense@mifgsm_0.2_5@0.1_0.1_0.1_5_10_at/metrics',
            'test0819white/VAD_base_e2e_mini/defense@pgdl1_288000.0_5@0.1_0.1_0.1_5_10_at/metrics',
            'test0819white/VAD_base_e2e_mini/defense@pgdl2_240.0_5@0.1_0.1_0.1_5_10_at/metrics',
            'test0819white/VAD_base_e2e_mini/defense@pgdloo_0.2_5@0.1_0.1_0.1_5_10_at/metrics',
            'test0819white/VAD_base_e2e_mini/defense@no_attack@0.1_0.1_0.1_5_10_at/metrics',
        ]
    }
    header = ['attack', 'vanilla', 'fgsmat', 'p1at', 'p2at', 'plooat', 'atavg', 'mat']
    attack_types = ['fgsm', 'mifgsm', 'pgdl1', 'pgdl2', 'pgdloo', 'origin', ]

        # Create empty DataFrames for each metric
    detection_df = pd.DataFrame(columns=header)
    motion_df = pd.DataFrame(columns=header)
    map_df = pd.DataFrame(columns=header)
    plan_df = pd.DataFrame(columns=header)

    # Populate the DataFrames
    for i, attack in enumerate(attack_types):
        # Create rows for each metric type
        detection_row = {'attack': attack}
        motion_row = {'attack': attack}
        map_row = {'attack': attack}
        plan_row = {'attack': attack}

        for folder_type, paths in folders.items():
            detection, motion, map, plan = read_metrics(paths[i])
            
            # Fill rows for each metric type
            detection_row[folder_type] = detection
            motion_row[folder_type] = motion
            map_row[folder_type] = map
            plan_row[folder_type] = plan
            

        # Calculate atavg as the mean of fgsmat, p1at, p2at, and plooat
        for row in [detection_row, motion_row,  map_row, plan_row]:
            row['atavg'] = (row.get('fgsmat', 0) + row.get('p1at', 0) + row.get('p2at', 0) + row.get('plooat', 0)) / 4

        # Append data to each DataFrame
        detection_df = pd.concat([detection_df, pd.DataFrame([detection_row])], ignore_index=True)
        motion_df = pd.concat([motion_df, pd.DataFrame([motion_row])], ignore_index=True)
        map_df = pd.concat([map_df, pd.DataFrame([map_row])], ignore_index=True)
        plan_df = pd.concat([plan_df, pd.DataFrame([plan_row])], ignore_index=True)   
    
    # Round all numerical values to two decimal places
    detection_df = detection_df.round(2)
    motion_df = motion_df.round(2)
    map_df = map_df.round(2)
    plan_df = plan_df.round(2)

    os.makedirs(save_folder, exist_ok=True)

    # Save each DataFrame to a CSV file
    detection_df.to_csv(os.path.join(save_folder, 'detection_metrics.csv'), index=False)
    motion_df.to_csv(os.path.join(save_folder, 'motion_metrics.csv'), index=False)
    map_df.to_csv(os.path.join(save_folder, 'map_metrics.csv'), index=False)
    plan_df.to_csv(os.path.join(save_folder, 'plan_metrics.csv'), index=False)


if __name__ == "__main__":
    fill_table_white()
