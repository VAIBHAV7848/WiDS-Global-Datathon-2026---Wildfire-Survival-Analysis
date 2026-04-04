"""
h_blend Ensemble Script for WiDS 2026
=====================================
Replicates the exact h_blend approach from the friend's notebook that scored 0.97175.

The approach:
1. Takes N submission CSV files as input
2. Splits each into 4 separate files (one per time horizon: 12h, 24h, 48h, 72h)
3. For each time horizon, runs the h_blend algorithm:
   a. Merges all model predictions for that horizon on event_id
   b. For each row, sorts model predictions (ascending or descending)
   c. Uses the sort order to assign "correct weights" (subwts) to each model
   d. Computes: ensemble = sum(pred_j * (weight_j + subwt[rank_j]))
   e. Final blend = desc_weight * desc_ensemble + asc_weight * asc_ensemble
4. Merges the 4 blended horizons back into a single submission

Key parameters from the friend's notebook:
- Ensemble_of_solutions: ['0.97055','0.97085','0.97092','0.97167']
- weights:               [0.05,     0.05,     0.05,     0.85   ]
- type_sort: 'asc/desc' with asc=0.30, desc=0.70
- subwts: [-0.07, -0.03, -0.01, +0.11]

The subwts are "correction weights" that reward/penalize models based on their rank
position for each individual row. A model that is the highest (or lowest, depending
on sort direction) gets subwt[0], the next gets subwt[1], etc.
"""

import numpy as np
import pandas as pd
import os
import sys
import copy
import shutil


def h_blend(params, details=False):
    """
    Horizontal Blend function - the core ensemble algorithm.
    
    Parameters
    ----------
    params : dict with keys:
        'path'      : str, directory containing the split model CSVs
        'id_target' : list[str, str], [id_column_name, target_column_name]
        'type_sort' : list, ['asc/desc', asc_weight, desc_weight]
        'subwts'    : list[float], correction weights for each rank position
        'subm'      : list[dict], each with 'name', 'weight', 'color'
    details : bool, if True print intermediate results
    
    Returns
    -------
    pd.DataFrame with columns [id_col, target_col]
    """
    dk = copy.deepcopy(params)
    
    type_sort   = params['type_sort'][0]
    dk['asc']   = params['type_sort'][1]
    dk['desc']  = params['type_sort'][2]
    dk['id']    = params['id_target'][0]
    dk['target']= params['id_target'][1]
    
    def read(dk, i):
        """Read a single model's split CSV and rename the target column to the model name."""
        tnm = dk["subm"][i]["name"]
        FiN = dk["path"] + tnm + ".csv"
        df = pd.read_csv(FiN).rename(columns={'target': tnm, 'pred': tnm, dk["target"]: tnm})
        return df
    
    def merge(dfs_subm):
        """Merge all model DataFrames on the id column."""
        df_subms = pd.merge(dfs_subm[0], dfs_subm[1], on=[dk['id']])
        for i in range(2, len(dk["subm"])):
            df_subms = pd.merge(df_subms, dfs_subm[i], on=[dk['id']])
        return df_subms
    
    def da(dk, sorting_direction, show_details):
        """
        Core blend logic for a single sorting direction.
        
        For each row:
        1. Sort models by their prediction value (asc or desc)
        2. Assign correction weights based on sort rank
        3. Compute weighted sum: sum(pred_j * (weight_j + subwt[rank_of_j]))
        """
        df_subms = merge([read(dk, i) for i in range(len(dk["subm"]))])
        cols = [col for col in df_subms.columns if col != dk['id']]
        short_name_cols = [c for c in cols]
        
        def alls(x, sd=sorting_direction, cs=cols):
            """Sort model names by their prediction values for this row."""
            reverse = True if sd == 'desc' else False
            tes = {c: x[c] for c in cs}.items()
            subms_sorted = [t[0] for t in sorted(tes, key=lambda k: k[1], reverse=reverse)]
            return subms_sorted
        
        # Build the weight structure
        correct_sub_weights = [wt for wt in dk["subwts"]]
        weights = [subm['weight'] for subm in dk["subm"]]
        
        def correct(x, cs=cols, w=weights, cw=correct_sub_weights):
            """
            Compute the blended prediction for a single row.
            
            For each model j:
              - Find its rank position ic[j] in the sorted order
              - Contribution = pred_j * (main_weight_j + correction_weight[rank_position])
            """
            ic = [x['alls'].index(c) for c in short_name_cols]
            cS = [x[cols[j]] * (w[j] + cw[ic[j]]) for j in range(len(cols))]
            return sum(cS)
        
        # Apply sorting and blending
        df_subms['alls'] = df_subms.apply(lambda x: alls(x), axis=1)
        df_subms[dk["target"]] = df_subms.apply(lambda x: correct(x), axis=1)
        
        # Rename columns for display
        schema_rename = {old_nc: new_shnc for old_nc, new_shnc in zip(cols, short_name_cols)}
        df_subms = df_subms.rename(columns=schema_rename)
        df_subms = df_subms.rename(columns={dk["target"]: "ensemble"})
        
        # Add separator columns for display
        df_subms.insert(loc=1, column=' _ ', value=['   '] * len(df_subms))
        df_subms[' _ '] = df_subms[' _ '].astype(str)
        
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.float_format', '{:.5f}'.format)
        
        vcols = [dk['id']] + [' _ '] + short_name_cols + [' _ '] + ['alls'] + [' _ '] + ['ensemble']
        df_subms = df_subms[vcols]
        
        if show_details and sorting_direction == 'desc':
            print(f"\n=== {dk['target']} ({sorting_direction}) ===")
            print(df_subms.head(5).to_string())
        
        df_subms = df_subms.rename(columns={"ensemble": dk["target"]})
        return df_subms[[dk['id'], dk['target']]]
    
    def ensemble_da(dk, show_details):
        """
        Combine ascending and descending sorted blends.
        Final = desc_weight * desc_blend + asc_weight * asc_blend
        """
        dfD = da(dk, 'desc', show_details)
        dfA = da(dk, 'asc', show_details)
        dfA[dk['target']] = dk['desc'] * dfD[dk['target']] + dfA[dk['target']] * dk['asc']
        return dfA
    
    result = ensemble_da(dk, details)
    return result


def read_subm(path, name):
    """Read a submission CSV file."""
    return pd.read_csv(f'{path}{name}.csv')


def split_df(df):
    """Split a submission DataFrame into 4 DataFrames, one per time horizon."""
    df12h = df.copy().drop(columns=['prob_24h', 'prob_48h', 'prob_72h'])
    df24h = df.copy().drop(columns=['prob_12h', 'prob_48h', 'prob_72h'])
    df48h = df.copy().drop(columns=['prob_12h', 'prob_24h', 'prob_72h'])
    df72h = df.copy().drop(columns=['prob_12h', 'prob_24h', 'prob_48h'])
    return [df12h, df24h, df48h, df72h]


def save_dfs(dfs, path, prefix_name):
    """Save the 4 split DataFrames and return their short file names."""
    dfs[0].to_csv(f'{path}{prefix_name}_12h.csv', index=False)
    dfs[1].to_csv(f'{path}{prefix_name}_24h.csv', index=False)
    dfs[2].to_csv(f'{path}{prefix_name}_48h.csv', index=False)
    dfs[3].to_csv(f'{path}{prefix_name}_72h.csv', index=False)
    return [f'{prefix_name}_{hours}' for hours in '12h,24h,48h,72h'.split(',')]


def split_and_save(df, path, prefix_name):
    """Split a submission into 4 horizon files and save them."""
    dfs = split_df(df)
    short_file_names = save_dfs(dfs, path, prefix_name)
    return short_file_names


def reGroup(i, Ms):
    """Regroup: for time horizon index i, get that entry from each model's name list."""
    group_i_col = [M[i] for M in Ms]
    return group_i_col


def run_h_blend(submission_files, submission_weights, output_file='submission_hblend.csv',
                subwts=None, type_sort=None):
    """
    Main entry point: Run the h_blend ensemble.
    
    Parameters
    ----------
    submission_files : list[str], paths to submission CSV files
    submission_weights : list[float], weight for each submission (should sum to ~1.0)
    output_file : str, path for the output submission
    subwts : list[float], correction weights for rank positions (default: [-0.07,-0.03,-0.01,+0.11])
    type_sort : list, ['asc/desc', asc_weight, desc_weight] (default: ['asc/desc', 0.30, 0.70])
    
    Returns
    -------
    pd.DataFrame, the final blended submission
    """
    if subwts is None:
        subwts = [-0.07, -0.03, -0.01, +0.11]
    if type_sort is None:
        type_sort = ['asc/desc', 0.30, 0.70]
    
    n_models = len(submission_files)
    assert len(submission_weights) == n_models, "Must have same number of weights as files"
    assert len(subwts) == n_models, "Must have same number of subwts as files"
    
    # Create temp working directory
    path_work = os.path.join(os.path.dirname(output_file) or '.', '_hblend_temp') + '/'
    if os.path.isdir(path_work):
        shutil.rmtree(path_work)
    os.makedirs(path_work, exist_ok=True)
    
    colors = ['mediumblue', 'orange', 'green', 'crimson', 'purple', 'brown', 'pink'][:n_models]
    
    try:
        # Step 1: Read all submissions and split by time horizon
        print(f"Loading {n_models} submissions...")
        all_dfs = []
        for i, fpath in enumerate(submission_files):
            df = pd.read_csv(fpath)
            prefix = f'm{i+1}'
            names = split_and_save(df, path_work, prefix)
            all_dfs.append(names)
            print(f"  [{i+1}] {os.path.basename(fpath)} -> {names}")
        
        # Step 2: Regroup by time horizon
        names12h = reGroup(0, all_dfs)
        names24h = reGroup(1, all_dfs)
        names48h = reGroup(2, all_dfs)
        names72h = reGroup(3, all_dfs)
        
        # Step 3: Run h_blend for each time horizon
        horizons = [
            ('prob_12h', names12h),
            ('prob_24h', names24h),
            ('prob_48h', names48h),
            ('prob_72h', names72h),
        ]
        
        horizon_results = {}
        for target, names in horizons:
            print(f"\nBlending {target}...")
            params = {
                'path': path_work,
                'id_target': ['event_id', target],
                'type_sort': type_sort,
                'subwts': subwts,
                'subm': [
                    {'name': names[i], 'weight': submission_weights[i], 'color': colors[i]}
                    for i in range(n_models)
                ]
            }
            horizon_results[target] = h_blend(params, details=True)
        
        # Step 4: Merge all horizons
        print("\nMerging horizons...")
        df = horizon_results['prob_12h']
        for target in ['prob_24h', 'prob_48h', 'prob_72h']:
            df = pd.merge(df, horizon_results[target], on='event_id')
        
        # Step 5: Validate
        print(f"\nFinal submission shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nPrediction ranges:")
        for col in ['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']:
            print(f"  {col}: [{df[col].min():.5f}, {df[col].max():.5f}] mean={df[col].mean():.5f}")
        
        # Check monotonicity
        mono_violations = 0
        for _, row in df.iterrows():
            vals = [row['prob_12h'], row['prob_24h'], row['prob_48h'], row['prob_72h']]
            for j in range(len(vals)-1):
                if vals[j] > vals[j+1]:
                    mono_violations += 1
                    break
        print(f"  Monotonicity violations: {mono_violations}/{len(df)} rows")
        
        # Check for NaN/null
        null_count = df.isnull().sum().sum()
        print(f"  Null values: {null_count}")
        
        # Save
        df.to_csv(output_file, index=False)
        print(f"\n✅ Saved to: {output_file}")
        print(df.head(10).to_string())
        
        return df
    
    finally:
        # Cleanup temp directory
        if os.path.isdir(path_work):
            shutil.rmtree(path_work)
            print(f"\nCleaned up temp directory: {path_work}")


if __name__ == '__main__':
    # ==========================================================================
    # CONFIGURATION - Modify these to match your submissions
    # ==========================================================================
    
    # Option 1: Use the EXACT same submissions as the friend (requires Kaggle download)
    # To download, authenticate kaggle CLI and run:
    #   kaggle kernels output sarthakniwate13/0-97-microeda-gbsa-lgbm-rankblend-plattcalib -p kaggle_subs/0.97055
    #   kaggle kernels output furqonaryadana/0-9707-cv-bagged-lgbm-survival-eda -p kaggle_subs/0.97085
    #   kaggle kernels output loopassembly/0-97092-450-model-fold-fused-survival-engine -p kaggle_subs/0.97092
    #   kaggle kernels output furqonaryadana/0-9716-tri-survival-stack-distancestratifiedblend -p kaggle_subs/0.97167
    # Then rename each output submission.csv to the score name.
    
    kaggle_subs_dir = os.path.join(os.path.dirname(__file__), 'kaggle_subs')
    
    use_kaggle_subs = all(
        os.path.exists(os.path.join(kaggle_subs_dir, f'{score}.csv'))
        for score in ['0.97055', '0.97085', '0.97092', '0.97167']
    )
    
    if use_kaggle_subs:
        print("=" * 60)
        print("Using Kaggle public submissions (same as friend's notebook)")
        print("=" * 60)
        submission_files = [
            os.path.join(kaggle_subs_dir, '0.97055.csv'),
            os.path.join(kaggle_subs_dir, '0.97085.csv'),
            os.path.join(kaggle_subs_dir, '0.97092.csv'),
            os.path.join(kaggle_subs_dir, '0.97167.csv'),
        ]
        # Exact weights from the friend's notebook
        weights = [0.05, 0.05, 0.05, 0.85]
        subwts = [-0.07, -0.03, -0.01, +0.11]
        type_sort = ['asc/desc', 0.30, 0.70]
    else:
        print("=" * 60)
        print("Kaggle submissions not found locally.")
        print("Using best local submissions instead.")
        print("=" * 60)
        print()
        print("To get the EXACT same result as your friend (0.97175),")
        print("you need to download the 4 public Kaggle submissions.")
        print()
        print("Steps:")
        print("1. Set up Kaggle API authentication:")
        print("   - Go to https://www.kaggle.com/settings")
        print("   - Click 'Create New Token' to download kaggle.json")
        print("   - Place it at C:\\Users\\<username>\\.kaggle\\kaggle.json")
        print()
        print("2. Run these commands:")
        print(f"   mkdir {kaggle_subs_dir}")
        cmds = [
            ('sarthakniwate13/0-97-microeda-gbsa-lgbm-rankblend-plattcalib', '0.97055'),
            ('furqonaryadana/0-9707-cv-bagged-lgbm-survival-eda', '0.97085'),
            ('loopassembly/0-97092-450-model-fold-fused-survival-engine', '0.97092'),
            ('furqonaryadana/0-9716-tri-survival-stack-distancestratifiedblend', '0.97167'),
        ]
        for kernel, score in cmds:
            print(f"   kaggle kernels output {kernel} -p {kaggle_subs_dir}/{score}")
            print(f"   (then rename submission.csv to {score}.csv)")
        print()
        print("3. Re-run this script.")
        print()
        
        # Fall back to best local submissions
        local_subs = [
            'submission_09.csv',  # Our best local
            'submission_11.csv',
            'submission_12.csv',
            'submission_13.csv',
        ]
        submission_files = [
            os.path.join(os.path.dirname(__file__), f)
            for f in local_subs
            if os.path.exists(os.path.join(os.path.dirname(__file__) or '.', f))
        ]
        
        if len(submission_files) < 2:
            print("ERROR: Need at least 2 submission files for blending.")
            sys.exit(1)
        
        n = len(submission_files)
        print(f"\nFound {n} local submissions:")
        for f in submission_files:
            print(f"  - {os.path.basename(f)}")
        
        # Use equal weights with heavy weight on best (like the friend's approach)
        if n == 4:
            weights = [0.05, 0.05, 0.05, 0.85]
            subwts = [-0.07, -0.03, -0.01, +0.11]
        elif n == 3:
            weights = [0.05, 0.10, 0.85]
            subwts = [-0.05, -0.02, +0.07]
        elif n == 2:
            weights = [0.15, 0.85]
            subwts = [-0.05, +0.05]
        else:
            # Equal weights
            w = 1.0 / n
            weights = [w] * n
            subwts = [0.0] * n
        
        type_sort = ['asc/desc', 0.30, 0.70]
    
    # Run the ensemble
    result = run_h_blend(
        submission_files=submission_files,
        submission_weights=weights,
        output_file=os.path.join(os.path.dirname(__file__) or '.', 'submission_hblend.csv'),
        subwts=subwts,
        type_sort=type_sort,
    )
