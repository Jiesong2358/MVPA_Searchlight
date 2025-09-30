import os
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.image import load_img, mean_img
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.masking import compute_epi_mask
import matplotlib.pyplot as plt

# ========================
# Configuration
# ========================
root_dir = 'S:/GVuilleumier/GVuilleumier/groups/jies/Spatial_Navigation/Final_data/nifti_new/derivatives'
events_root_dir = 'S:/GVuilleumier/GVuilleumier/groups/jies/Spatial_Navigation/Final_data/nifti_new'
output_root = 'S:/GVuilleumier/GVuilleumier/groups/jies/Spatial_Navigation/Final_data/GLM/runwise_beta_maps_unsmoothed'
if not os.path.exists(output_root):
    os.makedirs(output_root)
tr = 1.3  # Repetition time in seconds

# Subjects to process (excluding bad subjects)
subject_range = [sub for sub in range(1, 50) if sub not in [1, 2, 3, 4, 6, 11, 13, 14, 23, 36, 48]]

# ========================
# GLM Analysis per Run (No Smoothing)
# ========================
def run_glm_per_run(subject, run):
    print(f"\nProcessing sub-{subject:02d} run-{run:02d}")
    
    # Create output directory
    sub_dir = os.path.join(output_root, f'sub-{subject:02d}')
    os.makedirs(sub_dir, exist_ok=True)
    
    # ========================
    # 1. Data Preparation
    # ========================
    # Input files
    func_file = os.path.join(root_dir, f'sub-{subject:02d}/func/sub-{subject:02d}_task-int3_run-{run:02d}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
    events_file = os.path.join(events_root_dir, f'sub-{subject:02d}/func/sub-{subject:02d}_task-rep_ret_run-{run:02d}_imagination_slidingwindows_4s_window6-10s.tsv')
    confound_file = os.path.join(root_dir, f'sub-{subject:02d}/func/sub-{subject:02d}_task-int3_run-{run:02d}_desc-confounds_timeseries.tsv')
    
    # Load fMRI data (unsmoothed)
    func_img = load_img(func_file)
    n_scans = func_img.shape[3]
    frame_times = np.arange(n_scans) * tr
    
    # ========================
    # 2. Preprocessing
    # ========================
    # Create mask from unsmoothed data
    mask_file = os.path.join(sub_dir, f'sub-{subject:02d}_run-{run:02d}_mask.nii.gz')
    if not os.path.exists(mask_file):
        mean_img_ = mean_img(func_img)  # Using original unsmoothed data
        mask_img = compute_epi_mask(mean_img_)
        mask_img.to_filename(mask_file)
    else:
        mask_img = load_img(mask_file)
    
    # ========================
    # 3. Design Matrix
    # ========================
    # Load events and confounds
    events = pd.read_csv(events_file, sep='\t')
    confounds = pd.read_csv(confound_file, sep='\t')
    
    # Select motion regressors
    motion_regressors = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    confounds = confounds[motion_regressors + ['global_signal'] + [f'a_comp_cor_0{i}' for i in range(10)]]
    
    # Create design matrix
    design_matrix = make_first_level_design_matrix(
        frame_times,
        events,
        hrf_model='spm',
        drift_model='polynomial',
        drift_order=3,
        add_regs=confounds
    )
    
    # Save design matrix plot
    plot_file = os.path.join(sub_dir, f'sub-{subject:02d}_run-{run:02d}_sliding_windows_design_matrix.png')
    plot_design_matrix(design_matrix)
    plt.savefig(plot_file)
    plt.close()
    
    # ========================
    # 4. GLM Fitting (No Smoothing)
    # ========================
    glm = FirstLevelModel(
        t_r=tr,
        mask_img=mask_img,
        smoothing_fwhm=None,  # No spatial smoothing
        high_pass=1/128,      # High-pass filter cutoff (1/128 Hz)
        standardize=True,     # Z-score normalization
        noise_model='ar1',    # Auto-regressive noise model
        minimize_memory=True
    )
    
    # Fit GLM to unsmoothed data
    glm.fit(func_img, design_matrices=design_matrix)
    
    # ========================
    # 5. Contrast Estimation
    # ========================
    # Mapping between design matrix columns and desired output names
    condition_mapping = {
        'car_repe': 'Sliding_win_6_10_Imag_repe_car',
        'int1_repe': 'Sliding_win_6_10_Imag_repe_int1',
        'int1_rerta': 'Sliding_win_6_10_Imag_retra_int1',
        'int2_repe': 'Sliding_win_6_10_Imag_repe_int2',
        'int2_rerta': 'Sliding_win_6_10_Imag_retra_int2',
        'phone_retra': 'Sliding_win_6_10_Imag_retra_phone'
    }
#     condition_mapping = {
#     'Car_I1': 'Car_I1_decision',
#     'Car_I2': 'Car_I2_decision',
#     'Car_I3': 'Car_I3_decision',
#     'PhoneBox_I1': 'PhoneBox_I1_decision',
#     'PhoneBox_I2': 'PhoneBox_I2_decision',
#     'PhoneBox_I3': 'PhoneBox_I3_decision',
#     'Rep_I1_I1': 'Rep_I1_I1_decision',
#     'Rep_I1_I2': 'Rep_I1_I2_decision',
#     'Rep_I1_I3': 'Rep_I1_I3_decision',
#     'Rep_I2_I2': 'Rep_I2_I2_decision',
#     'Rep_I2_I3': 'Rep_I2_I3_decision',
#     'Ret_I1_I1': 'Ret_I1_I1_decision',
#     'Ret_I1_I2': 'Ret_I1_I2_decision',
#     'Ret_I1_I3': 'Ret_I1_I3_decision',
#     'Ret_I2_I2': 'Ret_I2_I2_decision',
#     'Ret_I2_I3': 'Ret_I2_I3_decision'
# }
    
    # Verify all expected conditions exist in design matrix
    for design_name in condition_mapping.keys():
        if design_name not in design_matrix.columns:
            raise ValueError(f"Design matrix column {design_name} not found. Available columns: {design_matrix.columns.tolist()}")
    
    # Create one contrast per condition
    for design_name, output_name in condition_mapping.items():
        contrast = np.zeros(len(design_matrix.columns))
        contrast[design_matrix.columns.get_loc(design_name)] = 1
        
        # Compute beta map (effect size)
        beta_map = glm.compute_contrast(contrast, output_type='effect_size')
        beta_map.to_filename(os.path.join(sub_dir, f'sub-{subject:02d}_run-{run:02d}_{output_name}_beta_map.nii.gz'))
        
        # Compute t-map
        t_map = glm.compute_contrast(contrast, stat_type='t')
        t_map.to_filename(os.path.join(sub_dir, f'sub-{subject:02d}_run-{run:02d}_{output_name}_t_map.nii.gz'))
        
        # Compute z-map
        z_map = glm.compute_contrast(contrast, output_type='z_score')
        z_map.to_filename(os.path.join(sub_dir, f'sub-{subject:02d}_run-{run:02d}_{output_name}_z_map.nii.gz'))
        
        print(f"Generated maps for {output_name} (from design column: {design_name})")
    
    print(f"Completed sub-{subject:02d} run-{run:02d}")

# ========================
# Main Execution
# ========================
if __name__ == '__main__':
    for subject in subject_range:
        for run in range(1, 7):  # Runs 1-6
            try:
                run_glm_per_run(subject, run)
            except Exception as e:
                print(f"Error processing sub-{subject:02d} run-{run:02d}: {str(e)}")
                continue