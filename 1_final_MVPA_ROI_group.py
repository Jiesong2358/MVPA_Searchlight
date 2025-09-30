### This code performs group-level MVPA analysis including multiple comparison corrections,
### statistical testing, and generates publication-quality box-dot plots.
### It processes individual subject accuracy and coefficient maps, applies a mask,
### computes group statistics, and visualizes the results.
### To use it, change the output directory and mask path as needed.

import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import stats
from nilearn import plotting, image
from nilearn.image import math_img, resample_to_img
from nilearn.masking import apply_mask, unmask
from nilearn.glm import threshold_stats_img
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import fdrcorrection

def perform_group_analysis(out_dir):
    """
    Perform complete group-level MVPA analysis with:
    - Multiple comparison corrections
    - Debugged statistical testing
    - Publication-quality box-dot plots
    """
    # =============================================
    # 1. Collect and organize accuracy results
    # =============================================
    print("\n[1/6] Collecting accuracy results...")
    accuracy_files = glob.glob(os.path.join(out_dir, '*_accuracy_beta.txt'))
    
    if not accuracy_files:
        raise ValueError("No accuracy files found in the output directory")
    
    results = []
    for acc_file in accuracy_files:
        subject_id = os.path.basename(acc_file).split('_')[0]
        with open(acc_file, 'r') as f:
            lines = f.readlines()
            mean_acc = float(lines[0].split(': ')[1].strip())
            fold_accs = [float(line.split(': ')[1].strip()) for line in lines[2:]]
        results.append({
            'subject': subject_id,
            'mean_accuracy': mean_acc,
            'fold_accuracies': fold_accs
        })
    
    df = pd.DataFrame(results)
    acc_csv_path = os.path.join(out_dir, 'group_accuracy_results.csv')
    df.to_csv(acc_csv_path, index=False)

    # =============================================
    # 2. Load and verify mask
    # =============================================
    print("\n[2/6] Loading and verifying mask...")
    mask_path = 'S:/GVuilleumier/GVuilleumier/groups/jies/Spatial_Navigation/Final_data/Atlas/ROIs/PIT_mask.nii.gz'
    mask_img = nib.load(mask_path)
    print(f"Mask shape: {mask_img.shape}")
    print(f"Mask voxel size: {mask_img.header.get_zooms()}")

    # =============================================
    # 3. Process coefficient maps with alignment checks
    # =============================================
    print("\n[3/6] Processing coefficient maps...")
    coef_files = glob.glob(os.path.join(out_dir, '*_coef_beta.nii.gz'))
    if not coef_files:
        raise ValueError("No coefficient maps found")

    first_coef = nib.load(coef_files[0])
    print(f"First coef map shape: {first_coef.shape}")
    print(f"Coef map voxel size: {first_coef.header.get_zooms()}")

    # Resample mask if needed
    if not np.allclose(mask_img.affine, first_coef.affine, atol=1e-3):
        print("Resampling mask to match coefficient maps...")
        mask_img = resample_to_img(mask_img, first_coef, interpolation='nearest')

    # Load all data with verification
    coef_data = []
    for coef_file in coef_files:
        img = nib.load(coef_file)
        if not np.allclose(img.affine, first_coef.affine, atol=1e-3):
            img = resample_to_img(img, first_coef)
        coef_data.append(img.get_fdata())
    
    # =============================================
    # 4. Compute group averages
    # =============================================
    print("\n[4/6] Computing group averages...")
    mean_coef = np.mean(coef_data, axis=0)
    mean_coef_img = nib.Nifti1Image(mean_coef, first_coef.affine)
    mean_coef_img.to_filename(os.path.join(out_dir, 'group_mean_coef.nii.gz'))

    # =============================================
    # 5. Statistical testing with robust corrections
    # =============================================
    print("\n[5/6] Running statistical tests...")
    masked_data = np.array([apply_mask(nib.Nifti1Image(d, first_coef.affine), mask_img) for d in coef_data])
    t_vals, p_vals = stats.ttest_1samp(masked_data, 0, axis=0)
    n_voxels = np.sum(mask_img.get_fdata() > 0)
    print(f"Total voxels in mask: {n_voxels}")
    print(f"T-value range: [{np.min(t_vals):.2f}, {np.max(t_vals):.2f}]")
    print(f"P-value range: [{np.min(p_vals):.4f}, {np.max(p_vals):.4f}]")

    # Create proper statistical maps
    t_map_img = unmask(t_vals, mask_img)
    
    # Uncorrected threshold
    uncorrected_thresh = stats.t.ppf(1 - 0.001/2, len(coef_files)-1)
    uncorrected_sig = (np.abs(t_vals) > uncorrected_thresh)
    print(f"\nUncorrected (p<0.001): {np.sum(uncorrected_sig)} significant voxels")

    # FDR correction
    valid_mask = ~np.isnan(p_vals)  # Exclude NaN values
    valid_pvals = p_vals[valid_mask].flatten()

    try:
        fdr_rejected, _ = fdrcorrection(valid_pvals, alpha=0.05)
        fdr_sig = np.zeros_like(p_vals, dtype=bool)
        fdr_sig[valid_mask] = fdr_rejected.reshape(p_vals[valid_mask].shape)
        print(f"FDR-corrected: {np.sum(fdr_sig)} significant voxels")
    except Exception as e:
        print(f"FDR correction failed: {str(e)}")
    # Cluster correction
    cluster_threshold = stats.t.ppf(1 - 0.01/2, len(coef_files)-1)  # p<0.01
    if np.sum(uncorrected_sig) > 0:
        cluster_map, threshold = threshold_stats_img(
            t_map_img,
            alpha=0.05,
            height_control='fpr',
            threshold=cluster_threshold,
            cluster_threshold=10  # Minimum cluster size in voxels
        )
        cluster_sig = cluster_map.get_fdata() > 0

    # Save statistical maps
    print("\nSaving statistical maps...")
    nib.save(t_map_img, os.path.join(out_dir, 'group_t_values.nii.gz'))
    nib.save(unmask(uncorrected_sig.astype(float), mask_img), os.path.join(out_dir, 'group_sig_uncorrected.nii.gz'))
    nib.save(unmask(fdr_sig.astype(float), mask_img), os.path.join(out_dir, 'group_sig_fdr.nii.gz'))
    cluster_sig = np.zeros_like(mask_img.get_fdata(), dtype=bool)
    if np.sum(cluster_sig) > 0:
        nib.save(cluster_map, os.path.join(out_dir, 'group_sig_clusters.nii.gz'))

    # =============================================
    # 6. Generate visualizations
    # =============================================
    print("\n[6/6] Generating visualizations...")

    # 6A. Create vertical box plot of mean accuracies by subject
    plt.figure(figsize=(8, 8))
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11
    })

    # Create vertical boxplot with transparent boxes
    box = sns.boxplot(
        y='mean_accuracy',
        data=df,
        color='white',
        width=0.2,
        linewidth=1.5,
        fliersize=0,
        boxprops=dict(facecolor='none', edgecolor='black'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
        medianprops=dict(color='red', linewidth=2)
    )

    # Add individual data points with subject labels
    sns.stripplot(
        y='mean_accuracy',
        data=df,
        color='black',
        size=8,
        alpha=0.7,
        jitter=0.2,
        edgecolor='black',
        linewidth=0.5
    )

    # Add reference lines and labels
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1.5, label='Chance')
    plt.title('Group Decoding Accuracy by Subject', pad=20)
    plt.ylabel('Accuracy', labelpad=10)
    plt.xlabel('')  # Remove x-label as it's not needed
    plt.ylim(0, 1.05)
    plt.xlim(-0.5, 0.5)  # Adjust to make room for subject labels
    plt.yticks(np.arange(0.4, 1.1, 0.1))

    # Add group mean line
    group_mean = df['mean_accuracy'].mean()
    group_std = df['mean_accuracy'].std()
    plt.axhline(group_mean, color='blue', linestyle=':', linewidth=1.5, 
            label=f'Group Mean = {group_mean:.2f} ± {group_std:.2f}')
    plt.legend(loc='upper right')

    # Remove x-axis ticks and labels
    plt.xticks([])

    # Save plot
    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'group_mean_accuracy_boxplot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy plot to: {plot_path}")
    # 6B. Additional statistical visualizations
    plotting.plot_stat_map(
        t_map_img,
        title='Group T-map',
        output_file=os.path.join(out_dir, 'group_t_map.png'),
        display_mode='ortho',
        draw_cross=False,
    )

    if np.sum(fdr_sig) > 0:
        # Create masked t-values (only show significant voxels)
        masked_t = np.where(fdr_sig, t_vals, np.nan)
        masked_t_img = unmask(masked_t, mask_img)
        
        plotting.plot_stat_map(
            masked_t_img,  # Now shows real t-values
            title=f'FDR-corrected Results\n{np.sum(fdr_sig)} voxels',
            output_file=os.path.join(out_dir, 'group_sig_fdr.png'),
            display_mode='ortho',
            draw_cross=False,
            colorbar=True,  # Now shows meaningful t-values
            #cut_coords=[9, -43, 0], # MNI coordinates for RSC
            #cut_coords=[23, -37, 14], # MNI coordinates for PHC
            #cut_coords=[-29, -17, -23], # MNI coordinates for H
            #cut_coords=[13, 16, 0], # MNI coordinates for Caudate
            # cut_coords=[5, -64, 57], # MNI coordinates for SPC
            cut_coords=[50, -72, -13], # MNI coordinates for PIT

        )
        
    # plot the mean coefficient map
    plotting.plot_stat_map(
        mean_coef_img,
        title='Group Mean Coefficient Map',
        output_file=os.path.join(out_dir, 'group_mean_coef.png'),
        display_mode='ortho',
        draw_cross=False,
        colorbar=True,
        threshold='auto',
        #cut_coords=[9, -43, 0], # MNI coordinates for RSC
        # cut_coords=[23, -37, 14], # MNI coordinates for PHC
        #cut_coords=[-29, -17, -23], # MNI coordinates for H
        # cut_coords=[13, 16, 0], # MNI coordinates for Caud
        # cut_coords=[5, -64, 57], # MNI coordinates for SPC
        cut_coords=[50, -72, -13], # MNI coordinates for PIT

    )
    # plot the FDR-corrected results but show the mean coefficient values instead????
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    out_dir = 'S:/GVuilleumier/GVuilleumier/groups/jies/Spatial_Navigation/Final_data/GLM/runwise_beta_maps_unsmoothed/2s_bins1-2_long_short_PIT'
    os.makedirs(out_dir, exist_ok=True)
    perform_group_analysis(out_dir)