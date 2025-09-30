### This code is used to perform searchlight MVPA with Leave-One-Run-Out cross-validation
### on fMRI beta-maps for different subjects. It loads beta-maps, applies a mask, and computes
### searchlight accuracy maps, saving the results for each subject.
### To use it, change the data path and mask path, change the conditions in the loading function.
import os
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.image import concat_imgs, resample_img
from nilearn.decoding import SearchLight
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from nilearn import plotting

def load_beta_maps_for_subject(subject_dir):
    """Load beta-maps for a subject and return file paths grouped by run."""
    beta_maps = {}
    for file_name in os.listdir(subject_dir):
        # if file_name.endswith('_beta_map.nii.gz') and (
        #     'Imag_repe_int1_bin1_2s' in file_name or 'Imag_repe_int1_bin4_2s' in file_name or
        #     'Imag_retra_int1_bin1_2s' in file_name or 'Imag_retra_int1_bin4_2s' in file_name or
        #     'Imag_repe_int2_bin1_2s' in file_name or 'Imag_repe_int2_bin4_2s' in file_name or
        #     'Imag_retra_int2_bin1_2s' in file_name or 'Imag_retra_int2_bin4_2s' in file_name
        # ):
        if file_name.endswith('_beta_map.nii.gz') and (
            'Imag_repe_int1_bin1_2s' in file_name or 'Rep_I1_I1_decision' in file_name or
            'Imag_retra_int1_bin1_2s' in file_name or 'Ret_I1_I1_decision' in file_name or
            'Imag_repe_int2_bin1_2s' in file_name or 'Rep_I2_I2_decision' in file_name or
            'Imag_retra_int2_bin1_2s' in file_name or 'Ret_I2_I2_decision' in file_name
        ):
            run_id = file_name.split('_')[1]
            if run_id not in beta_maps:
                beta_maps[run_id] = []
            beta_maps[run_id].append(os.path.join(subject_dir, file_name))
    return beta_maps

def process_subject_with_searchlight(subject_dir, mask_img_path, out_dir):
    """Perform searchlight MVPA for the subject."""
    beta_maps = load_beta_maps_for_subject(subject_dir)
    if not beta_maps:
        print(f"No beta-maps found for {subject_dir}")
        return
    
    X_imgs = []
    y = []
    runs = []

    for run_id, file_paths in beta_maps.items():
        if len(file_paths) < 8:
            print(f"Skipping {run_id} (only {len(file_paths)} trials, need 8)")
            continue

        for beta_map_path in sorted(file_paths):
            X_imgs.append(beta_map_path)
            y.append(1 if 'bin1' in beta_map_path else 2)
            runs.append(run_id)
    
    if not X_imgs:
        print(f"No usable data for {subject_dir}")
        return
    
    # Stack beta maps into one 4D NIfTI
    all_beta_imgs = concat_imgs(X_imgs)
    y = np.array(y)
    runs = np.array(runs)

    print(f"Stacked 4D beta maps shape: {all_beta_imgs.shape}, labels: {y.shape}")
    
    # Load mask and resample to match beta images
    original_mask_img = nib.load(mask_img_path)
    resampled_mask = resample_img(
        original_mask_img,
        target_affine=all_beta_imgs.affine,
        target_shape=all_beta_imgs.shape[:3],
        interpolation='nearest'
    )
    print("Resampled mask to beta image grid.")

    # Optional: visualize to verify alignment
    plotting.plot_roi(resampled_mask, bg_img=all_beta_imgs.slicer[..., 0],
                      title="Resampled mask check")

    # Setup searchlight
    searchlight = SearchLight(
        mask_img=resampled_mask,
        radius=5,  
        estimator=SVC(kernel="linear"),
        cv=LeaveOneGroupOut(),
        n_jobs=-1,
        verbose=1
    )
    
    # Fit
    searchlight.fit(all_beta_imgs, y, groups=runs)
    
    # Save searchlight accuracy map
    subject_id = os.path.basename(subject_dir)
    searchlight_img = nib.Nifti1Image(searchlight.scores_, affine=all_beta_imgs.affine)
    
    accuracy_file = os.path.join(out_dir, f"{subject_id}_searchlight_accuracy.nii.gz")
    searchlight_img.to_filename(accuracy_file)
    print(f"Saved searchlight accuracy map to: {accuracy_file}")
    
    # Plotting for quick visualization
    plot_file = os.path.join(out_dir, f"{subject_id}_searchlight_accuracy.png")
    plotting.plot_stat_map(searchlight_img,
                           title=f"Searchlight accuracy - {subject_id}",
                           threshold=0.5,
                           output_file=plot_file,
                           draw_cross=False,
                           display_mode="ortho")
    print(f"Saved searchlight plot to: {plot_file}")

def main():
    root_dir = 'S:/GVuilleumier/GVuilleumier/groups/jies/Spatial_Navigation/Final_data/GLM/runwise_beta_maps_unsmoothed'
    mask_img = 'S:/GVuilleumier/GVuilleumier/groups/jies/Spatial_Navigation/Final_data/Atlas/ROIs/GM_mask.nii.gz'
    out_dir = 'S:/GVuilleumier/GVuilleumier/groups/jies/Spatial_Navigation/Final_data/GLM/runwise_beta_maps_unsmoothed/0searchlight_new/2s_bins1-5_searchlight'
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(mask_img):
        raise FileNotFoundError(f"Mask file not found: {mask_img}")
    
    subject_folders = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                       if f.startswith('sub-') and os.path.isdir(os.path.join(root_dir, f))]
    
    for subject_folder in subject_folders:
        print(f"\nProcessing {subject_folder}")
        process_subject_with_searchlight(subject_folder, mask_img, out_dir)

if __name__ == "__main__":
    main()
