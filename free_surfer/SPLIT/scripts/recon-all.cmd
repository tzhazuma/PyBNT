\n\n#---------------------------------
# New invocation of recon-all Mon Jan 20 02:04:16 CST 2025 
\n mri_convert /Users/azuma/PycharmProjects/fmriproj/00000005.nii /Users/azuma/PycharmProjects/fmriproj/free_surfer_util/SPLIT/mri/orig/001.mgz \n
#--------------------------------------------
#@# MotionCor Mon Jan 20 02:04:56 CST 2025
\n cp /Users/azuma/PycharmProjects/fmriproj/free_surfer_util/SPLIT/mri/orig/001.mgz /Users/azuma/PycharmProjects/fmriproj/free_surfer_util/SPLIT/mri/rawavg.mgz \n
\n mri_info /Users/azuma/PycharmProjects/fmriproj/free_surfer_util/SPLIT/mri/rawavg.mgz \n
\n mri_convert /Users/azuma/PycharmProjects/fmriproj/free_surfer_util/SPLIT/mri/rawavg.mgz /Users/azuma/PycharmProjects/fmriproj/free_surfer_util/SPLIT/mri/orig.mgz --conform --cw256 \n
\n mri_add_xform_to_header -c /Users/azuma/PycharmProjects/fmriproj/free_surfer_util/SPLIT/mri/transforms/talairach.xfm /Users/azuma/PycharmProjects/fmriproj/free_surfer_util/SPLIT/mri/orig.mgz /Users/azuma/PycharmProjects/fmriproj/free_surfer_util/SPLIT/mri/orig.mgz \n
\n mri_info /Users/azuma/PycharmProjects/fmriproj/free_surfer_util/SPLIT/mri/orig.mgz \n
\n mri_synthstrip --threads 4 -i /Users/azuma/PycharmProjects/fmriproj/free_surfer_util/SPLIT/mri/orig.mgz -o /Users/azuma/PycharmProjects/fmriproj/free_surfer_util/SPLIT/mri/synthstrip.mgz \n
