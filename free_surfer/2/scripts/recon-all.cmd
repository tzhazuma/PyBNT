\n\n#---------------------------------
# New invocation of recon-all Mon Jan 20 01:59:05 CST 2025 
\n mri_convert /Users/azuma/PycharmProjects/fmriproj/00000005.nii /Users/azuma/PycharmProjects/fmriproj/2/mri/orig/001.mgz \n
#--------------------------------------------
#@# MotionCor Mon Jan 20 01:59:11 CST 2025
\n cp /Users/azuma/PycharmProjects/fmriproj/2/mri/orig/001.mgz /Users/azuma/PycharmProjects/fmriproj/2/mri/rawavg.mgz \n
\n mri_info /Users/azuma/PycharmProjects/fmriproj/2/mri/rawavg.mgz \n
\n mri_convert /Users/azuma/PycharmProjects/fmriproj/2/mri/rawavg.mgz /Users/azuma/PycharmProjects/fmriproj/2/mri/orig.mgz --conform \n
\n mri_add_xform_to_header -c /Users/azuma/PycharmProjects/fmriproj/2/mri/transforms/talairach.xfm /Users/azuma/PycharmProjects/fmriproj/2/mri/orig.mgz /Users/azuma/PycharmProjects/fmriproj/2/mri/orig.mgz \n
\n mri_info /Users/azuma/PycharmProjects/fmriproj/2/mri/orig.mgz \n
