# MRI Preprocessing

Motion Correction: Realigns volumes to a reference to correct for head motion.
Six-parameter rigid-body transformation (3 translations, 3 rotations).
Excessive motion (>3mm or >3 degrees) may warrant volume censoring (scrubbing).
Common tools: MCFLIRT (FSL), mcflirt, SPM realign.

Slice Timing Correction: Corrects for temporal offsets between slice
acquisitions in multi-slice sequences. Essential for event-related fMRI.
Interpolation methods: linear, spline, or sinc. Reference slice is typically
middle slice or first slice.

Spatial Normalization: Transforms individual brain images to standard
stereotaxic space (e.g., MNI152). Enables group-level statistics.
Nonlinear registration methods include DARTEL (SPM), FNIRT (FSL), ANTs SyN.

Spatial Smoothing: Applies Gaussian kernel (FWHM typically 4-8mm) to increase
SNR and meet random field theory assumptions. Smoothing kernel should be 2-3x
voxel size. May reduce spatial specificity for small structures.

Skull Stripping (Brain Extraction): Removes non-brain tissue (skull, scalp,
meninges). Key tools: BET (FSL), ROBEX, ANTs Atropos, HD-BET (deep learning).
Critical for registration and segmentation accuracy.

Intensity Normalization: Harmonizes intensity values across subjects and
sessions. Methods: min-max, z-score, histogram matching, WhiteStripe.
Important for quantitative analyses and machine learning.

Temporal Filtering: High-pass filter removes low-frequency drift (typically
0.01 Hz cutoff). Low-pass filter removes high-frequency noise. Bandpass
filtering (0.01-0.1 Hz) common for resting-state fMRI.

Nuisance Regression: Removes confounding signals from fMRI data. Common
regressors: 6 motion parameters, white matter signal, CSF signal, global
signal (controversial). Derivatives and squared terms often included
(Friston-24 model). aCompCor extracts noise components from WM/CSF masks.

ICA Denoising: ICA-AROMA identifies and removes motion-related independent
components. FIX classifies noise vs. signal components using hand-labeled
training data. Manual component classification requires expertise.

Confound Correction: Physiological noise (cardiac, respiratory) can be modeled
using RETROICOR or aCompCor. RETROICOR requires external physiological
recordings; aCompCor is data-driven.

Field Map Correction: Corrects geometric distortions from magnetic field
inhomogeneities. Requires field map acquisition (phase and magnitude images).
Important for EPI-based sequences (fMRI, DWI).
