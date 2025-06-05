from nilearn.input_data import NiftiSpheresMasker
from nilearn.connectome import ConnectivityMeasure
import numpy as np
from nilearn.maskers import NiftiMasker
import nibabel as nib;
import os;
import cv2 as cv;
import ImageProcess;
def getcor_matrix(pathimg=None):
    if(isinstance(pathimg,str)):
        print("read from path");
        if(pathimg[-3:0]=="nil" or pathimg[-6:0]=="nii.gz"):
            print("read from 4D nil");
            video=nib.load(pathimg).get_fdata();
            if(len(video.shape)!=4):
                print("shape error, it's not a 3D image video");
                return;
            fmri_filename=pathimg;
            ratio=video.shape[0]//50;
            seed_coords = [(0, -52, 18)]
            seed_masker = NiftiSpheresMasker(
                seed_coords, radius=8*ratio,
                detrend=True, standardize=True)
            seed_time_series = seed_masker.fit_transform(fmri_filename)
            brain_masker = NiftiMasker(
                smoothing_fwhm=6*ratio,
                detrend=True, standardize=True)
            brain_time_series = brain_masker.fit_transform(fmri_filename)
            connectome_measure = ConnectivityMeasure(kind='correlation')
            correlation_matrix = connectome_measure.fit_transform([brain_time_series])[0]
            print("Cor_matrix_shape", correlation_matrix.shape)
            print(correlation_matrix)
            return correlation_matrix;
        else:
            print("read from image dir");
            # read images from image dir and combine into one nil image
            imgpl=os.listdir(pathimg);
            imgl=[];
            for imgp in imgpl:
                if(imgp [-3:0]=="nil" or imgp [-6:0]=="nii.gz"):
                    img=nib.load(os.path.join(pathimg,imgp)).get_fdata();
                else:
                    img=cv.imread(os.path.join(pathimg,imgp));
                img=ImageProcess.normalize(img);
                imgl.append(img);
            img4d=np.array(imgl);
            ratio=img4d.shape[0]//50;
            img4dnib=nib.Nifti1Image(img4d,np.eye(4));
            nib.save(img4dnib,os.path.join(pathimg,"temp.nii"));
            fmri_filename = os.path.join(pathimg,"temp.nii");
            seed_coords = [(0, -52, 18)]
            seed_masker = NiftiSpheresMasker(
                seed_coords, radius=8*ratio,
                detrend=True, standardize=True)
            seed_time_series = seed_masker.fit_transform(fmri_filename)
            brain_masker = NiftiMasker(
                smoothing_fwhm=6*ratio,
                detrend=True, standardize=True)
            brain_time_series = brain_masker.fit_transform(fmri_filename)
            connectome_measure = ConnectivityMeasure(kind='correlation')
            correlation_matrix = connectome_measure.fit_transform([brain_time_series])[0]
            print("Cor_matrix_shape", correlation_matrix.shape)
            print(correlation_matrix)
            return correlation_matrix;
    else:
        if(isinstance(pathimg,nib.Nifti1Image) or isinstance(pathimg,nib.Nifti2Image)):
            nib.save(pathimg,"temp.nii");
            fmri_filename="temp.nii";
            seed_coords = [(0, -52, 18)]
            ratio=pathimg.get_fdata().shape[0]//50;
            seed_masker = NiftiSpheresMasker(
                seed_coords, radius=8*ratio,
                detrend=True, standardize=True)
            seed_time_series = seed_masker.fit_transform(fmri_filename)
            brain_masker = NiftiMasker(
                smoothing_fwhm=6*ratio,
                detrend=True, standardize=True)
            brain_time_series = brain_masker.fit_transform(fmri_filename)
            connectome_measure = ConnectivityMeasure(kind='correlation')
            correlation_matrix = connectome_measure.fit_transform([brain_time_series])[0]
            print("Cor_matrix_shape", correlation_matrix.shape)
            print(correlation_matrix)
            return correlation_matrix;
        else:
            img4d=np.array(pathimg);
            ratio=img4d.shape[0]//50;
            img4dnib=nib.Nifti1Image(img4d,np.eye(4));
            nib.save(img4dnib,"temp.nii");
            fmri_filename = "temp.nii";
            seed_coords = [(0, -52, 18)]
            seed_masker = NiftiSpheresMasker(
                seed_coords, radius=8*ratio,
                detrend=True, standardize=True)
            seed_time_series = seed_masker.fit_transform(fmri_filename)
            brain_masker = NiftiMasker(
                smoothing_fwhm=6*ratio,
                detrend=True, standardize=True)
            brain_time_series = brain_masker.fit_transform(fmri_filename)
            connectome_measure = ConnectivityMeasure(kind='correlation')
            correlation_matrix = connectome_measure.fit_transform([brain_time_series])[0]
            print("Cor_matrix_shape", correlation_matrix.shape)
            print(correlation_matrix)
            return correlation_matrix;

