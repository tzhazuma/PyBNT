import pyelastix as el;
import cv2 as cv;
import nibabel as nib;
import numpy as np;
def correct_path(inputpath,refpath,outpath):
    params=el.get_default_params();
    outimg,_=el.register(inputpath,refpath,params,verbose=2);
    if(outpath[-3:]=="nil" or outpath[-6:]=="nii.gz"):
        nib.save(nib.Nifti1Image(outimg,np.eye(4)),outpath);
    else:
        cv.imwrite(outpath,outimg);
def correct_img(inputimg,refimg,outpath=None):
    params = el.get_default_params();
    outimg, _ = el.register(inputimg, refimg, params);
    if(outpath is not None):
      if (outpath[-3:] == "nil"):
        nib.save(nib.Nifti1Image(outimg, np.eye(4)), outpath);
      else:
        cv.imwrite(outpath, outimg);
    return outimg;