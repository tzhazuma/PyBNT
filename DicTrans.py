import pydicom as dic;
import nibabel as nib;
import cv2 as cv;
import numpy as np;
import ImageProcess;
def d2nif(dpath:str,npath:str):
    ds=dic.dcmread(dpath);
    img=ds.pixel_array;
    img=ImageProcess.normalize(img);
    nib.save(nib.Nifti1Image(img,np.eye(4)),npath);
def d2img(dpath:str,ipath:str):
    ds=dic.dcmread(dpath);
    img=ds.pixel_array;
    img=ImageProcess.normalize(img);
    cv.imwrite(ipath,img);
def n2img(npath:str,ipath:str=None,normalize=False):
    img=nib.load(npath).get_fdata();
    img=np.array(img);
    if(normalize):
        img=ImageProcess.normalize(img);
    if(ipath is not None):
        cv.imwrite(ipath,img);
    else:
        return img;
def img2n(img:np.ndarray,npath:str):
    img=ImageProcess.normalize(img);
    nib.save(nib.Nifti1Image(img,np.eye(4)),npath);
def nshape(npath:str):
    return np.array(nib.load(npath).get_fdata()).shape;


