import cv2 as cv;
import numpy as np;
import skimage as ski;
def To_2D(img3d,z_axis=2):
    shape=img3d.shape;
    imgl=[];
    for i in range(shape[z_axis]):
        if(z_axis==0):
            img=img3d[i,:,:];
        elif(z_axis==1):
            img=img3d[:,i,:];
        else:
            img=img3d[:,:,i];
        imgl.append(img);
    return imgl;

def To_3D(imgl:list,z_axis=2):
    shape=imgl[0].shape;
    img3d=np.zeros((shape[0],shape[1],len(imgl)),dtype=np.uint8);
    for i in range(len(imgl)):
        if(z_axis==0):
            img3d[i,:,:]=imgl[i];
        elif(z_axis==1):
            img3d[:,i,:]=imgl[i];
        else:
            img3d[:,:,i]=imgl[i];
    return img3d;

def normalize(img:np.ndarray):
    return cv.normalize(img,None,0,255,cv.NORM_MINMAX);


def snr(img:np.ndarray,type="all"):
    if(type=="all"):
        return np.mean(img)/np.std(img);
    elif(type=="roi"):
        x,y=img.shape;
        xmin=x//2-x//10;
        xmax=x//2+x//10;
        ymin=y//2-x//10;
        ymax=y//2+x//10;
        subimg=img[xmin:xmax,ymin:ymax];
        xsurmin=0
        xsurmax=x//50;
        ysurmin=0;
        ysurmax=y//50;
        surimg=img[xsurmin:xsurmax,ysurmin:ysurmax];
        snr=np.mean(subimg)/np.std(surimg);
        return snr;
    else:
        raise ValueError("The type is not supported");

def cnr(img:np.ndarray,type="all"):
    if(type=="all"):
        dif=np.max(img)-np.min(img);
        return dif/np.std(img);
    elif(type=="roi"):
        x,y=img.shape;
        xmin=x//2-x//10;
        xmax=x//2+x//10;
        ymin=y//2-x//10;
        ymax=y//2+x//10;
        subimg=img[xmin:xmax,ymin:ymax];
        xsurmin=0
        xsurmax=x//50;
        ysurmin=0;
        ysurmax=y//50;
        surimg=img[xsurmin:xsurmax,ysurmin:ysurmax];
        dif=np.mean(subimg)-np.mean(surimg);
        cnr=abs(dif)/np.sqrt(np.std(subimg)**2+np.std(surimg)**2);
        return cnr;
    else:
        raise ValueError("The type is not supported");


def psnr(img1:np.ndarray,img2:np.ndarray):
    return ski.metrics.peak_signal_noise_ratio(img1,img2,data_range=255);

def ssim(img1:np.ndarray,img2:np.ndarray):
    return ski.metrics.structural_similarity(img1,img2,data_range=255);

def rmse(img1:np.ndarray,img2:np.ndarray):
    return ski.metrics.normalized_root_mse(img1,img2);

def transform(img,angle=0,scale=1,flip=0):
    rows,cols=img.shape;
    M=cv.getRotationMatrix2D((cols/2,rows/2),angle,scale);
    img=cv.warpAffine(img,M,(cols,rows));
    if(flip==1):
        img=cv.flip(img,1);
    return img;



