import cv2 as cv;
def vid2img(vpath:str,ipath:str):
    cap=cv.VideoCapture(vpath);
    i=0;
    while(cap.isOpened()):
        ret,frame=cap.read();
        if(ret==False):
            break;
        cv.imwrite(ipath+f"/{i}.jpg",frame);
        i+=1;
    cap.release();
import os;
def imgl2vid(ipath:str,vpath:str,fps=30):
    imgl=os.listdir(ipath);
    imgl=os.path.join(ipath,imgl);
    img=cv.imread(imgl[0]);
    h,w,c=img.shape;
    fourcc=cv.VideoWriter.fourcc(*"mp4v");
    out=cv.VideoWriter(vpath,fourcc,fps,(w,h));
    for i in imgl:
        img=cv.imread(i);
        out.write(img);
    out.release();
