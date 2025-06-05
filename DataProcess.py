import cv2 as cv;
import os;
import ImageProcess;
import DicTrans;
def split_train_test(imgpath,trainpath,testpath,type="path",ratio=0.8,nifti=False):
    if(type=="path"):
        imgp=os.listdir(imgpath);
        trainp=imgp[:int(len(imgp)*ratio)];
        testp=imgp[int(len(imgp)*ratio):];
        for i in trainp:
            if(nifti):
                imgl=ImageProcess.To_2D(DicTrans.n2img(os.path.join(imgpath,i)),2);
                for img in imgl:
                    img=ImageProcess.normalize(img);
                    cv.imwrite(os.path.join(trainpath,i),img);
            else:
                img=cv.imread(os.path.join(imgpath,i),cv.IMREAD_GRAYSCALE);
                img=ImageProcess.normalize(img);
                cv.imwrite(os.path.join(trainpath,i),img);
        for i in testp:
            if (nifti):
                imgl = ImageProcess.To_2D(DicTrans.n2img(os.path.join(imgpath, i)), 2);
                for img in imgl:
                    img = ImageProcess.normalize(img);
                    cv.imwrite(os.path.join(trainpath, i), img);
            img=cv.imread(os.path.join(imgpath,i),cv.IMREAD_GRAYSCALE);
            img=ImageProcess.normalize(img);
            cv.imwrite(os.path.join(testpath,i),img);
    elif(type=="array"):
        trainp=imgpath[:int(len(imgpath)*ratio)];
        testp=imgpath[int(len(imgpath)*ratio):];
        for i in trainp:
            img=ImageProcess.normalize(i);
            cv.imwrite(os.path.join(trainpath,i),img);
        for i in testp:
            img=ImageProcess.normalize(i);
            cv.imwrite(os.path.join(testpath,i),img);


