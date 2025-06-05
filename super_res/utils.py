
import os;

import DataProcess;
def split_train_test(imgpath,trainpath,testpath):
    if not os.path.exists(trainpath):
        os.makedirs(trainpath)
    if not os.path.exists(testpath):
        os.makedirs(testpath)
    DataProcess.split_train_test(imgpath,trainpath,testpath)