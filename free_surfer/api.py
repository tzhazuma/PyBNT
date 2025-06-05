import os

import split;
import download;
def split(inputpath,inputname,outputpath):
    if("freesurfer"  not in os.listdir(os.curdir)):
        download.download_freesurfer();
    split.split(inputpath,inputname,outputpath);