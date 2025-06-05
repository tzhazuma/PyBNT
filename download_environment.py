import os;
import urllib3;
import requests;
import pathlib;
import math;
from collections import *;
import subprocess;
from correct_elastix.download import  *;
from free_surfer.download import *;
## this is a script to download the freesurfer, the elastix
def download_all():
    download_freesurfer();
    download_elastix();
    os.system("pip install -r requirements.txt");