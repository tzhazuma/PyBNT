import os
import numpy as np
import cor;
import download;
def needdownload()->bool:
    l=np.array(os.listdir(os.curdir));
    if("elastix" in l):
        return False;
    return True;

def correct_from_path(inputpath,refpath,outpath):
    if(needdownload()):
        download.download_elastix();
    print("load success!")
    os.environ["ELASTIX_PATH"]=os.path.join(os.getcwd(),"elastix");
    os.environ["PATH"]=os.path.join(os.getcwd(),"elastix/bin")+os.pathsep+os.environ["PATH"];
    os.environ["PATH"]=os.path.join(os.getcwd(),"elastix/lib")+os.pathsep+os.environ["PATH"];
    os.system("echo $ELASTIX_PATH");
    os.system(f"cp {os.getcwd()}/elastix/bin/elastix /usr/local/bin");
    os.system(f"cp {os.getcwd()}/elastix/bin/transformix /usr/local/bin");
    os.system(f"cp {os.getcwd()}/elastix/lib/libANNlib-5.1.1.dylib /usr/local/lib");
    return cor.correct_path(inputpath,refpath,outpath);
def correct_from_image(inputimg,refimg,outpath=None):
    if(needdownload()):
        download.download_elastix();
    print("load success!")
    os.environ["ELASTIX_PATH"]=os.path.join(os.getcwd(),"elastix");
    os.environ["PATH"]=os.path.join(os.getcwd(),"elastix/bin")+os.pathsep+os.environ["PATH"];
    os.environ["PATH"]=os.path.join(os.getcwd(),"elastix/lib")+os.pathsep+os.environ["PATH"];
    os.system("echo $ELASTIX_PATH");
    os.system(f"cp {os.getcwd()}/elastix/bin/elastix /usr/local/bin");
    os.system(f"cp {os.getcwd()}/elastix/bin/transformix /usr/local/bin");
    os.system(f"cp {os.getcwd()}/elastix/lib/libANNlib-5.1.1.dylib /usr/local/lib");
    return cor.correct_img(inputimg,refimg,outpath);

