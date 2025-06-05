import os;
import sys;
import platform;
import wget;
import zipfile;

def download_elastix():
    cur_platform = platform.system();
    print(cur_platform)
    if (cur_platform == "Windows"):
        wget.download("https://github.com/SuperElastix/elastix/releases/download/5.1.0/elastix-5.1.0-win64.zip",
                      "elastix-5.1.0-win64.zip");
        f = zipfile.ZipFile("elastix-5.1.0-win64.zip");
        f.extractall(path="elastix");

        sys.path.append(os.path.join(os.getcwd(), "elastix"));
        os.environ["PATH"]+=os.pathsep+os.path.join(os.getcwd(), "elastix");
    elif (cur_platform == "Linux"):
        wget.download("https://github.com/SuperElastix/elastix/releases/download/5.1.0/elastix-5.1.0-linux.zip",
                      "elastix-5.1.0-linux.zip");
        f = zipfile.ZipFile("elastix-5.1.0-linux.zip");
        f.extractall(path="elastix");
        sys.path.append(os.path.join(os.getcwd(), "elastix"));
        os.environ["PATH"]+=os.pathsep+os.path.join(os.getcwd(), "elastix");
        os.system(f"cp {os.getcwd()}/elastix/bin/elastix /usr/local/bin");
        os.system(f"cp {os.getcwd()}/elastix/bin/transformix /usr/local/bin");
        os.system(f"cp {os.getcwd()}/elastix/lib/libANNlib-5.1.1.so /usr/local/lib");

    elif (cur_platform == "Darwin"):
        wget.download("https://github.com/SuperElastix/elastix/releases/download/5.1.0/elastix-5.1.0-mac.zip",
                      "elastix-5.1.0-mac.zip");
        f = zipfile.ZipFile("elastix-5.1.0-mac.zip");
        f.extractall(path="elastix");
        sys.path.append(os.path.join(os.getcwd(), "elastix"));
        os.environ["PATH"]+=os.pathsep+os.path.join(os.getcwd(), "elastix");
        os.system(f"cp {os.getcwd()}/elastix/bin/elastix /usr/local/bin");
        os.system(f"cp {os.getcwd()}/elastix/bin/transformix /usr/local/bin");
        os.system(f"cp {os.getcwd()}/elastix/lib/libANNlib-5.1.1.dylib /usr/local/lib");

    else:
        print( "Platform not support for elastix");
