import os;
import platform;
import wget;

def download_freesurfer(password=""):
    if(platform.system()=="Win32"):
        print("freesurfer doesn't support windows, please use WSL instaed");
    elif(platform.system()=="Linux"):
        wget.download('https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/8.0.0-beta/freesurfer-linux-ubuntu22_x86_64-8.0.0-beta.tar.gz',"freesurfer.tar.gz");
        os.system("tar -xvzf freesurfer.tar.gz");
        os.system(f"echo {password} |sudo -S chmod -R 777 ./freesurfer")
        os.environ["FREESURFER_HOME"]=os.path.join(os.getcwd(),"freesurfer");
        os.system("source $FREESURFER_HOME/SetUpFreeSurfer.sh");

    elif(platform.system()=="Darwin"):
        if(platform.machine()=="x86_64"):
            wget.download("https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/8.0.0-beta/freesurfer-macOS-darwin_x86_64-8.0.0-beta.tar.gz","freesurfer.tar.gz");
        else:
            wget.download("https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/8.0.0-beta/freesurfer-macOS-darwin_arm64-8.0.0-beta.tar.gz","freesurfer.tar.gz");
        os.system("tar -xvzf freesurfer.tar.gz");
        os.system(f"echo {password} |sudo -S chmod -R 777 ./freesurfer")
        os.environ["FREESURFER_HOME"]=os.path.join(os.getcwd(),"freesurfer");
        os.system("source $FREESURFER_HOME/SetUpFreeSurfer.sh");
