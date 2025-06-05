import os;
os.environ["FREESURFER_HOME"]=os.path.join(os.getcwd(),"freesurfer");
cmd="source $FREESURFER_HOME/SetUpFreeSurfer.sh ;";
def split(inputpath,inputname,outputpath):
    global cmd;
    cmd+=f"export SUBJECTS_DIR={inputpath} ;"
    cmd+="export FS_ALLOW_DEEP=1 ;"
    cmd+=f"recon-all -parallel -i {inputname} -s  SPLIT -sd {outputpath} -cw256 -all"
    os.system(cmd)
