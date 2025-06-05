import nibabel as nib;
def parse_pial(path:str):
    nodes,edges=nib.freesurfer.io.read_geometry(path);
    return nodes,edges;



