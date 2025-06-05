import cv2
import numpy as np;
import pymeshlab;

def isint(a:str):
    try:
        q=int(a);
        return True;
    except:
        return False;
def isfloat(a:str):
    try:
        q=float(a);
        return True;
    except:
        return False;
def dis(v1,v2):
    v1=np.ndarray(v1);
    v2=np.ndarray(v2);
    dif=v2-v1;
    return dif.dot(dif);
def gaussian_dis(v1,v2,sigma=1):
    v1 = np.ndarray(v1);
    v2 = np.ndarray(v2);
    dif = v2 - v1;
    return np.exp(-dif.dot(dif)/(2*(sigma**2)));
def len(vec):
    return np.sqrt(vec.dot(vec));
def angle(vec1,vec2):
    vec1=np.array(vec1);
    vec2=np.array(vec2);
    return np.arccos(vec1.dot(vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)));
def proj(vec1,direct):
    return vec1-vec1.dot(direct)/len(direct)*(direct/len(direct));
def curvature(mesh=None,path=None):
    ms=pymeshlab.MeshSet();
    if(path):
        ms.load_new_mesh(path);
    else:
        mesh.save("temp.obj");
        ms.load_new_mesh("temp.obj");
    # 计算曲率
    ms.compute_curvature_principal_directions()

    # 获取曲率值
    mesh = ms.current_mesh()
    vertex_mean_curvature = mesh.vertex_mean_curvature()
    vertex_gaussian_curvature = mesh.vertex_gaussian_curvature()
    vertex_min_curvature = mesh.vertex_min_curvature()
    return vertex_mean_curvature,vertex_gaussian_curvature,vertex_min_curvature;
def disort(tri=tuple,direct:tuple=(0,0,0),type="square"):
    n1,n2,n3=tri[0:3];
    n1=np.array(n1);
    n2=np.array(n2);
    n3=np.array(n3);
    #cauclate the square of the three nodes triangle
    dis1=dis(n1,n2);
    dis2=dis(n1,n3);
    v1=n2-n1;
    v2=n3-n1;
    sq=0.5*dis1*dis2*angle(v1,v2);
    v11=proj(v1,direct);
    v22=proj(v2,direct);
    dis11=len(v11);
    dis22=len(v22);
    sq2=0.5*dis11*dis22*angle(v11,v22);
    if(type=="square"):
        return sq2/sq;
    elif(type=="angle"):
        return angle(v11,v22)/angle(v1,v2);
    else:
        print("not support type!");

def avrdistance(mesh1,mesh2):
    points1=mesh1.points;
    points2=mesh2.points;
    from scipy.spatial import KDTree

    # 构建 KDTree
    tree1 = KDTree(points1)
    tree2 = KDTree(points2)

    # 计算从 points1 到 points2 的最近距离
    distances1_to_2, _ = tree2.query(points1)

    # 计算从 points2 到 points1 的最近距离
    distances2_to_1, _ = tree1.query(points2)

    # 合并距离
    all_distances = np.concatenate([distances1_to_2, distances2_to_1])
    return np.mean(all_distances);
def readimage(imagep):
    if(imagep[-3:]=="nii" or imagep[-6:]=="nii.gz"):
        import nibabel as nib;
        img=nib.load(imagep);
        return img.get_fdata();
    else:
        return cv2.imread(imagep)



