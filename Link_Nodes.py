import numpy as np;
from Utils import isint ,isfloat

def cauclate_node_pos(brainareanodes:list[tuple[float,float,float]|tuple[float,float,float,float]]):
    """
    Calculate the position of the nodes in the brain area
    :param brainareanodes: list of tuples, each tuple contains the x,y,z coordinate of the node
    :return: list of tuples, each tuple contains the x,y,z coordinate of the node
    """
    mean_node=np.mean(np.array(brainareanodes[:,0:3]),axis=0);
    return mean_node;

def cauclate_node_value(brainareanodes:list[tuple[float,float,float,float]],valueaxis=3):
    """
    Calculate the value of the nodes in the brain area
    :param brainareanodes: list of tuples, each tuple contains the x,y,z coordinate of the node and the value
    :return: float, the value of the node
    """
    values=brainareanodes[:,valueaxis];
    mean_value=np.mean(values);
    return mean_value;

def cauclate_node_size(brainareanodes:list[tuple[float,float,float,float]],fac=0.01):
    """
    Calculate the size of the nodes in the brain area
    :param brainareanodes:  list of tuples, each tuple contains the x,y,z coordinate of the node and the value
    :param fac: factory to the size
    :return: float, the size of the node
    """
    l=len(brainareanodes);
    return l*fac;

def parse_node_text(path:str,mode="continue"):
    """
    Parse the node text file
    :param path: the node file path
    :return: node lists
    """
    f=open(path,"r");
    lines=f.readlines();
    outnodes=[];
    for line in lines:
        data=line.strip().split("\t");
        #data.remove
        try:
            x,y,z,value,size,label=data;
            node={"x":float(x),"y":float(y),"z":float(z),"value":float(value),"size":float(size),"label":label};
            outnodes.append(node);
        except:
            print(f"Data error at {data}");
            if(mode=="continue"):
                continue;
            else:
                f.close();
                return;
    f.close();
    return outnodes;

def parse_edge_text(path:str,mode="continue"):
    """
    Parse the edge text file
    :param path: the edge file path
    :return: edge matrix
    """
    f=open(path,"r");
    lines=f.readlines();
    outedges=[];
    for line in lines:
        data=line.strip().split("\t");
        #data.remove("");
        try:
            data=list(map(lambda x: float(x),data));
            outedges.append(data);
        except:
            print(f"Data error at {data}");
            if(mode=="continue"):
                continue;
            else:
                f.close();
                return;
    outedges=np.array(outedges);
    f.close()
    return outedges;

def make_edge_from_nodes(size:tuple[int,int],outpath:str,pairnodesl:list[tuple[int,int]|tuple[int,int,float]]):
    arr=np.array([]);
    try:
        arr=np.zeros(size);
    except:
        print( f"shape error at size{size}");
        return;
    for pairnodes in pairnodesl:
        if(len(pairnodes)==2):
            try:
                arr[pairnodes[0],pairnodes[1]]=1;
            except:
                print( f"nodes position error{pairnodes}");
                return;
        else:
            try:
                arr[pairnodes[0],pairnodes[1]]=pairnodes[2];
                if(not isfloat(pairnodes[2])):
                    print( f"nodes value error at value{pairnodes[2]}");
                    return;
            except:
                print( f"nodes position error{pairnodes}");
                return;
    arr=list(arr);
    f=open(outpath,"w");
    for vec in arr:
        f.write(" ".join([str(i) for i in vec])+"\n");
    f.close();
def roi_compute(roinodes:list[tuple[int,int,int,float]],scalfac=1,mode="average"):
    if(mode=="average"):
        return np.mean(np.array(roinodes)[:,3])*scalfac;
    elif(mode=="max"):
        return np.max(np.array(roinodes)[:,3])*scalfac;
    elif(mode=="min"):
        return np.min(np.array(roinodes)[:,3])*scalfac;
    else:
        print("mode is not supported!")
from copy import deepcopy;
def node2connect(nodelist,connectmatrix):
    lin=np.sum(connectmatrix,axis=1)/np.sum(connectmatrix);
    outnodelist=[];
    for i,node in enumerate(nodelist):
        outnode=deepcopy(node);
        outnode["value"]=lin[i];
        outnodelist.append(outnode);
    return outnodelist;
def nodevaluecauclate(image,regionmask,type="average"):
    nodevoxels=image[regionmask];
    if(type=="average"):
        return np.mean(nodevoxels);
    elif(type=="max"):
        return np.max(nodevoxels);
    elif(type=="min"):
        return np.min(nodevoxels);
    elif(type=="filteraverage"):
        m=np.mean(nodevoxels);
        t=np.std(nodevoxels);
        nodevoxels=nodevoxels[nodevoxels>m-t];
        nodevoxels=nodevoxels[nodevoxels<m+t];
        return np.mean(nodevoxels);





