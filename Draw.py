from parse_surface import parse;


import pyvista as pv;

def drawsurface(surfacepath,imagep=None,voxel=False,smooth=False,light="default",show=True):
    nodes,tris=parse(surfacepath);

    img=pv.make_tri_mesh(np.array(nodes),np.array(tris))

    ax=pv.Plotter();
    ax.add_mesh(img,color='white',show_edges=True);
    ax.enable_anti_aliasing("msaa",16)
    if(smooth):
        img=img.smooth();
    if(light=="default"):
        light=pv.Light(position=(-20,-20,0),focal_point=(0,0,0),intensity=0.5)
        ax.add_light(light)
    elif(light=="None"):
        pass
    elif(light=="head"):
        light=pv.Light(position=(0,0,1),focal_point=(0,0,0),intensity=0.5)
        ax.add_light(light)
    elif(light=="tail"):
        light=pv.Light(position=(0,0,-1),focal_point=(0,0,0),intensity=0.5)
        ax.add_light(light)
    elif(light=="left"):
        light=pv.Light(position=(0,1,0),focal_point=(0,0,0),intensity=0.5)
        ax.add_light(light)
    elif(light=="right"):
        light=pv.Light(position=(0,-1,0),focal_point=(0,0,0),intensity=0.5)
        ax.add_light(light)
    #if(voxel==True):
        #img2=pv.voxelize(img,density=0.5)
        #ax.add_mesh(img2)
    if(imagep is not None and voxel):
        import Utils;
        ax.add_mesh(drawvoxels(Utils.readimage(imagep)));
    if(show==True):
        ax.show()
    return ax;
from Link_Nodes import *;
def drawnode(nodesp,edgep=None,surfacep=None,imagep=None,voxel=False,type="all",colortype=("red","blue","white"),smooth=False,proj_type="nearest",show=True):
    nodes = parse_node_text(nodesp);
    edges = parse_edge_text(edgep);
    import pyvista as pv;
    ax = pv.Plotter();
    for node in nodes:
        ball = pv.Sphere(radius=node["size"], center=(node["x"], node["y"], node["z"]));
        ax.add_mesh(ball, color=colortype[0]);
    if(type!="node" and edgep is not None):
     m, n = edges.shape;
     for i in range(m):
        for j in range(n):
            if (edges[i, j] > 0):
                node1 = nodes[i];
                node2 = nodes[j];
                line = pv.Line(pointa=(node1["x"], node1["y"], node1["z"]),
                               pointb=(node2["x"], node2["y"], node2["z"]));
                ax.add_mesh(line, color=colortype[1]);
    from parse_surface import parse;
    from surface_projection import Triangle,Surfaces;
    if(surfacep is not None):
        n, t = parse(surfacep);
        img = pv.make_tri_mesh(np.array(n), np.array(t));
        img=img.extract_surface(nonlinear_subdivision=20);
        if(smooth):
            img=img.smooth();
        if(type=="all"):
         tl=[];
         for triindex in t:
            tri=Triangle((n[triindex[0]],n[triindex[1]],n[triindex[2]]),nodesindex=triindex);
            tl.append(tri);
         sf=Surfaces(tl);
         sf.proj_surface_all(ps=nodes,type=proj_type);
         val=np.zeros(img.points.shape[0]);
         for tri in sf:
            for i in range(3):
                val[tri.nodesindex[i]]+=tri.nodes[i].value;
         img.point_data["value"] = val
         img.integrate_data();
        if(type!="nande" and type!="node"):
            ax.add_mesh(img, color=colortype[2], scalars="value", show_edges=True);
    if(imagep is not None and voxel):
        import Utils;
        ax.add_mesh(drawvoxels(Utils.readimage(imagep)));
    if(show):
        ax.show();
    return ax;
def drawvoxels(voxelarray,show=False):
    import pyvista as pv;
    img=pv.UniformGrid();
    img.dimensions=voxelarray.shape;
    img.origin=(0,0,0);
    img.spacing=(1,1,1);
    img.point_arrays["voxel"]=voxelarray.flatten(order="F");
    if(show):
        img.plot(show_edges=True);
    return img;
def drawroi(img,roimask,show=False):
    import pyvista as pv;
    #let the value not in mask be 0
    img[roimask==0]=0;
    return drawvoxels(img,show);
def drawroiwithsurface(surfacepath,img,roimask,show=True):
    ax=drawsurface(surfacepath,show=False);
    roi=drawroi(img,roimask,show=False);
    ax.add_mesh(roi,opacity=0.5);
    if(show):
        ax.show();
    return ax;
