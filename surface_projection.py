from math import *;
import numpy as np;

import Utils

type pos=tuple[int|float,int|float,int|float]
class Triangle(object):
    def __init__(self,nodes:tuple[pos,pos,pos],brain_re:str=None,nodesindex=None):
        self.nodes=np.array(list(nodes));
        self.region=brain_re;
        self.value=0;
        self.nodesindex=nodesindex;
    def caculate_mid(self):
        return np.mean(self.nodes);
    def clear(self):
        self.value=0;
    def clear_region(self):
        self.region="";
    def proj_value(self,value):
        self.value+=value;
    def set_region(self,region):
        self.region=region;
    def cauclate_h_line(self):
        n1=self.nodes[0];
        n2=self.nodes[1];
        n3=self.nodes[2];
        l1=n2-n1;
        l2=n3-n1;
        x1,y1,z1=l1[0:3]
        x2,y2,z2=l2[0:3]
        k=z1*y2-z2*y1;
        if(k!=0):
            x3=1;
            z3=(-x1*y2+x2*y1)/k;
            if(y1!=0):
                y3=(-x1-z1*z3)/y1;
                return(x3,y3,z3);
            else:
                y3=(-x2-z2*z3)/y2;
                return (x3,y3,z3);
        else:
            if(x1*y2!=x2*y1):
                x3=0;
                if(not(z1==0 and y1==0)):
                    return (x3,z1,-y1);
                else:
                    return(x3,z2,-y2);
            else:
                if(not (x1==0 and y1==0)):
                    return(y1,-x1,0);
                else:
                    z3=0;
                    return(y2,-x2,0);

    def proj_pos(self,p:pos|np.ndarray):
        proj_line = np.array(self.cauclate_h_line());
        n1=self.nodes[0];
        l=np.array(p)-n1;
        s=-l.dot(proj_line);
        q=proj_line.dot(proj_line);
        t=s/q;
        po=np.array(p)+t*proj_line;
        return po;
    def veclen(self,v:np.ndarray):
        return np.sqrt(v.dot(v));
    def vecang(self,v1:np.ndarray,v2:np.ndarray):
        s=v1.dot(v2)/(self.veclen(v1)*self.veclen(v2));
        return acos(s);
    def pos_in(self,p:pos):
        n1,n2,n3=self.nodes[0:3];
        p=np.array(p);
        l1=n2-n1;l2=n3-n1;d=p-n1;
        s=self.vecang(l1,l2);
        s1=self.vecang(l1,d);
        s2=self.vecang(l2,d);
        if(s!=s1+s2):
            return False;
        else:
            l1=n3-n2;l2=n1-n2;d=p-n2;
            s=self.vecang(l1,l2);
            s1=self.vecang(l1,d);
            s2=self.vecang(l2,d);
            if(s!=s1+s2):
                return False;
            else:
                return True;
    def need_proj(self,p:dict,regionbound=True):
        if(p["label"]!=self.region and regionbound==True):
            return False;
        proj_p=self.proj_pos((p["x"],p["y"],p["z"]));
        return self.pos_in(proj_p);
    def dis(self,p1:pos|np.ndarray,p2:pos|np.ndarray):
        p1=np.array(p1);
        p2=np.array(p2);
        dif=p2-p1;
        return np.sqrt(dif.dot(dif));
    def pos_dis(self,p:dict):
        proj_p=np.array(self.proj_pos((p["x"],p["y"],p["z"])));
        ori_p=np.array((p["x"],p["y"],p["z"]));
        return self.dis(proj_p,ori_p);
    def mean_node(self):
        return np.mean(np.array(self.nodes),axis=0);
class Surfaces(object):
    def __init__(self,triangles:list[Triangle],regionnodes:dict=None):
        self.tri=triangles;
        if(regionnodes is not None):
         for triangle in self.tri:
            triangle.set_region(regionnodes[triangle.mean_node()]);
    def proj_single(self,p:dict,type="All",neard=10)->None:
        if(type=="all"):
            for triangle in self.tri:
                if triangle.need_proj(p):
                    triangle.proj_value(p["value"]);
        elif(type=="nearest"):
            mindis=9999999.0;
            minidx=0;
            for index,triangle in enumerate(self.tri):
                if triangle.need_proj(p):
                    d=triangle.pos_dis(p);
                    if(d<mindis):
                        mindis=d;
                        minidx=index;
            self.tri[minidx].proj_value(p["value"]);
        elif(type=="surround"):
            for triangle in self.tri:
                d=triangle.pos_dis(p);
                if(d<=neard and triangle.need_proj(p)):
                    triangle.proj_value(p["value"]);
        else:
            raise"Projection type error!"
    def proj_all(self,ps:list[dict],type="All"):
        for p in ps:
            self.proj_single(p,type);
        return ;
    def proj_surface_all(self,ps:list[dict],type="nearest",disrange=10,regionbound=False):
        for triangle in self.tri:
            triangle.clear();
            mean_node=triangle.mean_node();
            if(type=="nearest"):
                mindis=9999999.0;
                minidx=0;
                for index,p in enumerate(ps):
                    d=triangle.dis(mean_node,(p["x"],p["y"],p["z"]));
                    if(d<mindis and triangle.need_proj(p,regionbound)):
                        mindis=d;
                        minidx=index;
                triangle.proj_value(ps[minidx]["value"]);
            elif(type=="minium"):
                mindis = 9999999.0;
                for index, p in enumerate(ps):
                    d = triangle.dis(mean_node, (p["x"], p["y"], p["z"]));
                    if (d < mindis and triangle.need_proj(p,regionbound)):
                        mindis = d;
                disr=mindis+disrange;
                neighbors=[];
                for p in ps:
                    d = triangle.dis(mean_node, (p["x"], p["y"], p["z"]));
                    if(d<=disr and triangle.need_proj(p,regionbound)):
                        neighbors.append(p["value"]);
                triangle.proj_value(min(neighbors));
            elif(type=="maxium"):
                mindis = 9999999.0;
                for index, p in enumerate(ps):
                    d = triangle.dis(mean_node, (p["x"], p["y"], p["z"]));
                    if (d < mindis and triangle.need_proj(p,regionbound)):
                        mindis = d;
                disr = mindis + disrange;
                neighbors = [];
                for p in ps:
                    d = triangle.dis(mean_node, (p["x"], p["y"], p["z"]));
                    if (d <= disr and triangle.need_proj(p,regionbound)):
                        neighbors.append(p["value"]);
                triangle.proj_value(max(neighbors));
            elif(type=="average"):
                mindis = 9999999.0;
                for index, p in enumerate(ps):
                    d = triangle.dis(mean_node, (p["x"], p["y"], p["z"]));
                    if (d < mindis and triangle.need_proj(p,regionbound)):
                        mindis = d;
                disr = mindis + disrange;
                neighbors = [];
                for p in ps:
                    d = triangle.dis(mean_node, (p["x"], p["y"], p["z"]));
                    if (d <= disr and triangle.need_proj(p,regionbound)):
                        neighbors.append(p["value"]);
                triangle.proj_value(np.mean(neighbors));
            elif(type=="gaussian_nearest"):
                mindis = 9999999.0;
                minidx = 0;
                for index, p in enumerate(ps):
                    d = Utils.gaussian_dis(mean_node, (p["x"], p["y"], p["z"]));
                    if (d < mindis and triangle.need_proj(p, regionbound)):
                        mindis = d;
                        minidx = index;
                triangle.proj_value(ps[minidx]["value"]);
            elif(type=="gaussian_minium"):
                mindis = 9999999.0;
                for index, p in enumerate(ps):
                    d = Utils.gaussian_dis(mean_node, (p["x"], p["y"], p["z"]));
                    if (d < mindis and triangle.need_proj(p, regionbound)):
                        mindis = d;
                disr = mindis + disrange;
                neighbors = [];
                for p in ps:
                    d = Utils.gaussian_dis(mean_node, (p["x"], p["y"], p["z"]));
                    if (d <= disr and triangle.need_proj(p, regionbound)):
                        neighbors.append(p["value"]);
                triangle.proj_value(min(neighbors));
            elif(type=="gaussian_maxium"):
                mindis = 9999999.0;
                for index, p in enumerate(ps):
                    d = Utils.gaussian_dis(mean_node, (p["x"], p["y"], p["z"]));
                    if (d < mindis and triangle.need_proj(p, regionbound)):
                        mindis = d;
                disr = mindis + disrange;
                neighbors = [];
                for p in ps:
                    d = Utils.gaussian_dis(mean_node, (p["x"], p["y"], p["z"]));
                    if (d <= disr and triangle.need_proj(p, regionbound)):
                        neighbors.append(p["value"]);
                triangle.proj_value(max(neighbors));
            elif(type=="gaussian_average"):
                mindis = 9999999.0;
                for index, p in enumerate(ps):
                    d = Utils.gaussian_dis(mean_node, (p["x"], p["y"], p["z"]));
                    if (d < mindis and triangle.need_proj(p, regionbound)):
                        mindis = d;
                disr = mindis + disrange;
                neighbors = [];
                for p in ps:
                    d = Utils.gaussian_dis(mean_node, (p["x"], p["y"], p["z"]));
                    if (d <= disr and triangle.need_proj(p, regionbound)):
                        neighbors.append(p["value"]);
                triangle.proj_value(np.mean(neighbors));
            elif(type=="interpolate"):
                #hline=triangle.cauclate_h_line();
                mindis = 9999999.0;
                for index, p in enumerate(ps):
                    d = Utils.gaussian_dis(mean_node, (p["x"], p["y"], p["z"]));
                    if (d < mindis and triangle.need_proj(p, regionbound)):
                        mindis = d;
                disr = mindis + disrange;
                neighbors = [];
                for p in ps:
                    d=Utils.dis(mean_node,(p["x"],p["y"],p["z"]));
                    if(d<=disr and triangle.need_proj(p,regionbound)):
                        neighbors.append((p["value"],np.array([p["x"],p["y"],p["z"]])));
                out=np.zeros(3);
                for value,pos in neighbors:
                    proj_pos=triangle.proj_pos(pos);
                    l=proj_pos-mean_node;
                    out+=value*l;
                triangle.proj_value(np.sum(out));




