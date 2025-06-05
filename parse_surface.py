import os;
import time
from curses.ascii import isdigit
from Utils import isint
import numpy as np;
import cv2 as cv;
#This document is to parse the surface file
import free_surfer.parse as parsepial;
def parse(path:str):
    if(path[-5:]==".pial"):
        return parsepial.parse_pial(path);
    f=open(path,"r");
    #jump the none-data
    nodenum=0;
    nodes=[];
    while(True):
        line=f.readline();
        if(not isint(line)):
            continue;
        else:
            nodenum=int(line);
            break;
    print(f"Parsing Nodenums:{nodenum}")
    print("Starting parsing nodes position")

    for i in range(nodenum):
        line=f.readline();
        line=line.strip().split(" ");

        if(len(line)<3):
            f.close();
            raise f"Nodes data error! At the nodes{i}";
        x,y,z=line[0:3];
        nodes.append([float(x),float(y),float(z)]);
    trinum=f.readline();
    tris=[];
    if(not isint(trinum) ):
        f.close();
        raise f"Error parsing trinum";
    for i in range(int(trinum)):
        line = f.readline();
        line = line.strip().split(" ");

        if (len(line) < 3):
            f.close();
            raise f"Nxodes data error! At the nodes{i}";
        x, y, z = line[0:3];
        tris.append([int(x),int(y),int(z)]);
    f.close();
    return (nodes,tris);


