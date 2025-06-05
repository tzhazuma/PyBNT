import numpy as np;
import cv2 as cv;
import train;
import os;
import pathlib;
import argparse;
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0";
import matplotlib.pyplot as plt
from dataset import dataset
from model import edsr,combi;
#import train;
from train import EdsrTrainer
from common import resolve
from common import resolve_single
from common import evaluate
import  tensorflow as tf
from dataset import dataset;
import utils;
depth = 4
scale = 2
filters = 64
loss = 'MAE'
def train_data(img_path:str,save_path:str,labelpath:str,ref_path=None):
    train_path=os.path.join(img_path,"train");
    val_path=os.path.join(img_path,"val");
    utils.split_train_test(img_path,train_path,val_path);
    label_train_path=os.path.join(labelpath,"train");
    label_val_path=os.path.join(labelpath,"val");
    utils.split_train_test(labelpath,label_train_path,label_val_path);
    train_ds=dataset(train_path,  label_train_path, val_path,label_val_path, ref_path,batch_size=16, repeat_count=1, random_transform=True)
    valid_ds=dataset(train_path, label_train_path,val_path, label_val_path, ref_path,batch_size=1, repeat_count=1, random_transform=False,subset='valid')
    #weights_dir = './weights/edsr/'
    #weights_file = os.path.join(weights_dir, 'weightsB{}F{}-{}.h5'.format(depth, filters, loss))
    #os.makedirs(weights_dir, exist_ok=True)
    if(ref_path is None):
        model = edsr(scale=scale, num_filters=filters, num_res_blocks=depth, res_block_scaling=0.1)
    else:
        model = combi(scale=scale, num_filters=filters, num_res_blocks=depth, res_block_scaling=0.1)
    os.makedirs(f"{save_path}/checkpoint", exist_ok=True)
    trainer = EdsrTrainer(model=model, loss=loss, checkpoint_dir=f"{save_path}/checkpoint")
    trainer.checkpoint.save(f"{save_path}/checkpoint/1.ckpt")
    with tf.device("/gpu:0"):
        trainer.train(train_ds, valid_ds.take(10), steps=15000, evaluate_every=500, save_best_only=True)

    psnrv = trainer.evaluate(valid_ds)
    print(f'PSNR = {psnrv.numpy():3f}')
    trainer.checkpoint.write(f"{save_path}/checkpoint/final.ckpt");
def run_inference(datapath:str,ckptpath:str="/ckpt",savepath="",refpath=None):
    if(refpath is None):
        model = edsr(scale=scale, num_filters=filters, num_res_blocks=depth, res_block_scaling=0.1)
    else:
        model = combi(scale=scale, num_filters=filters, num_res_blocks=depth, res_block_scaling=0.1)
    trainer=EdsrTrainer(model=model,loss=loss,checkpoint_dir=f"{ckptpath}/checkpoint");
    trainer.checkpoint.read(f"{ckptpath}/checkpoint/final.ckpt");
    if (refpath is not None):
        refl=os.listdir(refpath);
    for index,imgpath in enumerate(os.listdir(datapath)):
        img=cv.imread(os.path.join(datapath,imgpath));
        img=tf.cast(img,tf.float32)
        if(refpath is not None):
            refimg=cv.imread(os.path.join(refpath,refl[index]));
            refimg=tf.cast(refimg,tf.float32);
            img=trainer.checkpoint.model([img,refimg],training=False);
        else:
            img=trainer.checkpoint.model(img,training=False);
        cv.imwrite(f"{savepath}/output/{index}.png",img);
