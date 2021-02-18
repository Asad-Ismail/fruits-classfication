import tensorflow as tf
import tensorflow.keras as K
import argparse
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import numpy as np
import time
from utils import predict_one,pred_one_lite
from Model import Model


parser=argparse.ArgumentParser(description="Input parser")
parser.add_argument("--weights",type=str,default="weights_best.hdf5")
parser.add_argument("--input",type=str,default="test.jpg")
args=parser.parse_args()

if __name__=="__main__":
    with open("labels.txt","r") as f:
        labels=[name.strip() for name in f.readlines()]
    model=Model()
    model.load_weights(args.weights)
    pred_one_lite("/home/asad/projs/fruits_classification/fruits-class/tf-lite-models/model_quant.tflite",args.input)
    #cls_pred,score=predict_one(model,args.input)
    #print(f"The model predicted the label {labels[cls_pred]} with probability {score}")