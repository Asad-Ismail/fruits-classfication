import tensorflow as tf
import tensorflow.keras as K
import matplotlib.pyplot as plt
import argparse
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from datetime import datetime
import numpy as np
import pathlib
import cv2
import time
from Model import Model
from utils import show_images


parser = argparse.ArgumentParser(description='Input Arguments')
parser.add_argument("--train_dir",type=str,default="/home/asad/projs/fruits_classification/Fruit-Images-Dataset/Training")
parser.add_argument("--test_dir",type=str,default="/home/asad/projs/fruits_classification/Fruit-Images-Dataset/Test")
parser.add_argument("--epochs",type=int,default=60)
parser.add_argument("--batch_size",type=int,default=150)
parser.add_argument("--img_size",type=int,default=100)
parser.add_argument("--load_weights",type=bool,default=True)
parser.add_argument("--out_path",type=str,default="./")
args=parser.parse_args()

b_sz=args.batch_size
epochs=args.epochs
# Total train and val images of dataset
total_train_images=67692
total_val_images=22688
image_size=(args.img_size,args.img_size)
steps_per_epochs=total_train_images//b_sz
val_steps=total_val_images//b_sz
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")

#filepath="/home/asad/projs/fruits_classification/fruits-class/weights_best.hdf5"

load_weights=args.load_weights

def train(model,train_data,test_data):
    metric = 'val_accuracy'
    checkpoint= ModelCheckpoint(args.out_path+"weights_best.hdf5", monitor=metric, verbose=1,save_best_only=True, mode='max')
    tensorboard_callback = K.callbacks.TensorBoard(log_dir=logdir)
    callbacks_list = [checkpoint,tensorboard_callback]
    history = model.fit(train_data,epochs=epochs,validation_data=test_data,steps_per_epoch=steps_per_epochs,
           validation_steps=val_steps,callbacks=[callbacks_list])
    return history


def main():
    model=Model()
    if load_weights:
        local_weights="weights_best.hdf5"
        model.load_weights(local_weights)
    print(model.summary())
    model.compile(optimizer=Adam(0.001),loss="categorical_crossentropy",metrics=["accuracy"])
    train_datagen = ImageDataGenerator(rescale=1/255.0,
                                    rotation_range=20,
                                    horizontal_flip=True,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.3,
                                    zoom_range=0.5,
                                    vertical_flip=True
                                    )
    train_data=train_datagen.flow_from_directory(args.train_dir,target_size=(100,100),batch_size=b_sz)
    #show_images(train_data)
    test_datagen=ImageDataGenerator(rescale=1/255.0)
    test_data=test_datagen.flow_from_directory(args.test_dir,target_size=(100,100),batch_size=b_sz)
    #his=train(model,train_data,test_data)
    score = model.evaluate(test_data, verbose=0)
    print("Test accuracy Non quantized model:", score[1])


if __name__ =="__main__":
    main()
