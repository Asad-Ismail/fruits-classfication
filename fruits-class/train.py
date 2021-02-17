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

parser = argparse.ArgumentParser(description='Processing Input Arguments')
parser.add_argument("--train_dir",type=str,default="/home/asad/projs/fruits_classification/Fruit-Images-Dataset/Training")
parser.add_argument("--test_dir",type=str,default="/home/asad/projs/fruits_classification/Fruit-Images-Dataset/Test")
args=parser.parse_args()
b_sz=150
epochs=30
total_train_images=67692
total_val_images=22688
image_size=(100,100)
steps_per_epochs=total_train_images//b_sz
val_steps=total_val_images//b_sz
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")


def show_images(ds):
    plt.figure(figsize=(10, 10))
    #x,y=next(ds)
    for images in ds.next():
        for i in range(9):
            vis_img = images[i]
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(vis_img)
            plt.axis("off")
        plt.show()
        break

def get_all_files(root_path):
    files=[]
    for path, subdirs, files in os.walk(root_path):
        for name in files:
            files.append(os.path.join(path, name))
    return files



def post_quantization(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path("/home/asad/projs/fruits-class/tf-lite/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir/"inception_model.tflite"
    tflite_model_file.write_bytes(tflite_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    tflite_model_quant_file = tflite_models_dir/"inception_model_quant.tflite"
    tflite_model_quant_file.write_bytes(tflite_quant_model)

filepath="/home/asad/projs/fruits_classification/fruits-class/weights_best.hdf5"
metric = 'val_accuracy'
checkpoint= ModelCheckpoint(filepath, monitor=metric, verbose=1,save_best_only=True, mode='max')
tensorboard_callback = K.callbacks.TensorBoard(log_dir=logdir)
callbacks_list = [checkpoint,tensorboard_callback]
load_weights=True

def Model():
    pre_trained_model = InceptionV3(input_shape=(100,100,3),weights='imagenet',include_top=False)
    #for layer in pre_trained_model.layers:
    #    layer.trainable=False
    pre_train_out_layer=pre_trained_model.get_layer("mixed7")
    pretrain_out=pre_train_out_layer.output
    x=K.layers.GlobalAveragePooling2D()(pretrain_out)
    x=K.layers.Dense(units=1024,activation="relu")(x)
    x=K.layers.Dropout(0.2)(x)
    x=K.layers.Dense(units=512,activation="relu")(x)
    x=K.layers.Dropout(0.1)(x)
    x=K.layers.Dense(units=131,activation="softmax")(x)
    model=K.Model(inputs=pre_trained_model.input,outputs=x)
    return model
    

def train(model):
    
    rain_datagen = ImageDataGenerator(rescale=1/255.0,
                                    rotation_range=20,
                                    horizontal_flip=True,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.3,
                                    zoom_range=0.5,
                                    vertical_flip=True
                                    )
    train_data=train_datagen.flow_from_directory(args.train_dir,target_size=(100,100),batch_size=b_sz)
    show_images(train_data)
    test_datagen=ImageDataGenerator(rescale=1/255.0)
    test_data=test_datagen.flow_from_directory(args.test_dir,target_size=(100,100),batch_size=b_sz)
    history = model.fit(train_data,epochs=epochs,validation_data=test_data,steps_per_epoch=steps_per_epochs,
           validation_steps=val_steps,callbacks=[callbacks_list])
    

def predict_one(model):
    img = K.preprocessing.image.load_img("/home/asad/projs/fruits_classification/cpp-inference/test.jpg", target_size=image_size)
    img_array = K.preprocessing.image.img_to_array(img)/255.0
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions,axis=1)
    return predicted_class[0],predictions[0,predicted_class]


def pred_one_lite(interpreter_path,img_pth):
  interpreter = tf.lite.Interpreter(model_path=str(interpreter_path))
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]
  output_shape = interpreter.get_output_details()[0]["shape"]
  interpreter.allocate_tensors()
  tst_img=cv2.imread(img_pth)
  tst_img=tst_img[:,:,[2,1,0]]
  tst_img=np.expand_dims(tst_img,0)
  tst_img=tst_img.astype(np.float32)
  tst_img/=255.0
  interpreter.set_tensor(input_index, tst_img)
  t1=time.time()*1000
  interpreter.invoke()
  t2=time.time()*1000
  print(f"The time taken in python {t2-t1}")
  output = interpreter.tensor(output_index)
  digit = np.argmax(output(),axis=1)
  print(digit)


def evaluate_lite_model(interpreter_path,test_data):
  interpreter = tf.lite.Interpreter(model_path=str(interpreter_path))
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]
  output_shape = interpreter.get_output_details()[0]["shape"]
  input_shape=[150,100,100,3]
  interpreter.resize_tensor_input(input_index,input_shape)
  interpreter.resize_tensor_input(output_index,[150, 1, output_shape[1]])  
  interpreter.allocate_tensors()
  # Run predictions on every image in the "test" dataset.
  prediction = []
  gt=[]
  print(f"Total test images batches {len(test_data)}")
  for i,(test_image,labels) in enumerate(test_data):
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    if i==len(test_data)-1:
        break
    test_image = test_image.astype(np.float32)
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    #print(output())
    #print(output().shape)
    #print(len(output))
    digit = np.argmax(output(),axis=1)
    prediction.extend(digit)
    gt.extend(np.argmax(labels,1))
    print(f"Procesed {i} batches")
    #if i==20:
    #    break


  # Compare prediction results with ground truth labels to calculate accuracy.
  assert len(gt)==len(prediction), print("Length of predictions and GT are not equal")
  accurate_count = 0
  for index in range(len(prediction)):
    if prediction[index] == gt[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction)
  
  return accuracy


def main():
    model=Model()
    if load_weights:
        local_weights="weights_best.hdf5"
        model.load_weights(local_weights)
    print(model.summary())
    model.compile(optimizer=Adam(0.001),loss="categorical_crossentropy",metrics=["accuracy"])
    #post_quantization(model)
    test_datagen=ImageDataGenerator(rescale=1/255.0)
    test_data=test_datagen.flow_from_directory(args.test_dir,target_size=(100,100),batch_size=b_sz)
    #q_acc=evaluate_lite_model("/home/asad/projs/fruits-class/tf-lite/inception_model_quant.tflite",test_data)
    labels = (test_data.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    #with open("labels.txt","w") as f:
    #    for k,v in labels.items():
    #        f.writelines(v+"\n")


    pred_one_lite("/home/asad/projs/fruits-class/tf-lite/inception_model_quant.tflite","/home/asad/projs/cpp-inference/test.jpg")
    cls_pred,score=predict_one(model)
    #print(f"The model predicted the label {labels[cls_pred]} with probability {score}")
    print(f"Test accuracy Quantized model: {q_acc}")
    score = model.evaluate(test_data, verbose=0)
    #print("Test loss:", score[0])
    print("Test accuracy Non quantized model:", score[1])





if __name__ =="__main__":
    main()
