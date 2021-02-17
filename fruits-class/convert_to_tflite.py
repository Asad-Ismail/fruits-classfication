import tensorflow as tf
import tensorflow.keras as K
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Model import Model
import pathlib
from utils import evaluate_lite_model

parser = argparse.ArgumentParser(description='Input Arguments')
parser.add_argument("--weight_file",type=str,default="/home/asad/projs/fruits_classification/fruits-class/weights_best.hdf5")
parser.add_argument("--test_dir",type=str,default="/home/asad/projs/fruits_classification/Fruit-Images-Dataset/Test")
args=parser.parse_args()

def post_quantization(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_models_dir = pathlib.Path("./tf-lite-models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir/"model.tflite"
    tflite_model_file.write_bytes(tflite_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    tflite_model_quant_file = tflite_models_dir/"model_quant.tflite"
    tflite_model_quant_file.write_bytes(tflite_quant_model)
    print("Converted model to tflite!!!")
    return str(tflite_model_file),str(tflite_model_quant_file)


if __name__=="__main__":
    model=Model()
    local_weights=args.weight_file
    model.load_weights(local_weights)
    tflite_model,tflite_quant_model=post_quantization(model)
    print("Starting Evaluation of converted models ...")
    test_datagen=ImageDataGenerator(rescale=1/255.0)
    test_data=test_datagen.flow_from_directory(args.test_dir,target_size=(100,100),batch_size=150)
    lite_model_path,quantized_model_path=post_quantization(model)
    print("."*20+"Evaluating Lite model"+"."*20)
    lite_model_acc=evaluate_lite_model(lite_model_path,test_data)
    print(f"Lite model has accuracy: {lite_model_acc}")
    print("."*20+"Evaluating Quantized Lite model"+"."*20)
    lite_model_acc=evaluate_lite_model(quantized_model_path,test_data)
    print(f"Quantized Lite model has accuracy: {lite_model_acc}")




