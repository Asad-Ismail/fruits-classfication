# Fruits-classfication tensorflow 2.x. 
# Features
1. Train, Prediction and Evaluation scripts for classifcation model
2. Conversion from tensorflow model to tensorflow lite
3. cpp files to use tensorflow lite model with option of using CPU or GPU delegates (The script can be used in some embedded device for inference)


The repostiry contains tensorflow implementation of fruits 360 classification dataset which contains 131 coategories of fruits (https://github.com/Horea94/Fruit-Images-Dataset)
## Example images from dataset
![r_67_100](https://user-images.githubusercontent.com/22799415/108272167-2cfe3600-7172-11eb-800c-8cd8bc15a1d9.jpg)
![50_100](https://user-images.githubusercontent.com/22799415/108272203-37b8cb00-7172-11eb-8ddd-a64242345f2b.jpg)
![65_100](https://user-images.githubusercontent.com/22799415/108272246-43a48d00-7172-11eb-8806-836ea7ea1f9f.jpg)
![244_100](https://user-images.githubusercontent.com/22799415/108272278-4c955e80-7172-11eb-9f39-1bd27cda439a.jpg)

### Training
1) Download the dataset from the above link
2) To train the model python train.py --train_dir [path to train set] --test_dir [path to test set] --out_dir [path to output dir]

### Results
After 30 epochs model achieves 99.7% accuracy on test set

### Infer on image  
To test on one image python pred_one.py --weights [path to weight file] --input [path to input image]

## Conversion to tflite model
To convert to tflite model python convert_to_tflite.py --weights_dir [path to .hdf5 file] --test_dir [path to test set]

