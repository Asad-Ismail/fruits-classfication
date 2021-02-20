# Traing and Seriving using custom tensorflow container and pipe mode

## Instructions for building locally in your pc

1) Copy your small dataset inside sagemaker/local_test/test_dir/input/data. It should have training and validation folders for data
2) cd sagemaker
3) docker build -t k-fruits-class .
4) cd local_test && ./train_local.sh k-fruit-class:latest
5) cd local_test && ./serve_local.sh k-fruit-class:latest
6) Server will run on http://localhost:8080
7) Use scripts/sample_post.py to make a sample prediction from deployed model

## Instructions for building and deploying in Sagemaker 

1) Run Sagemaker_training_deploment.ipynb for training and deploying endpoint as well as making predictions
