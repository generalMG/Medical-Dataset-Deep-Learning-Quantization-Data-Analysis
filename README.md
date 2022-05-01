# A Case Study of Quantizing Convolutional Neural Networks for Fast Disease Diagnosis on Portable Medical Devices
The current repository discusses a reseach paper published on MDPI Sensors (DOI: https://doi.org/10.3390/s22010219) and provides coding part regarding the project.

There are different types of CNN models that can be found in [models](models) directory. The training and testing are done in PyTorch Framework.

1. A file [models_training.py](models_training.py) is a python code for CNN model training with VGG16, GoogleNet, Resnet architectures. In order to train the model using a particular architecture type, type in terminal: 

```bash
$ python3 models_training.py --model help
```
Insert a model name from the suggested list of model.

2. Different arguments can be passed to the [models_training.py](models_training.py) script in order to change:

```bash
$ python3 models_training.py --model help --lr 1e-4 --epoch 500 --batch 32
```
Learning rate, the number of epochs and batch size can be passed as arguments.

3. [models_testing.py](models_testing.py) tests the saved PyTorch model and prints out the accuracy numbers.


## Training process 


######NB: Due to the privacy rules, the dataset for medical images (benign and malignant breast cancer images) cannot be provided in this repository. However, the dataset can be changed to different, open-source dataset.
