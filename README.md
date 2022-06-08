# A Case Study of Quantizing Convolutional Neural Networks for Fast Disease Diagnosis on Portable Medical Devices
The current repository discusses a reseach paper published on MDPI Sensors (DOI: https://doi.org/10.3390/s22010219) and provides coding part regarding the project.

There are different types of CNN models that can be found in [models](models) directory. The training and testing are done in PyTorch Framework.

1. A file [models_training.py](models_training.py) is a python code for CNN model training with VGG16, GoogleNet, ResNet architectures. In order to train the model using a particular architecture type, type in terminal: 

```bash
$ python3 models_training.py --model help
```
Insert a model name from the suggested list of model.

2. A different set of arguments can be passed to the [models_training.py](models_training.py) script in order to change the training parameters:

```bash
$ python3 models_training.py --model help --lr 1e-4 --epoch 500 --batch 32
```
Learning rate, the number of epochs and batch size can be passed as arguments.

3. [models_testing.py](models_testing.py) tests the saved PyTorch model and prints out the accuracy numbers.


## Training process 

The training of the models (VGG, GoogleNet, ResNet) is done using a custom medical dataset of benign and malignant breast cancer ultrasound images. The dataset was acquired from KNU Chilgok Hospital and is a property of KNU Chilgok Hospital.

Dataset Details:

Cancer type|Age|Tumor Size
:---:|:---:|:---: 
Benign|44.9±8.8|9.70±5.5
Malignant|51.2±10.4|19.1±9.0

Dataset Parameters|Value
:---:|:---:
Image size|224x224
Training Images|1000
Validation Images|200
Test Images|200

###### NB: Due to the privacy rules, the dataset for medical images (benign and malignant breast cancer images) cannot be provided in this repository.

## Model Conversion from PyTorch to TensorFlow and TenorFlow TFLite

In order to apply different post-training quantization methods, the saved models after training shall be converted from PyTorch framework to TensorFlow framework through ONNX. The interesting reading can be found [here](https://towardsdatascience.com/converting-a-simple-deep-learning-model-from-pytorch-to-tensorflow-b6b353351f5d) about the conversion process.

The [converter's code](converter_pytorch2tensorflow.py) firstly converts the saved PyTorch model to the intermediate Open Neural Network Exchange framework (aforementioned ONNX). Secondly, it converts from ONNX framework to TensorFlow model. It is possible to convert the model and save it as TensorFlow graphs, however, there were certain complications with the approach.

The model type shall be passed as an argument while executing the Python script:
```bash
$ python3 converter_pytorch2tensorflow.py --model help
```

[TFLite](https://www.tensorflow.org/lite/guide) is a set of tools that enables on-device machine learning by helping developers **run their models on mobile, embedded, and edge devices.**

In order to implement post-quantization technique and run the machine learning model on mobile device, the conversion process from TensorFlow Graphs to FlatBuffers shall be completed. Therefore, in the research, [TFLite Converter](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter) was used to accomplish the task.

The Python script [converter2tflite.py](converter2tflite.py) converts saved TensorFlow model to FlatBuffer (mobile friendly framework) with quantization techniques applied. In order to execute the Python script and apply one of the quantization methods, the quantization type name shall be passed as argument as well as the model type. TFLite Converter includes 4 different options (no quantization, dynamic-range, half-precision floating-point, and full-integer quantizations) for the quantization according to the [documentation](https://www.tensorflow.org/lite/performance/model_optimization).

```bash
$ python3 converter2tflite.py --model help --q_type help
```
If no argument passed for the quantization method, then the model simply converted from TensorFlow model to TFLite model without any quantization method by default.



Acknowledgements:
KNU Chilgok Hospital for providing the breast cancer ultrasound images dataset.
