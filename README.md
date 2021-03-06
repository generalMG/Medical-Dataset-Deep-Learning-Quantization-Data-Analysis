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
A simplified scheme of model conversion from PyTorch framework to TensorFlow framework is shown in figure below:

![conversion_scheme.pdf](https://github.com/generalMG/Medical-Dataset-Deep-Learning-Quantization-Data-Analysis/files/8876774/conversion_scheme.pdf)


[TFLite](https://www.tensorflow.org/lite/guide) is a set of tools that enables on-device machine learning by helping developers **run their models on mobile, embedded, and edge devices.**

In order to implement post-quantization technique and run the machine learning model on mobile device, the conversion process from TensorFlow Graphs to FlatBuffers shall be completed. Therefore, in the research, [TFLite Converter](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter) was used to accomplish the task.

The Python script [converter2tflite.py](converter2tflite.py) converts saved TensorFlow model to FlatBuffer (mobile friendly framework) with quantization techniques applied. In order to execute the Python script and apply one of the quantization methods, the quantization type name shall be passed as argument as well as the model type. TFLite Converter includes 4 different options (no quantization, dynamic-range, half-precision floating-point, and full-integer quantizations) for the quantization according to the [documentation](https://www.tensorflow.org/lite/performance/model_optimization).

```bash
$ python3 converter2tflite.py --model help --q_type help
```

If no argument passed for the quantization method, then the model simply converted from TensorFlow model to TFLite model without any quantization method by default.

The overall conversion scheme is given in figure below:

![The overall conversion scheme](https://github.com/generalMG/Medical-Dataset-Deep-Learning-Quantization-Data-Analysis/files/8876665/converter_scheme.pdf)

## Model Accuracy Check

Model Quantization accuracy check is completed by running [tflite_checker.py](tflite_checker.py) Python Script. As stated before, certain arguments shall be passed while running the script:

```bash
$ python3 tflite_checker.py --model help --q_type help
```

Choosing a model type only leads to accuracy check of none quantized model. In order to check accuracy of models with quantization, please specify the 
`--q_type` parameter's argument.

## Results

Results of the study as follows:

### Inference Accuracy

The accuracy check was completed using customized Python script [tflite_checker.py](tflite_checker.py) on Server-Class machine (Intel Xeon CPU). The table below shows accuracy results of each CNN model with different kind of quantization methods applied.

Optimization type/Model Type|VGG16|GoogleNet|ResNet34
:---:|:---:|:---:|:---:
No opt. (FP32)|87.0%|88.5%|77.0%
Dynamic Range|87.0%|88.0%|77.0%
Half-Precision (FP16)|87.0%|88.5%|77.0%
Full-Integer (INT8)|86.5%|88.0%|76.5%

### Output Data Analysis

**A point-cloud based CNN model confidence graphs** were build using in order to analyze output data and accuracy difference of each quantization method. The graphs below are so called "point-cloud based CNN model confidence" graphs.

![prediction_graph.pdf](https://github.com/generalMG/Medical-Dataset-Deep-Learning-Quantization-Data-Analysis/files/8876833/prediction_graph.pdf)

Y-axis plots results of different quantized model outputs and compares them with original model (FP32) results on X-axis. As it can be seen from the graphs above, models with the full-integer quantization show higher confidence levels than other quantized models.

Except point-cloud based confidence graphs, the research includes **weight distribution histograms** that were acquired during CNN model inference.

**VGG model weight distribution histogram:**

![vgg_hist.pdf](https://github.com/generalMG/Medical-Dataset-Deep-Learning-Quantization-Data-Analysis/files/8876995/vgg_hist.pdf)

**GoogleNet model weight distribution histogram:**

![googlenet_hist.pdf](https://github.com/generalMG/Medical-Dataset-Deep-Learning-Quantization-Data-Analysis/files/8877009/googlenet_hist.pdf)

**ResNet34 model weight distribution histogram:**

![resnet_hist.pdf](https://github.com/generalMG/Medical-Dataset-Deep-Learning-Quantization-Data-Analysis/files/8877012/resnet_hist.pdf)

Figures above display that weight distribution between original, non-quantized model (FP32) **(right)** and the full-integer quantization applied CNN model **(left)**. Please note, that X-axis range of weight distribution histogram of the original model is between real values, whereas full-integer quantized models weigth distribution histogram range is between `[-127 and 127]` due to the **symmetric quantization of weight values**.

In order to understand internal process and explain such low accuracy degradation, the feature maps and filters were extracted during each CNN model inference. Feature maps and filters of each CNN model is given below:

**VGG model feature maps and filters figures extracted from first and last convolution layers of both original (left) and full-integer quantized (right) models**

![first_last_vgg.pdf](https://github.com/generalMG/Medical-Dataset-Deep-Learning-Quantization-Data-Analysis/files/8877184/first_last_vgg.pdf)

**GoogleNet model feature maps and filters figures extracted from first and last convolution layers of both original (left) and full-integer quantized (right) models**

![first_last_google.pdf](https://github.com/generalMG/Medical-Dataset-Deep-Learning-Quantization-Data-Analysis/files/8877203/first_last_google.pdf)

**ResNet34 model feature maps and filters figures extracted from first and last convolution layers of both original (left) and full-integer quantized (right) models**

![first_last_resnet.pdf](https://github.com/generalMG/Medical-Dataset-Deep-Learning-Quantization-Data-Analysis/files/8877208/first_last_resnet.pdf)

As it can be seen from figures above, there is no difference between feature maps and filters of both original and full-integer quantized models. That could explain insignificant accuracy difference in the inference accuracy table above.

## Inference time

Figure below displays inference time or inference latency of models on various hardware (CPU, GPU and NPU) of Snapdragon 865 Hardware Development Kit.
There is no significant difference in inference latency between original (FP32) and half-precision floating-point (FP16). However, there is speed-up in model inference latency with dynamic range and full-integer quantization. The best result can be noted when model is quantized using full-integer quantization and utilized on NPU; a hardware built for running Neural Network models. NPU performs computations only in fixed-precision, therefore, only  full-integer quantized model can be computed on NPU. As a result, NPU inference latency is improved up to 97% in comparison to other quantization methods.

![inference_time.pdf](https://github.com/generalMG/Medical-Dataset-Deep-Learning-Quantization-Data-Analysis/files/8877264/inference_time.pdf)


Acknowledgements:
KNU Chilgok Hospital for providing the breast cancer ultrasound images dataset.
