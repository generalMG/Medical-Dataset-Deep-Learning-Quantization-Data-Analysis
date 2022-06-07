import tensorflow as tf
import cv2
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='TensorFlow to TFLite Converter')
parser.add_argument('--model', type=str, required=True, help='---Model type: conv, googlenet, resnet34, resnet50---')
parser.add_argument('--type', type=str, default='none', help='---Choose the model either teacher or student---')
parser.add_argument('--q_type', type=str, default='none', help='---Choose quant type: none, dynamic, float, int---')
args, unparsed = parser.parse_known_args()


def representative_data_gen():
    for idx in range(0, 199):
        image = cv2.imread('./dataset/validation_dataset/check/image (' + str(idx) + ').jpg', 0)
        image = np.reshape(image, (1, 1, 224, 224))
        image = image.astype('float32')
        image = image / 255.0
        yield [image]


def choosing_model():
    model_base = tf.keras.models.load_model(f'./base_model_saved/{args.model}_base.h5', compile=False)
    return model_base


model = choosing_model()
converter = tf.lite.TFLiteConverter.from_keras_model(model)


def quantization():
    if args.q_type == 'none':
        tflite_model = converter.convert()
        return tflite_model
    elif args.q_type == 'dynamic':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        return tflite_model
    elif args.q_type == 'float':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()
        return tflite_model
    elif args.q_type == 'int':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        tflite_model = converter.convert()
        return tflite_model


def saving_model(passed_data):
    open(f'./base_model_saved/{args.model}_{args.q_type}.tflite', 'wb').write(passed_data)
    return f"Completed with success! BASE: {args.base}, TYPE: {args.type}, Quantization method: {args.q_type}."


final = saving_model(quantization())
print(final)
print("Completed with success!")
