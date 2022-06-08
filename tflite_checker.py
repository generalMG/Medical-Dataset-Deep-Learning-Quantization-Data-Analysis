import time
import cv2
import tensorflow as tf
import numpy as np
import faulthandler
import argparse
faulthandler.enable()

parser = argparse.ArgumentParser(description='TFLite Model Check-up')
parser.add_argument('--model', type=str, required=True, help='---Model type: conv, googlenet, resnet34, resnet50---')
parser.add_argument('--q_type', type=str, default='none', help='---Choose quant type: none, dynamic, float, int---')
args, unparsed = parser.parse_known_args()

physical_devices = tf.config.list_physical_devices('CPU')
def choosing_model():
    interpreter = tf.lite.Interpreter(model_path=f'./base_model_saved/{args.model}_{args.q_type}.tflite')
    return interpreter

def normalization(image):
    if args.q_type == 'int':
        image = np.reshape(image, (1, 1, 224, 224))
        image = np.array(image, dtype=np.uint8)
        return image
    else:
        image = np.reshape(image, (1, 1, 224, 224))
        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        return image


interpreter = choosing_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()
print(input_details)
print(output_details)
lbl0 = [0] * 100
lbl1 = [1] * 100
lbl = lbl0 + lbl1
labels = ['ben', 'mal']
amount = 200

def eval_mod(interpreter):
    accurate_count = 0
    sum_time = 0
    predictions = []
    start_time = time.time()
    for pic in range(0, 200):
        start_time1 = time.time()
        image = cv2.imread('./dataset/validation_dataset/check/image (' + str(pic) + ').jpg', 0)
        image = normalization(image)
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        digits = np.argmax(output)
        predictions.append(digits)
        print('Of pic #: ', pic,'   ', 'Output: ', digits)
        if digits == lbl[pic]:
            accurate_count = accurate_count + 1
        else:
            pass
        each_time = time.time() - start_time1
        sum_time = sum_time + each_time
    average_for_each = sum_time / amount
    accuracy = accurate_count * 1.0 / amount
    print("Model accuracy: ", accuracy)
    print("Average time for each iteration: ", average_for_each)
    print("Time for all images: ", time.time() - start_time)
    print("Accurately predicted: ", accurate_count)


print('\n')
print("NOTICE: Model is being executed on CPU")
print(f'Model: {args.model}, Q_type: {args.q_type}\n')
print('\n')
predictions = eval_mod(interpreter)
print(f'Model: {args.model}, Q_type: {args.q_type}\n')
print("Completed with success!")