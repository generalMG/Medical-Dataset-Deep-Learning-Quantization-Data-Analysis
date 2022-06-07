import torch
from torch.autograd import Variable
import models
import argparse
from onnx2keras import onnx_to_keras
import keras
import onnx

parser = argparse.ArgumentParser(description='Converting the model from PyTorch to TensorFlow.')
parser.add_argument('--model', type=str, required=True, help='---Model type: conv, googlenet, resnet34, resnet50---')
args, unparsed = parser.parse_known_args()
model_names = ['ConvNet','GoogLeNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']

# Initialization of the model.
def build_model():
    if args.model == 'conv':
        return models.__dict__[model_names[0]]()
    elif args.model == 'googlenet':
        return models.__dict__[model_names[1]]()
    elif args.model == 'resnet18':
        return models.__dict__[model_names[2]]()
    elif args.model == 'resnet34':
        return models.__dict__[model_names[3]]()
    elif args.model == 'resnet50':
        return models.__dict__[model_names[4]]()
    elif args.model == 'resnet101':
        return models.__dict__[model_names[5]]()


# Load saved PyTorch model and convert it to ONNX.
model_pytorch = build_model()
model_pytorch.load_state_dict(torch.load(f'./{args.model}_base.pth', map_location=torch.device('cpu')))
dummy_input = Variable(torch.randn(1, 1, 224, 224))
torch.onnx.export(model_pytorch, dummy_input, f"./{args.model}_base.onnx")


# Load previously saved ONNX model and convert it to TensorFlow.
onnx_model = onnx.load(f"./{args.model}_base.onnx")
k_model = onnx_to_keras(onnx_model, ['input.1'])
keras.models.save_model(k_model, f"./{args.model}_base.h5", overwrite=True, include_optimizer=False)
