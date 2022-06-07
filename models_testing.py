import os
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import models

parser = argparse.ArgumentParser(description='Testing the main model. Accuracy and Sensitivity Information')
parser.add_argument('--model', type=str, required=True, help='---Model type: conv, googlenet, resnet34, resnet50---')
parser.add_argument('--lr', type=float, default=1e-4, help='---Learning Rate can be customized here (default: 1e-4)---')
parser.add_argument('--epoch', type=int, default=500, help='---Number of Epochs (default: 500)---')
parser.add_argument('--batch', type=int, default=32, help='---Batch number (default: 32)---')
parser.add_argument('--temp', type=int, default=20, help='---Temperature (default: 20)---')
args, unparsed = parser.parse_known_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

min_epoch = 1
SEED = 42


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


# Test the model.
def test(model, loader):
    model.eval()

    outputs = []
    targets = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):

            data = data.to(device)
            target = target.to(device)

            output = model(data)

            outputs = torch.cat([outputs, output], dim=0) if batch_idx else output
            targets = torch.cat([targets, target], dim=0) if batch_idx else target

    #return outputs, targets
    return models.ConfusionMatrix(outputs, targets, is_prob=True)


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)


torch.cuda.manual_seed(SEED)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Please note that the dataset CANNOT BE PROVIDED due to the privacy concerns.
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(),
    transforms.ToTensor()])

external_data = ImageFolder(
    f'./dataset/external', transform=transform)

external_loader = DataLoader(
    external_data, batch_size=args.batch, shuffle=False)


model = build_model().to(device)

model.load_state_dict(torch.load(f'./{args.model}.pth'))


# will be revised in future revisions.
#truest_out = ignite.metrics.ConfusionMatrix(2, output_transform=test(model, internal_loader, device=device(type='cpu')))
#print(truest_out)


c = test(model, external_loader)
text = 'External\n'
text += f'ACC: {c.calc_accuracy():.4f}\n'
text += f'SENS: {c.calc_sensitivity():.4f}\n'
text += f'SPEC: {c.calc_specificity():.4f}\n'
text += f'AUC: {c.calc_roc_auc():.4f}\n'
print(text)
