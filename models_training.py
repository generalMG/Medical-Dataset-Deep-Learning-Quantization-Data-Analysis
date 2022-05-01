import sys
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import models
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='training the main model')
parser.add_argument('--model', type=str, required=True, help='---Model type: conv, googlenet, resnet34, resnet50---')
parser.add_argument('--lr', type=float, default=1e-4, help='---Learning Rate can be customized here (default: 1e-4)---')
parser.add_argument('--epoch', type=int, default=500, help='---Number of Epochs (default: 500)---')
parser.add_argument('--batch', type=int, default=32, help='---Batch number (default: 32)---')
args, unparsed = parser.parse_known_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

min_epoch = 1
SEED = 42

model_names = ['ConvNet','GoogLeNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']

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


print('Model type: ', args.model)


def train(model, loader, optimizer):
    model.train()

    running_loss = 0
    running_metric = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        running_loss += loss.item()
        running_metric += metric_fn(pred, target)

    running_loss /= (batch_idx + 1)
    running_metric /= (batch_idx + 1)
    return running_loss, running_metric


def validate(model, loader):
    model.eval()

    running_loss = 0
    running_metric = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss_val = loss_fn(output, target)

            pred = output.argmax(dim=1)
            running_loss += loss_val.item()
            running_metric += metric_fn(pred, target)

    running_loss /= (batch_idx + 1)
    running_metric /= (batch_idx + 1)
    return running_loss, running_metric


def loss_fn(output, target):
    loss = F.cross_entropy(output, target)
    return loss


def metric_fn(output, target):
    num_data = output.size(0)
    target = target.view_as(output)
    correct = output.eq(target).sum().item()
    return correct / num_data


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor()])

train_data = ImageFolder(
    f'./dataset/train', transform=transform)
val_data = ImageFolder(
    f'./dataset/internal', transform=transform)

train_loader = DataLoader(
    train_data, batch_size=args.batch, shuffle=True)
val_loader = DataLoader(
    val_data, batch_size=args.batch, shuffle=False)

model = build_model()
optimizer_teacher = optim.Adam(model.parameters(), lr=args.lr)
scheduler_teacher = lr_scheduler.CosineAnnealingLR(optimizer_teacher, T_max=args.epoch)

best_loss = sys.maxsize
accuracy = []
val_accuracy = []


print('Started model training\n')


for epoch in range(args.epoch):
    start_time = time.time()
    loss, metric = train(model, train_loader, optimizer_teacher)
    val_loss, val_metric = validate(model, val_loader)
    scheduler_teacher.step()

    if epoch >= min_epoch and (val_loss - best_loss) < 0:
        best_loss = val_loss
        torch.save(model.state_dict(), f'./{args.model}.pth')

    print(f'Epoch: {epoch + 1}/{args.epoch} - ', end='')
    print(f'ACC: {val_metric:.4f} - ', end='')
    print(f'Loss: {val_loss:.4f} - ', end='')
    print(f'took {time.time() - start_time:.2f}s')
    accuracy.append(metric)
    val_accuracy.append(val_metric)


accuracy_np = np.asarray(accuracy)

print("Accuracy shape: ", accuracy_np.shape, '\n', "Accuracy size: ", accuracy_np.size)

plt.figure(figsize=(10, 5))
plt.title("STUDENT: Training and Validation Accuracy")
plt.plot(accuracy, 'g', label='Training Accuracy')
plt.plot(val_accuracy, 'b', label='Validation Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
