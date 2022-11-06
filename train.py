'''Resnet18(50,101)、Efficient3,5,7 Image classfication for cifar-10 or cifar-100 with PyTorch
Author 'Wang-junjie'.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcontrib.optim import SWA
import torchvision
import torchvision.transforms as transforms
import argparse
import os
import CCHW
from efficientnet import *

def Model(layer):
    # assert data_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert layer in [3,5,7], 'network depth should be 18, 34, 50 or 101'
    if layer == 3:
            model = efficientnet_b0()
    elif layer == 5:
        model = efficientnet_b5()
    elif layer == 7:
        model = efficientnet_b7()

    return model

# use GPU or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #the path of ouput
args = parser.parse_args()

# hyperparameter
EPOCH = 135   
pre_epoch = 0  
BATCH_SIZE = 128      
LR = 0.01        

# prepare data and pretreatment 
transform_train = transforms.Compose([

    transforms.RandomCrop(32, padding=4),  #Fill 0 around first, and then cut the image into 32 * 32 randomly
    transforms.RandomHorizontalFlip(),  #Half the probability of image flipping, half the probability of not flipping
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R.G.B The mean and variance used in the normalization of each layer
])

transform_test = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./Cifar-10', train=True, download=True, transform=transform_train) #train dataset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)   

testset = torchvision.datasets.CIFAR10(root='./Cifar-10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# Cifar-10 label
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
#net = ResNet18().to(device)

model_type=['efficientnet3','efficientnet5','efficientnet7']
model_layer=[3,5,7]
# train
if __name__ == "__main__":
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc = 85  #2 init best test accuracy
    print("Start Training!")  
    for layer in range(len(model_layer)): 
        net=Model(model_layer[layer]).to(device)
        # loss function and optimizer metod
        criterion = nn.CrossEntropyLoss()  
        #optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #mini-batch momentum-SGD，L2 Regularization
        base_opt = torch.optim.SGD(model.parameters(), lr=LR,momentum=0.9, weight_decay=5e-4) #mini-batch momentum-SGD，L2 Regularization
        opt = torchcontrib.optim.SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)
        with open(os.path.join('results',model_type[layer],"acc.txt"), "w") as f:
            with open(os.path.join('results',model_type[layer],"log.txt"), "w")as f2:
                for epoch in range(pre_epoch, EPOCH):
                    print('\nEpoch: %d' % (epoch + 1))
                    net.train()
                    sum_loss = 0.0
                    correct = 0.0
                    total = 0.0
                    for i, data in enumerate(trainloader, 0):
                        length = len(trainloader)
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        opt.zero_grad()

                        # forward + backward
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        opt.step()

                        # print loss and ACC
                        sum_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += predicted.eq(labels.data).cpu().sum()
                        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                        f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                              % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                        f2.write('\n')
                        f2.flush()

                    # after training one epoch,test the acc
                    print("Waiting Test!")
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for data in testloader:
                            net.eval()
                            images, labels = data
                            images, labels = images.to(device), labels.to(device)
                            outputs = net(images)
                            # the class with the highest score,th index of outputs.data
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum()
                        print('test acc score：%.3f%%' % (100 * correct // total))
                        acc = 100. * correct / total
                        # write the result in the acc.txt
                        f.write("EPOCH=%03d,Accuracy= %.3f" % (epoch + 1, acc))
                        f.write('\n')
                        f.flush()
                        # record the best test acc and write in the best_acc.txt
                        if acc > best_acc:
                            f3 = open(os.path.join('results',model_type[layer],"best_acc.txt"), "w")
                            f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                            f3.close()
                            best_acc = acc
                            print('Saving better model......')
                            torch.save(net.state_dict(), 'best.pth')
                print("Training Finished, TotalEPOCH=%d" % EPOCH)
        opt.swap_swa_sgd()
