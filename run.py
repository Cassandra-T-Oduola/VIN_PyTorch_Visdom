
from builtins import dict
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import torchvision.transforms as transforms

from dataset import *
from VIN import *


# Automatic swith of GPU mode if available
use_GPU = torch.cuda.is_available()

# Parsing training parameters

class Args:
    def __init__(self, args):
        [setattr(self, k,v) for (k,v) in args.items()]

args = Args({'datafile':'./data/gridworld_8x8.npz', 
        'imsize':8, 
        'lr':0.002, 
        'epochs':30, 
        'k':10, 
        'ch_i':2, 
        'ch_h':150, 
        'ch_q':10, 
        'batch_size':128})

# Instantiate a VIN model

def train_vin(datafile='./data/gridworld_8x8.npz',imsize=8,
              lr=.002, epochs=30, k=10, ch_i = 2, ch_h =150, ch_q = 10,
              batch_size=128):
    net = VIN(Args(locals()))

    if use_GPU:
         net = net.cuda()

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, eps=1e-6)

    # Dataset transformer: torchvision.transforms
    transform = None # 

    # Define Dataset
    trainset = GridworldData(args.datafile, imsize=args.imsize, train=True, transform=transform)
    

    # Create Dataloader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    for epoch in range(args.epochs): # Loop over dataset multiple times
        running_losses = []

        start_time = time.time()
        for i, data in enumerate(trainloader): # Loop over batches of data
            # Get input batch
            X, S1, S2, labels = data

            if X.size()[0] != args.batch_size: # TODO: Bug with DataLoader
                continue # Drop those data, if not enough for a batch

            # Send Tensors to GPU if available
            if use_GPU:
                X = X.cuda()
                S1 = S1.cuda()
                S2 = S2.cuda()
                labels = labels.cuda()

            # Wrap to autograd.Variable
            X, S1, S2, labels = Variable(X), Variable(S1), Variable(S2), Variable(labels)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(X, S1, S2, args)

            # Loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update params
            optimizer.step()

            # Accumulate running losses
            running_losses.append(loss.data[0]) # Take out value from 1D Tensor

        time_duration = time.time() - start_time
        # Print epoch logs
        print('[Epoch # {:3d} ({:.1f} s)] Loss: {:.4f}'.format(epoch + 1, time_duration, np.mean(running_losses)))

    print('\nFinished training. \n')

    # Testing...
    return net
def test_vin(net, args):
    testset = GridworldData(args.datafile, imsize=args.imsize, train=False, transform=None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    
    correct = 0
    total = 0
    for i, data in enumerate(testloader):
        # Get inputs
        X, S1, S2, labels = data

        if X.size()[0] != args.batch_size: # TODO: Bug with DataLoader
            continue # Drop those data, if not enough for a batch

        # Send Tensors to GPU if available
        if use_GPU:
            X = X.cuda()
            S1 = S1.cuda()
            S2 = S2.cuda()
            labels = labels.cuda()

        # Wrap to autograd.Variable
        X, S1, S2 = Variable(X), Variable(S1), Variable(S2)

        # Forward pass
        outputs = net(X, S1, S2, args)

        # Select actions with max scores(logits)
        _, predicted = torch.max(outputs, dim=1)

        # Unwrap autograd.Variable to Tensor
        predicted = predicted.data

        # Compute test accuracy
        correct += (predicted == labels).sum()
        total += labels.size()[0] # args.batch_size*num_batches, TODO: Check if DataLoader drop rest examples less than batch_size

    print('Test Accuracy (with {:d} examples): {:.2f}%'.format(total, 100*(correct/total)))
    print('\nFinished testing.\n')

    # Compute reward image and its value images for test sample

    # Randomly sample an index in test set
    idx = np.random.randint(0, len(testset))

    # Convert them to Tensor
    X = torch.from_numpy(np.array([testset.images[idx]]))
    S1 = torch.from_numpy(np.array([testset.S1[idx]]))
    S2 = torch.from_numpy(np.array([testset.S2[idx]]))

    # Wrap to autograd.Variable
    X = Variable(X.cuda())
    S1 = Variable(S1.cuda())
    S2 = Variable(S2.cuda())

    # Forward pass
    net(X, S1, S2, args, record_images=True)

    # Save grid image, reward image and value images
    imgs = np.concatenate([net.grid_image] + [net.reward_image] + net.value_images)
    np.savez_compressed('learned_rewards_values_{:d}x{:d},k={:d}'.format(args.imsize, args.imsize, imgs, args.k))

    print('\nRecorded reward and value images.\n')
    return net, imgs
