from torchvision import transforms, datasets
from torch.autograd.variable import Variable
import torch
from torch import nn
import pandas as pd
import numpy as np

class KannadaDataSet(torch.utils.data.Dataset):
    def __init__(self, images,transforms = None, IMGSIZE = 28):
        self.X = images
        self.transforms = transforms
        self.IMGSIZE = IMGSIZE
        
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X.iloc[i,:]
        data = np.array(data).astype(np.uint8).reshape(self.IMGSIZE,self.IMGSIZE,1)
        
        if self.transforms:
            data = self.transforms(data)
                
        return data

def mnist_dataloader(batch_size, num_workers):
    compose = transforms.Compose(
        [transforms.Resize(64),
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))
        ])
    out_dir = './dataset'
    data = datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers = num_workers)

def kmnist_dataloader(data_file, batch_size, num_workers):
    # Load Data
    train=pd.read_csv(data_file)
    train_images=train.drop('label',axis=1)

    compose = transforms.Compose(
        [transforms.ToPILImage(),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
    ])
    IMGSIZE = 28

    data = KannadaDataSet(train_images, compose, IMGSIZE)
    return torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers = num_workers)

def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

# Noise
def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available(): return n.cuda()
    return n

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)

def real_data_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def train_discriminator(optimizer, discriminator, loss, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    
    # 1. Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(real_data.size(0)))
    error_real.backward()

    # 2. Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0)))
    error_fake.backward()
    
    # Update weights with gradients
    optimizer.step()
    
    return error_real + error_fake, prediction_real, prediction_fake
    return (0, 0, 0)

def train_generator(optimizer, discriminator, loss, fake_data):
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error
