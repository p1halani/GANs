# loading in and transforming data
import os
# import click as ck
import torch
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim

# visualizing data
import matplotlib.pyplot as plt
import numpy as np
import warnings
from helpers import (get_data_loader, imshow, scale, print_models, real_mse_loss,
                        fake_mse_loss, cycle_consistency_loss, save_samples, checkpoint)

from Dis_model import Discriminator
from CycleGen_model import CycleGenerator
from Model import create_model

# @ck.command()
# @ck.option(
#     '--batch-size', '-bs', default=100,
#     help='Batch size')
# @ck.option(
#     '--epochs', '-e', default=200,
#     help='Training epochs')
# @ck.option(
#     '--learning-rate', '-lr', default=0.0002,
#     help='Store Learning rate')
# @ck.option(
#     '--beta1', '-b1', default=0.5,
#     help='Beta 1 value')
# @ck.option(
#     '--beta2', '-b2', default=0.999,
#     help='Beta 2 value')
# @ck.option(
#     '--data-path', '-dp', default='./datasets/summer2winter',
#     help='Beta 2 value')
# @ck.option(
#     '--num-workers', '-nw', default=3,
#     help='Number of parallel workers')
# batch_size, epochs,learning_rate, beta1, beta2, data_path, num_workers
def main():
    # Create train and test dataloaders for images from the two domains X and Y
    # image_type = directory names for our data

    batch_size = 100
    epochs = 1
    learning_rate = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    data_path = './datasets/summer2winter'
    num_workers = 3

    dataloader_X, test_dataloader_X = get_data_loader(image_type='summer', image_dir=data_path, batch_size=batch_size)
    dataloader_Y, test_dataloader_Y = get_data_loader(image_type='winter', image_dir=data_path, batch_size=batch_size)

    # call the function to get models
    G_XtoY, G_YtoX, D_X, D_Y = create_model()

    # print all of the models
    print_models(G_XtoY, G_YtoX, D_X, D_Y)

    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, learning_rate, [beta1, beta2])
    d_x_optimizer = optim.Adam(D_X.parameters(), learning_rate, [beta1, beta2])
    d_y_optimizer = optim.Adam(D_Y.parameters(), learning_rate, [beta1, beta2])

    # train the network
    losses = training_loop(G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_x_optimizer, d_y_optimizer,
                                dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, epochs=epochs)

    fig, ax = plt.subplots(figsize=(12,8))
    losses = np.array(losses)
    print(losses)
    plt.plot(losses.T[0], label='Discriminator, X', alpha=0.5)
    plt.plot(losses.T[1], label='Discriminator, Y', alpha=0.5)
    plt.plot(losses.T[2], label='Generators', alpha=0.5)
    plt.title("Training Losses")
    plt.legend()

    import matplotlib.image as mpimg

    # helper visualization code
    def view_samples(iteration, sample_dir='samples_cyclegan'):
        
        # samples are named by iteration
        path_XtoY = os.path.join(sample_dir, 'sample-{:06d}-X-Y.png'.format(iteration))
        path_YtoX = os.path.join(sample_dir, 'sample-{:06d}-Y-X.png'.format(iteration))
        
        # read in those samples
        try: 
            x2y = mpimg.imread(path_XtoY)
            y2x = mpimg.imread(path_YtoX)
        except:
            print('Invalid number of iterations.')
        
        fig, (ax1, ax2) = plt.subplots(figsize=(18,20), nrows=2, ncols=1, sharey=True, sharex=True)
        ax1.imshow(x2y)
        ax1.set_title('X to Y')
        ax2.imshow(y2x)
        ax2.set_title('Y to X')

    # view samples at iteration 4000
    view_samples(1, 'samples_cyclegan')

def training_loop(G_XtoY, G_YtoX, D_X, D_Y, g_optimizer, d_x_optimizer, d_y_optimizer, 
                    dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, epochs=1000):
    
    print_every=1
    
    # keep track of losses over time
    losses = []
    
    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = test_iter_X.next()[0]
    fixed_Y = test_iter_Y.next()[0]
    fixed_X = scale(fixed_X) # make sure to scale to a range -1 to 1
    fixed_Y = scale(fixed_Y)

    # batches per epoch
    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)
    batches_per_epoch = min(len(iter_X), len(iter_Y))

    for epoch in range(1, epochs+1):

        # Reset iterators for each epoch
        if epoch % batches_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X, _ = iter_X.next()
        images_X = scale(images_X) # make sure to scale to a range -1 to 1

        images_Y, _ = iter_Y.next()
        images_Y = scale(images_Y)
        
        # move images to GPU if available (otherwise stay on CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        images_X = images_X.to(device)
        images_Y = images_Y.to(device)


        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        ##   First: D_X, real and fake loss components   ##

        # Train with real images
        d_x_optimizer.zero_grad()

        # 1. Compute the discriminator losses on real images
        out_x = D_X(images_X)
        D_X_real_loss = real_mse_loss(out_x)
        
        # Train with fake images
        
        # 2. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX(images_Y)

        # 3. Compute the fake loss for D_X
        out_x = D_X(fake_X)
        D_X_fake_loss = fake_mse_loss(out_x)
        

        # 4. Compute the total loss and perform backprop
        d_x_loss = D_X_real_loss + D_X_fake_loss
        d_x_loss.backward()
        d_x_optimizer.step()

        
        ##   Second: D_Y, real and fake loss components   ##
        
        # Train with real images
        d_y_optimizer.zero_grad()
        
        # 1. Compute the discriminator losses on real images
        out_y = D_Y(images_Y)
        D_Y_real_loss = real_mse_loss(out_y)
        
        # Train with fake images

        # 2. Generate fake images that look like domain Y based on real images in domain X
        fake_Y = G_XtoY(images_X)

        # 3. Compute the fake loss for D_Y
        out_y = D_Y(fake_Y)
        D_Y_fake_loss = fake_mse_loss(out_y)

        # 4. Compute the total loss and perform backprop
        d_y_loss = D_Y_real_loss + D_Y_fake_loss
        d_y_loss.backward()
        d_y_optimizer.step()


        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================

        ##    First: generate fake X images and reconstructed Y images    ##
        g_optimizer.zero_grad()

        # 1. Generate fake images that look like domain X based on real images in domain Y
        fake_X = G_YtoX(images_Y)

        # 2. Compute the generator loss based on domain X
        out_x = D_X(fake_X)
        g_YtoX_loss = real_mse_loss(out_x)

        # 3. Create a reconstructed y
        # 4. Compute the cycle consistency loss (the reconstruction loss)
        reconstructed_Y = G_XtoY(fake_X)
        reconstructed_y_loss = cycle_consistency_loss(images_Y, reconstructed_Y, lambda_weight=10)


        ##    Second: generate fake Y images and reconstructed X images    ##

        # 1. Generate fake images that look like domain Y based on real images in domain X
        fake_Y = G_XtoY(images_X)

        # 2. Compute the generator loss based on domain Y
        out_y = D_Y(fake_Y)
        g_XtoY_loss = real_mse_loss(out_y)

        # 3. Create a reconstructed x
        # 4. Compute the cycle consistency loss (the reconstruction loss)
        reconstructed_X = G_YtoX(fake_Y)
        reconstructed_x_loss = cycle_consistency_loss(images_X, reconstructed_X, lambda_weight=10)

        # 5. Add up all generator and reconstructed losses and perform backprop
        g_total_loss = g_YtoX_loss + g_XtoY_loss + reconstructed_y_loss + reconstructed_x_loss
        g_total_loss.backward()
        g_optimizer.step()


        # Print the log info
        if epoch % print_every == 0:
            # append real and fake discriminator losses and the generator loss
            losses.append((d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
            print('Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}'.format(
                    epoch, epochs, d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))

            
        sample_every=1
        # Save the generated samples
        if epoch % sample_every == 0:
            G_YtoX.eval() # set generators to eval mode for sample generation
            G_XtoY.eval()
            save_samples(epoch, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=16, sample_dir='samples_cyclegan')
            G_YtoX.train()
            G_XtoY.train()

        # uncomment these lines, if you want to save your model
        checkpoint_every=1000
        # Save the model parameters
        if epoch % checkpoint_every == 0:
            checkpoint(epoch, G_XtoY, G_YtoX, D_X, D_Y)

    return losses


if __name__ == '__main__':
    main()