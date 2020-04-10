import torch
from torch import nn, optim
from torch.autograd.variable import Variable

import pandas as pd
import numpy as np
import click as ck
import matplotlib.pyplot as plt
from utils import Logger
from Gen_model import GenerativeNet
from Dis_model import DiscriminativeNet
from helpers import (kmnist_dataloader,noise, init_weights, train_discriminator, train_generator)

@ck.command()
@ck.option(
    '--batch-size', '-bs', default=100,
    help='Batch size')
@ck.option(
    '--epochs', '-e', default=200,
    help='Training epochs')
@ck.option(
    '--data-file', '-df', default='kmnist_train.csv',
    help='path to Data')
@ck.option(
    '--num-test-samples', '-ts', default=16,
    help='Number of test samples')
@ck.option(
    '--num-workers', '-nw', default=3,
    help='Number of parallel workers')
@ck.option(
    '--learning-rate', '-lr', default=0.0002,
    help='Store Learning rate')
@ck.option(
    '--beta1', '-b1', default=0.5,
    help='Beta 1 value')
@ck.option(
    '--beta2', '-b2', default=0.999,
    help='Beta 2 value')


def main(batch_size, epochs, data_file, num_test_samples, num_workers, learning_rate, beta1, beta2):

    # Create loader with data, so that we can iterate over it
    data_loader = kmnist_dataloader(data_file, batch_size, num_workers)

    # Num batches
    num_batches = len(data_loader)

    generator, discriminator = create_model()

    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))
    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))

    # Loss function
    loss = nn.BCELoss()
    
    test_noise = noise(num_test_samples)

    logger = Logger(model_name='DCGAN', data_name='KMNIST')
    nn_output = []

    for epoch in range(epochs):
        d_loss, g_loss = 0, 0
        cnt = 0
        for n_batch, (real_batch) in enumerate(data_loader):
            
            # 1. Train Discriminator

            # ============================================
            #            TRAIN THE DISCRIMINATORS
            # ============================================
        
            real_data = Variable(real_batch)
            if torch.cuda.is_available(): real_data = real_data.cuda()
            # Generate fake data
            fake_data = generator(noise(real_data.size(0))).detach()
            # Train D
            d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, discriminator, loss, 
                                                                    real_data, fake_data)

            # 2. Train Generator

            # ============================================
            #            TRAIN THE GENERATORS
            # ============================================

            # Generate fake data
            fake_data = generator(noise(real_batch.size(0)))
            # Train G
            g_error = train_generator(g_optimizer, discriminator, loss, fake_data)
            # Log error
            logger.log(d_error, g_error, epoch, n_batch, num_batches)
            
            # Display Progress
            if (n_batch) % 100 == 0:
                # display.clear_output(True)
                # Display Images
                test_images = generator(test_noise).data.cpu()
                logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches);
                # Display status Logs
                (epoch, temp_d_loss, temp_g_loss) = logger.display_status(
                                            epoch, epochs, n_batch, num_batches,
                                            d_error, g_error, d_pred_real, d_pred_fake
                                        )
                d_loss += temp_d_loss
                g_loss += temp_g_loss
                cnt += 1
        
            # Model Checkpoints
            logger.save_models(generator, discriminator, epoch)

        d_loss, g_loss = d_loss/cnt, g_loss/cnt
        nn_output.append([epoch, d_loss, g_loss])

    pd_results = pd.DataFrame(nn_output, columns = ['epoch','d_loss','g_loss'])
    print(pd_results)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))
    axes.plot(pd_results['epoch'],pd_results['d_loss'], label='discriminative loss')
    axes.plot(pd_results['epoch'],pd_results['g_loss'], label='generative loss')
    # axes[0].plot(pd_results['epoch'],pd_results['test_loss'], label='test_loss')

    axes.legend()

    # axes[1].plot(pd_results['epoch'],pd_results['valid_acc'], label='validation_acc')
    # axes[1].plot(pd_results['epoch'],pd_results['train_acc'], label='train_acc')
    # # axes[1].plot(pd_results['epoch'],pd_results['test_acc'], label='test_acc')
    # axes[1].legend()

def create_model():
    # Create Network instances and init weights
    generator = GenerativeNet()
    generator.apply(init_weights)

    discriminator = DiscriminativeNet()
    discriminator.apply(init_weights)

    # Enable cuda if available
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()

    return generator, discriminator

if __name__ == '__main__':
    main()