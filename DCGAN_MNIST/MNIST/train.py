import torch
import click as ck
from torch import nn, optim
from torch.autograd.variable import Variable
from utils import Logger
from Gen_model import GenerativeNet
from Dis_model import DiscriminativeNet
from helpers import (mnist_dataloader,noise, init_weights, train_discriminator, train_generator)

@ck.command()
@ck.option(
    '--batch-size', '-bs', default=100,
    help='Batch size')
@ck.option(
    '--epochs', '-e', default=200,
    help='Training epochs')
@ck.option(
    '--num-test-samples', '-ts', default=16,
    help='path to Data')
@ck.option(
    '--num-workers', '-nw', default=3,
    help='Number of parallel workers')

def main(batch_size, epochs, num_test_samples, num_workers):

    # Create loader with data, so that we can iterate over it
    data_loader = mnist_dataloader(batch_size, num_workers)
    # Num batches
    num_batches = len(data_loader)

    generator, discriminator = create_model()

    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Loss function
    loss = nn.BCELoss()

    test_noise = noise(num_test_samples)

    logger = Logger(model_name='DCGAN', data_name='MNIST')

    for epoch in range(epochs):
        for n_batch, (real_batch,_) in enumerate(data_loader):
            
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

            var = input()
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
                logger.display_status(
                    epoch, epochs, n_batch, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )
            # Model Checkpoints
            logger.save_models(generator, discriminator, epoch)

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