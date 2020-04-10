from CycleGen_model import CycleGenerator
from Dis_model import Discriminator
import torch

def create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
        """Builds the generators and discriminators."""
        
        # Instantiate generators
        G_XtoY = CycleGenerator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)
        G_YtoX = CycleGenerator(conv_dim=g_conv_dim, n_res_blocks=n_res_blocks)
        # Instantiate discriminators
        D_X = Discriminator(conv_dim=d_conv_dim)
        D_Y = Discriminator(conv_dim=d_conv_dim)

        # move models to GPU, if available
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            G_XtoY.to(device)
            G_YtoX.to(device)
            D_X.to(device)
            D_Y.to(device)
            print('Models moved to GPU.')
        else:
            print('Only CPU available.')

        return G_XtoY, G_YtoX, D_X, D_Y