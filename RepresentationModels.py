'''
    Representation learning:
    Goal: use unsupervised learning techniques to learn a representation of given data.
    The hope is that this representation will be useful to reduce the amount of data that is needed for training the supervised model for solving the actual task.
    To verify how good the learned representation is, train a supervised model using these representations that predicts the available pretrain labels.

    Methology:
    1. Create a several neural networks that learn to encode the data into a representation.
    2. Train a supervised model on each of the learned representations. The superverised model trained on the different representations should be very shallow (1 or two fully connected layers) and should be trained for a very short time. The goal is to make the performance of the encoders comparable.
'''

# Dependencies
import torch
import torch.nn as nn

'''
    Autoencoder for dimensionality reduction:
    Both encoder and decoder using three linear layers
'''

# for this to make sense the encoding dimension should be significantly smaller than the input dimension
# specifically, the encoding_dim*3 shold be smaller than the input_size
class LinearAutoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(LinearAutoencoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoding_dim*3),
            nn.ReLU(),
            nn.Linear(int(encoding_dim*3), int(encoding_dim*2)),
            nn.ReLU(),
            nn.Linear(encoding_dim*2, encoding_dim)
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim*2),
            nn.ReLU(),
            nn.Linear(encoding_dim*2, encoding_dim*3),
            nn.ReLU(),
            nn.Linear(encoding_dim*3, input_size),
            nn.Sigmoid() # the feature values are between 0 and 1
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


'''
    Autocoder for dimensionality reduction:
    Using three convolutional/deconvolutional layers for encoder/decoder
'''
class ConvAutoencoder(nn.Module):
    def __init__(self, number_filters):
        super(ConvAutoencoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, number_filters*3, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(number_filters*3, number_filters*2, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(number_filters*2, number_filters, kernel_size=3, stride=2, padding=0)
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(number_filters, number_filters*2, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(number_filters*2, number_filters*3, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(number_filters*3, 1, kernel_size=3, stride=2, padding=0, output_padding=1), # need out padding to get the right size
            nn.Sigmoid() # the feature values are between 0 and 1
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.squeeze(1)
        return x
    
'''
    Autocoder for dimensionality reduction:
    Using two convolutional/deconvolutional layers and one fully connected layer for both encoder and decoder
'''
class ConvLinearAutoencoder(nn.Module):
    def __init__(self, number_filters, encoding_dim):
        super(ConvLinearAutoencoder, self).__init__()
        self.number_filters = number_filters
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, number_filters*2, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(number_filters*2, number_filters, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        ) 
        # bottleneck layer
        self.fcencoder = nn.Linear(249*number_filters, encoding_dim)
        self.fcdecoder = nn.Linear(encoding_dim, 249*number_filters)

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(number_filters, number_filters*2, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(number_filters*2, 1, kernel_size=3, stride=2, padding=0),
            nn.Sigmoid() # the feature values are between 0 and 1
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.view(-1, 249*self.number_filters)
        x = self.fcencoder(x)
        x = self.fcdecoder(x)
        x = x.view(-1,self.number_filters, 249)
        x = self.decoder(x)
        x = x.squeeze(1)
        return x
    
# contractive loss function
def contractive_loss(W, x, recons_x, h, lam=1e-4):
    mse_loss = nn.MSELoss()(recons_x, x)
    
    dh = h * (1 - h) # Derivative of sigmoid
    w_sum = torch.sum(Variable(W)**2, dim=1)
    w_sum = w_sum.unsqueeze(1) # Shape to 2D tensor
    contractive_loss = torch.sum(torch.mm(dh**2, w_sum), 0)
    return mse_loss + contractive_loss.mul_(lam)