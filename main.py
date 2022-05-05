import torch
from torch import nn
from tqdm import tqdm

import math
import matplotlib.pyplot as plt
from torch.nn.modules import loss

torch.manual_seed(111)

# prepare the training data
train_data_length = 1024
train_data = torch.zeros((train_data_length, 2))
train_data[:, 0] = 2*math.pi*torch.rand(train_data_length)
train_data[:, 1] = torch.sin(train_data[:,0])
train_labels = torch.zeros(train_data_length)
train_set = [(train_data[i], train_labels[i]) for i in range(train_data_length)]

plt.plot(train_data[:, 0], train_data[:, 1], ".")

# create pytorch data loader: this shuffles the data from train_set and return it in batches of 32 samples that will be use for nn training
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# discriminator 
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(.3), 
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Dropout(.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        output = self.model(x)
        return output

discriminator = Discriminator()

# generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        output = self.model(x)
        return output

generator = Generator()

learning_rate = .001
epochs = 500
loss_function = nn.BCELoss() # binary cross entropy

# use Adam algorithm to train the model
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=learning_rate)

for epoch in (pbar := tqdm(range(epochs))):
    pbar.set_description(f"GAN training")
    for n, (real_samples, _) in enumerate(train_loader):
        # data for training the discriminator
        real_samples_labels = torch.ones((batch_size, 1))
        latent_space_samples = torch.randn((batch_size, 2))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1))
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        # training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # data for training the generator
        latent_space_samples = torch.randn((batch_size, 2))

        #training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step() 

        pbar.set_postfix(loss_disc=loss_discriminator.item(), loss_gen=loss_generator.item())

        # debug loss
        # if n == batch_size -1:
        #    print(f"Epoch: {epoch} Loss D.: {loss_discriminator} Loss G.: {loss_generator}")
        
    # debug with some plot
    generated_samples = generated_samples.detach()
    plt.clf()
    plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
    plt.show(block=False)
    plt.pause(.1)
    
# check the samples obtained from the generator 
latent_space_samples = torch.randn(100, 2)
generated_samples = generator(latent_space_samples)

# show plots for debugging
plt.show()
