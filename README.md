# GAN for generating sin(x) data points from 0 to 2$\pi$.

Generative Adversarial Networks are a form of neural network architecture in which a discriminator and a generator contest with each other in a game. The architecture learns to generate new data with the same probability distribution as the training set. 

The generator network creates new data candidates, which are later evaludated by the discriminator. The generator or generative network learns to map the latent space to a data distribution of interest, while the discriminator or discriminative network evaluates the candidates produced by the generator and compares them to the true data distribution.

## Architecture 
In this case, the generator defined by `Discriminator()` class, has a two dimensional input, while the first hidden layer is composed by 256 neurons with ReLU activation. The second and third hidden layers of the generator are composed of 128 and 64 neurons respectively, with again ReLU activation. Finally, the output is composed by a single neuron with activation based on the Sigmoid function. All the three hidden layers use dropout to avoid overfitting.

On the other hand, the discriminator defined by `Generator()` class, has a two dimensional input, but in this case the first hidden layer is composed by 16 neurons. The second hidden layer is formed by 32 neurons, while the output layer is composed by two neurons. In this case, there is no dropout and all the layers used ReLU activation.

## Run 
By simply running the `main.py` file found in this repo, each epoch is plotted to see the evolution of the learnign process. 

### Requirements
```
python3
pytorch >= 1.4.0
matplotlib >= 3.3.3
```
