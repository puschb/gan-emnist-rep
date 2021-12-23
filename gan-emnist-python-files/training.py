import torch
import torch.nn.functional
from tqdm import tqdm
import numpy as np
from numpy.random import default_rng
import torch.utils.data
import time


def train(model_generator, model_discriminator, Goptimizer, Doptimizer, train_loader, epochs, cost_function,latent_size):

    #currently no validation loader (parameter) and metrics stuff because I haven't implemente any metrics yet

    if torch.cuda.is_available():
        print("Using cuda")
        device = torch.device("cuda")
    else:
        device = "cpu"

    model_generator.to(device)
    model_generator.train()
    model_discriminator.to(device)
    model_discriminator.train()
    start_time = time.time()

    for epoch in range(epochs):

        discriminator_loss = 0
        generator_loss = 0
        with tqdm(train_loader, unit= "Batch") as tepoch:
            for inputs,_ in tepoch:

                model_generator.zero_grad()
                model_discriminator.zero_grad()
                inputs = inputs.to(device)


                generated_inputs = model_generator(torch.tensor(np.random.normal(0,1,(inputs.shape[0],latent_size)),
                                                                dtype=torch.float32,device=device)) #this has to be type float32 to work for some reason

                generated = torch.tensor(np.zeros((inputs.shape[0],1)),device=device, dtype=torch.float32)
                real = torch.tensor(np.ones((inputs.shape[0],1)),device=device, dtype=torch.float32)

                outputs_from_real = model_discriminator(inputs)
                outputs_from_generated = model_discriminator(generated_inputs)

                #generator
                generator_loss = cost_function(outputs_from_generated, real)
                generator_loss.backward()
                Goptimizer.step()

                #discriminator
                discriminator_loss = (cost_function(model_discriminator(generated_inputs.detach()),generated) +
                                      cost_function(outputs_from_real,real)) / 2
                discriminator_loss.backward()
                Doptimizer.step()

                tepoch.set_description(f"Epoch {epoch}")
                tepoch.set_postfix(generator_loss=generator_loss.item(),discriminator_loss = discriminator_loss.item())

        print("\nEpoch {}, Generator loss: {:.2f}. Discriminator loss: {:.2f}.".format(epoch, generator_loss,
                                                                                        discriminator_loss))
    return (time.time()-start_time)/60