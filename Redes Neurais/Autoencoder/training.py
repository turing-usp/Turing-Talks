import Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets

# Importing the CIFAR10 dataset from torchvision and loading it into a
# DataLoader object
to_tensor = torchvision.transforms.ToTensor()
training_data = datasets.CIFAR10(root='./dataset', train=True, download=True,transform=to_tensor)
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=50, shuffle=True,num_workers=4, pin_memory=True)

# Instantiating the Autoencoder neural network
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Autoencoder.Autoencoder().to(device)

# Setting the number of epochs in the training
epochs = 5

# We'll be using the Adam optimizer with learning rate 0.01
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Instantiating our loss function, which will
# be the Mean Squared Error
criterion = nn.MSELoss()

# Training
for i in range(epochs):
    # Keeping tracking of things for displaying the progress of the training
    total = len(training_data)
    current = 0
    count = 0

    # Performing an epoch
    for batch, _ in training_dataloader:
        if not (count % 100): 
            print("Epoch: " + str(i+1) + " percentage: {:3.2f}%".format(100*current/total), end='\r', flush=True)

        # Sending batch to device (GPU or CPU)
        x = batch.to(device)
        
        # Erasing the gradients stored
        optimizer.zero_grad()

        # Sending batch to the Autoencoder and computing the loss
        y = net(x)
        loss = criterion(y, x)

        # Backpropagating gradients
        loss.backward()

        # Running the optimizer
        optimizer.step()

        # Keeping track of things
        current += len(batch)
        count += 1

    print("Epoch: " + str(i+1) + " percentage: {:3.2f}%".format(100*current/total))

# Saving our trained Autoencoder
torch.save(net.state_dict(), "neuralnet")
print("Done!")
