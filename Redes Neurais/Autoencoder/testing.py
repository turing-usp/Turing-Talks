import Autoencoder
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# Getting random sample from testing set
to_tensor = torchvision.transforms.ToTensor()
test_data = datasets.CIFAR10(root='./dataset', train=False, download=True, transform=to_tensor)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
sample = next(iter(test_dataloader))[0]

# Displaying original sample image
img1 = sample.numpy()[0].transpose(1, 2, 0)
fig, axes = plt.subplots(3, 1)
axes[0].imshow(img1)

# Loading Autoencoder
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
net = Autoencoder.Autoencoder()
loaded = torch.load('neuralnet', map_location=device)
net.load_state_dict(loaded)
net.eval()

# Encoding image and displaying it
encoded = net.encode(sample)
img2 = encoded.detach().numpy()[0].transpose(1, 2, 0)
axes[1].imshow(img2)

# Decoding image and displaying it
decoded = net.decode(encoded)
img3 = decoded.detach().numpy()[0].transpose(1, 2, 0)
axes[2].imshow(img3)

# Calculating and printing loss
criterion = nn.MSELoss()
print("Calculated loss: {:3.6f}".format(float(criterion(decoded, sample))))

axes[0].title.set_text('3 Channel Original image (32x32)')
axes[1].title.set_text('3 Channel Encoded image (15x15)')
axes[2].title.set_text('3 Channel Recovered image (32x32)')

axes[0].set_yticks([])
axes[0].set_xticks([])
axes[1].set_yticks([])
axes[1].set_xticks([])
axes[2].set_yticks([])
axes[2].set_xticks([])

plt.show()
