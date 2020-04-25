# Autoencoder for image compression

This is an implementation of an autoencoder for image compression, made with Torch.

The dataset used is the CIFAR-10, which contains 32x32 RGB images of the following classes:
 1. airplane
 2. automobile
 3. bird
 4. cat
 5. deer
 6. dog
 7. frog
 8. horse
 9. ship
 10. truck
 
The autoencoder managed to reduce the dimensions of the images to 15x15, which represents
a used storage space of only 22% of the original space occupied by each original image.

After the compression, the autoencoder succeeded in generating recovered 32x32 images which
are highly similar to the original ones.

The layers of the neural network used are the following
1. Encoding layers
  - 2D Convolutional
  - 2D Batch Normalization
  - 2D Convolutional
  - 2D Batch Normalization
  - 2D Convolutional
  - 2D Batch Normalization
2. Decoding layers
  - 2D Transposed Convolutional
  - 2D Batch Normalization
  - 2D Transposed Convolutional
  - 2D Batch Normalization
  - 2D Transposed Convolutional
  - 2D Batch Normalization
  
# Compression Example
![Compression example](https://i.ibb.co/rHSD445/Screenshot-from-2020-03-17-03-57-59.png)

# About the files
1. The Autoencoder.py file implements the Autoencoder class in torch.
2. The training.py file performs the training over the entire training dataset.
3. The testing.py file gets a random sample from the testing dataset and plots
an image similar to the one in the compression example, calculating the 
loss (Mean Squared Error) of the compression performed.
4. The neuralnet file is the saved trained autoencoder.
