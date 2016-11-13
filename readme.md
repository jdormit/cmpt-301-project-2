# CMPT-301 Project 2 - Linear Models and Neural Networks
### Jeremy Dormitzer

## Goals
This project aims to:
- Read in data from the MNIST data set and the 20 News Groups dataset
- Use gradient descent to develop a classifier, and test it
- Use a 2-layer neural network to develop a classifier, and test it

### Training and Testing Models
1) Divide the data: 80% training data, 20% testing data. Further divide the testing data into 80% testing data, 20% validation data.
2) Train the model using the training data.
3) Tune the hyperparameters using the validation data.
4) Test the model using the testing data.

## Reading the Data
Different processes are required to load the MNIST data and the 20 News Groups set

### MNIST
The MNIST data is given in a [unique data format](http://yann.lecun.com/exdb/mnist/):

> TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
> 
>    [offset] [type]          [value]          [description] 
>    0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
>    0004     32 bit integer  60000            number of items 
>    0008     unsigned byte   ??               label 
>    0009     unsigned byte   ??               label 
>    ........ 
>    xxxx     unsigned byte   ??               label
> The labels values are 0 to 9.
> 
> TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
> 
>    [offset] [type]          [value]          [description] 
>    0000     32 bit integer  0x00000803(2051) magic number 
>    0004     32 bit integer  60000            number of images 
>    0008     32 bit integer  28               number of rows 
>    0012     32 bit integer  28               number of columns 
>    0016     unsigned byte   ??               pixel 
>    0017     unsigned byte   ??               pixel 
>    ........ 
>    xxxx     unsigned byte   ??               pixel
>    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
> 
> TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
> 
>    [offset] [type]          [value]          [description] 
>    0000     32 bit integer  0x00000801(2049) magic number (MSB first) 
>    0004     32 bit integer  10000            number of items 
>    0008     unsigned byte   ??               label 
>    0009     unsigned byte   ??               label 
>    ........ 
>    xxxx     unsigned byte   ??               label
> The labels values are 0 to 9.
> 
> TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
> 
>    [offset] [type]          [value]          [description] 
>    0000     32 bit integer  0x00000803(2051) magic number 
>    0004     32 bit integer  10000            number of images 
>    0008     32 bit integer  28               number of rows 
>    0012     32 bit integer  28               number of columns 
>    0016     unsigned byte   ??               pixel 
>    0017     unsigned byte   ??               pixel 
>    ........ 
>    xxxx     unsigned byte   ??               pixel

The [`idxreader`](/code/idxreader.py) modules can be used to read the MNIST IDX files as a [`numpy.array`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html).

Usage example:

```python
import idxreader

training_labels = idxreader.read(1, "/path/to/train-labels-idx1-ubyte")
# training_labels is a 1xN numpy array

training_images = idxreader.read(3, "/path/to/train-images-idx3-ubyte")
# training_images is a MxN numpy array
```
