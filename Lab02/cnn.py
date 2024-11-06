import numpy as np
from scipy import signal
from skimage import measure
import data_generator

# Different activations functions


def activation(x, activation):

    # TODO: specify the different activation functions
    # 'activation' could be: 'linear', 'relu', 'sigmoid', or 'softmax'
    if activation == 'linear':
        return x
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif activation == 'relu':
        return np.maximum(0.0, x)
    elif activation == 'softmax':
        return (np.exp(x)/np.exp(x).sum())
        # TODO
    else:
        raise Exception("Activation function is not valid", activation)

# 2D convolutional layer


def conv2d_layer(h,     # activations from previous layer, shape = [height, width, channels prev. layer]
                 # conv. kernels, shape = [kernel height, kernel width, channels prev. layer, channels this layer]
                 W,
                 b,     # bias vector
                 act    # activation function
                 ):
    # Specify the number of input and output channels
    CI = h.shape[2]
    CO = W.shape[3]

    output = np.zeros((h.shape[0], h.shape[1], CO))

    for i in range(CO):
        # Temporary array to accumulate convolutions for each output channel
        conv_sum = np.zeros_like(output[:, :, i])

        for j in range(CI):
            kernel = W[:, :, j, i]
            flipped_kernel = np.flipud(np.fliplr(kernel))

            # Accumulate convolutions for each output channel
            conv_sum += signal.convolve2d(
                h[:, :, j], flipped_kernel, mode='same')

        # Add bias and apply activation function for the current output channel
        output[:, :, i] = activation(conv_sum + b[i], act)

    return output

# 2D max pooling layer
# activations from conv layer, shape = [height, width, channels]


def pool2d_layer(h):
    # TODO: implement the pooling operation
    # 1. Specify the height and width of the output
    sy, sx = h.shape[0] // 2, h.shape[1] // 2

    pool_size = (2, 2)
    # 2. Specify array to store output
    ho = np.zeros((sy, sx, h.shape[2]))

    # 3. Perform pooling for each channel.
    #    You can, e.g., look at the measure.block_reduce() function
    #    in the skimage library

    for channel in range(h.shape[2]):
        ho[:, :, channel] = measure.block_reduce(
            h[:, :, channel], block_size=pool_size, func=np.max)
        # Using block_reduce from skimage to perform max pooling

    return ho


# Flattening layer
# activations from conv/pool layer, shape = [height, width, channels]
def flatten_layer(h):
    # TODO: Flatten the array to a vector output.
    # You can, e.g., look at the np.ndarray.flatten() function
    return h.flatten()


# Dense (fully-connected) layer
def dense_layer(h,   # Activations from previous layer
                W,   # Weight matrix
                b,   # Bias vector
                act  # Activation function
                ):
    # TODO: implement the dense layer.
    # You can use the code from your implementation
    # in Lab 1. Make sure that the h vector is a [Kx1] array.
    h = h[:, np.newaxis]
    z = np.matmul(W, h)+b
    a = np.maximum(0.0, z)
    a = activation(z, act)
    return a[:, 0]


# ---------------------------------
# Our own implementation of a CNN
# ---------------------------------
class CNN:
    def __init__(
        self,
        dataset,         # DataGenerator
        verbose=True     # For printing info messages
    ):
        self.verbose = verbose
        self.dataset = dataset

    # Set up the CNN from provided weights
    def setup_model(
        self,
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        lname,               # List of layer names
        activation='relu'    # Activation function of layers
    ):
        self.activation = activation
        self.lname = lname

        self.W = W
        self.b = b

        # TODO: specify the total number of weights in the model
        #       (convolutional kernels, weight matrices, and bias vectors)

        self.N = sum(np.prod(np.array(w).shape) for w in self.W) + \
            sum(np.prod(np.array(b).shape) for b in self.b)

        print('Number of model weights: ', self.N)

    # Feedforward through the CNN of one single image
    def feedforward_sample(self, h):

        # Loop over all the model layers
        for l in range(len(self.lname)):
            act = self.activation

            if self.lname[l] == 'conv':
                h = conv2d_layer(h, self.W[l], self.b[l], act)
            elif self.lname[l] == 'pool':
                h = pool2d_layer(h)
            elif self.lname[l] == 'flatten':
                h = flatten_layer(h)
            elif self.lname[l] == 'dense':
                if l == (len(self.lname)-1):
                    act = 'softmax'
                h = dense_layer(h, self.W[l], self.b[l], act)
        return h

    # Feedforward through the CNN of a dataset
    def feedforward(self, x):
        # Output array
        y = np.zeros((x.shape[0], self.dataset.K))

        # Go through each image
        for k in range(x.shape[0]):
            if self.verbose and np.mod(k, 1000) == 0:
                print('sample %d of %d' % (k, x.shape[0]))

            # Apply layers to image
            y[k, :] = self.feedforward_sample(x[k])

        return y

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # TODO: formulate the training loss and accuracy of the CNN.
        # Assume the cross-entropy loss.
        # For the accuracy, you can use the implementation from Lab 1.
        yp_train = self.feedforward(self.dataset.x_train)
        yp_test = self.feedforward(self.dataset.x_test)

        train_loss = - \
            np.mean(
                np.log(yp_train[np.arange(len(yp_train)), self.dataset.y_train]))
        train_acc = np.mean(np.argmax(yp_train, 1) == self.dataset.y_train)
        print("\tTrain loss:     %0.4f" % train_loss)
        print("\tTrain accuracy: %0.2f" % train_acc)

        # TODO: formulate the test loss and accuracy of the CNN
        test_loss = - \
            np.mean(
                np.log(yp_test[np.arange(len(yp_test)), self.dataset.y_test]))
        test_acc = np.mean(np.argmax(yp_test, 1) == self.dataset.y_test)
        print("\tTest loss:      %0.4f" % test_loss)
        print("\tTest accuracy:  %0.2f" % test_acc)
