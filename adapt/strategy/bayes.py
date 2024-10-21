# import packages
import numpy as np
import tensorflow as tf
import math
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Activation
from tensorflow.keras.utils import to_categorical
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

from adapt.strategy.strategy import Strategy
from collections import defaultdict
# calculatedNeurons = []
# BOcomplete = 0

class BayesStrategy(Strategy):
  '''A strategy that randomly selects neurons from uncovered neurons.
  
  This strategy selects neurons from a set of uncovered neurons. This strategy
  is first introduced in the following paper, but not exactly same. Please see
  the following paper for more details:

  DeepXplore: Automated Whitebox Testing of Deep Learning Systems
  https://arxiv.org/abs/1705.06640
  '''

  def __init__(self, network):
    '''Create a strategy and initialize its variables.
    
    Args:
      network: A wrapped Keras model with `adapt.Network`.

    Example:

    >>> from adapt import Network
    >>> from adapt.strategy import UncoveredRandomStrategy
    >>> from tensorflow.keras.applications.vgg19 import VGG19
    >>> model = VGG19()
    >>> network = Network(model)
    >>> strategy = UncoveredRandomStrategy(network)
    '''

    super(BayesStrategy, self).__init__(network)
    self.network = network

    # A variable that keeps track of the covered neurons.
    self.covered = None

  def select(self, k, input, groundT):
    '''Select k uncovered neurons.
    
    Select k neurons, and returns their location.

    Args:
      k: A positive integer. The number of neurons to select.

    Returns:
      A list of locations of selected neurons.
    '''
    selectedNeurons = []
    # function that extract neurons from convolution layer
    def optimize_conv_layer_neurons(model, selectLayer, neuron_indices, X_train_subset, Y_train_subset, n_calls=10):
      """
      Function to optimize neuron weights in a specified convolutional layer.

      Args:
          model: Keras model object.
          selectLayer: Name of the convolutional layer (e.g., 'block1_conv1').
          neuron_indices: Number of neurons (output channels) to optimize.
          X_train_subset: Subset of training data for optimization.
          Y_train_subset: Subset of training labels for optimization.
          n_calls: Number of optimization iterations.

      Returns:
          A list of top neurons (output channels) based on the loss values.
      """


      # Get the shape of the weights for selected layer
      layer_weights_shape = model.get_layer(selectLayer).get_weights()[0].shape

      # Store the loss values and neuron indices
      loss_values_per_neuron = []
      neuron_indices_list = []
      optimzied_neuron_loss = defaultdict(list)

      # Backup original weights
      backup_bias = model.get_layer(selectLayer).get_weights()[1]
      backup_weights = model.get_layer(selectLayer).get_weights()[0]

      # Iterate through each neuron (output channel) in the convolutional layer
      for output_channel in range(layer_weights_shape[-1]):  # The last dimension represents the output channels
        # Optimize weights for the current output channel
        dimensions = [Real(-1.0, 1.0, name=f'weight_{i}') for i in
                      range(layer_weights_shape[0] * layer_weights_shape[1] * layer_weights_shape[2])]

        @use_named_args(dimensions=dimensions)
        def objective(**weights):
          # Convert the named dimensions to a tensor (reshape according to convolution kernel dimensions)
          weights_list = [weights[f'weight_{i}'] for i in range(len(weights))]
          weights_tensor = np.array(weights_list).reshape(
            (layer_weights_shape[0], layer_weights_shape[1], layer_weights_shape[2], 1))

          # Get the current weights of the selected layer
          current_weights_conv = model.get_layer(selectLayer).get_weights()[0]

          # Replace the weights for the current output channel
          current_weights_conv[:, :, :, output_channel:output_channel + 1] = weights_tensor

          # Set the weights for selected convolutional layer
          model.get_layer(selectLayer).set_weights(
            [current_weights_conv, np.zeros(layer_weights_shape[-1])])  # Set weights and bias (bias is 0)

          # Forward pass
          with tf.GradientTape() as tape:
            logits = model(X_train_subset, training=False)
            loss_value = tf.keras.losses.CategoricalCrossentropy()(Y_train_subset, logits)

          # Store the loss value
          loss_values_per_neuron.append(loss_value.numpy())

          # Restore original weights
          model.get_layer(selectLayer).set_weights([backup_weights, backup_bias])  # Reset original weights and bias

          return loss_value.numpy()

        # Perform the optimization for the current output channel
        result = gp_minimize(objective, dimensions=dimensions, n_calls=n_calls)

        # After each iteration, store the top neurons
        optimzied_neuron_loss[output_channel].append(result.fun)

      # Get top neurons based on lowest loss values
      sorted_neurons = sorted(optimzied_neuron_loss.items(), key=lambda x: np.mean(x[1]))[:neuron_indices]

      # Return the selected neurons with lowest loss
      return sorted_neurons, loss_values_per_neuron

    # function that extract neurons from specific layer
    def optimize_layer_neurons(model, selectLayer, neuron_number, X_train_subset, Y_train_subset, n_calls=10):
      """
      Function to optimize neurons in the selected layer of the model.

      Args:
          model: The Keras model object.
          selectLayer: The layer name to perform optimization (e.g., 'fc2', 'block1_conv1').
          neuron_number: Number of neurons to select based on lowest loss values.
          X_train_subset: Subset of training data for optimization.
          Y_train_subset: Subset of training labels for optimization.
          n_calls: Number of optimization iterations for each neuron.

      Returns:
          A dictionary of top neurons with their corresponding loss values.
      """

      # Get the shape of the weights for selected layer
      layer_weights_shape = model.get_layer(selectLayer).get_weights()[0].shape

      # Store the loss values and neuron indices
      optimzied_neuron_loss = defaultdict(list)
      backup_bias = model.get_layer(selectLayer).get_weights()[1]
      backup_weights = model.get_layer(selectLayer).get_weights()[0]

      # Iterate through each neuron (output) in the selected layer
      for neuron_index in range(layer_weights_shape[1]):
        # Define the optimization space for the current neuron
        dimensions = [Real(-1.0, 1.0, name=f'weight_{i}') for i in range(layer_weights_shape[0])]

        @use_named_args(dimensions=dimensions)
        def objective(**weights):
          # Convert weights to tensor
          weights_list = [weights[f'weight_{i}'] for i in range(layer_weights_shape[0])]
          weights_tensor = np.array(weights_list).reshape((layer_weights_shape[0], 1))

          # Get the current weights of the selected layer
          current_weights = model.get_layer(selectLayer).get_weights()[0]

          # Replace weights for the current neuron
          current_weights[:, neuron_index:neuron_index + 1] = weights_tensor

          # Set the optimized weights for the selected layer
          model.get_layer(selectLayer).set_weights(
            [current_weights, np.zeros(layer_weights_shape[1])])  # Set weights and bias (bias is 0)

          # Forward pass
          with tf.GradientTape() as tape:
            logits = model(X_train_subset, training=False)
            loss_value = tf.keras.losses.CategoricalCrossentropy()(Y_train_subset, logits)

          # Restore the original weights after computation
          model.get_layer(selectLayer).set_weights([backup_weights, backup_bias])

          # Return the loss value
          return loss_value.numpy()

        # Perform optimization for the current neuron
        result = gp_minimize(objective, dimensions=dimensions, n_calls=n_calls)

        # Store the optimization result
        optimzied_neuron_loss[neuron_index].append(result.fun)

      # Get top neurons based on lowest loss values
      sorted_neurons = sorted(optimzied_neuron_loss.items(), key=lambda x: np.mean(x[1]))[:neuron_number]

      # Return the selected neurons with their loss values
      return sorted_neurons

    # set neuron selection percentage
    NCpercent = 0.1
    # extract neurons from convolutuion layers
    # selectLayer = 'block1_conv1'
    # neuron_indices = 1
    #
    # # Call the function to optimize neurons in conv1
    # top_neurons, losses = optimize_conv_layer_neurons(self.network.model, selectLayer, neuron_indices, input, groundT)
    # neuron_conversion = [neuron[0] for neuron in top_neurons]
    # selectedNeurons.extend(neuron_conversion)
    #
    # print("conv1 neurons: ")
    # print(top_neurons)
    # print("selected neurons neurons: ")
    # print(selectedNeurons)
    #
    # selectLayer = 'block2_conv1'
    # neuron_indices = 1
    # # Call the function to optimize neurons in conv2
    # top_neurons, losses = optimize_conv_layer_neurons(self.network.model, selectLayer, neuron_indices, input, groundT)
    # neuron_conversion = [(neuron[0] + 12) for neuron in top_neurons]
    # # neuron_conversion = [(neuron[0] + 64) for neuron in top_neurons]
    # selectedNeurons.extend(neuron_conversion)
    #
    # print("conv2 neurons: ")
    # print(top_neurons)
    # print("selected neurons neurons: ")
    # print(selectedNeurons)

    # extract neurons from fully connected layers

    # selectLayer = 'fc1'
    # neuron_number = 5
    #
    # # Call the function to optimize neurons in the selected layer
    # top_neurons = optimize_layer_neurons(self.network.model, selectLayer, neuron_number, input, groundT)
    # neuron_conversion = [(neuron[0] + 44) for neuron in top_neurons]
    # selectedNeurons.extend(neuron_conversion)

    # print("fc1 neurons: ")
    # print(top_neurons)
    # print("selected neurons neurons: ")
    # print(selectedNeurons)

    # selectLayer = 'fc2'
    # neuron_number = 5
    #
    # # Call the function to optimize neurons in the selected layer
    # top_neurons = optimize_layer_neurons(self.network.model, selectLayer, neuron_number, input, groundT)
    # neuron_conversion = [(neuron[0] + 164) for neuron in top_neurons]
    # selectedNeurons.extend(neuron_conversion)
    #
    # print("fc2 neurons: ")
    # print(top_neurons)
    # print("selected neurons neurons: ")
    # print(selectedNeurons)

    selectLayer = 'before_softmax'
    # vgg16
    # selectLayer = 'dense_5'
    # resnet18
    # selectLayer = 'dense'
    neuron_number = 7

    # Call the function to optimize neurons in the selected layer
    top_neurons = optimize_layer_neurons(self.network.model, selectLayer, neuron_number, input, groundT)
    # lenet 5
    neuron_conversion = [(neuron[0] + 248) for neuron in top_neurons]
    # lenet 4
    # neuron_conversion = [(neuron[0] + 164) for neuron in top_neurons]
    # vgg16
    # neuron_conversion = [(neuron[0] + 6720) for neuron in top_neurons]
    # ResNet18
    # neuron_conversion = [(neuron[0] + 15936) for neuron in top_neurons]
    selectedNeurons.extend(neuron_conversion)

    print("before_softmax neurons: ")
    print(top_neurons)
    print("selected neurons neurons: ")
    print(selectedNeurons)
    #
    # print("select neuron number: ")
    # print(len(selectedNeurons))
    outputNeurons = [self.neurons[i] for i in selectedNeurons]
    return outputNeurons

  def init(self, covered, **kwargs):
    '''Initialize the variable of the strategy.

    This method should be called before all other methods in the class.

    Args:
      covered: A list of coverage vectors that the initial input covers.
      kwargs: Not used. Present for the compatibility with the super class.

    Returns:
      Self for possible call chains.

    Raises:
      ValueError: When the size of the passed coverage vectors are not matches
        to the network setting.
    '''

    # Flatten coverage vectors.
    self.covered = np.concatenate(covered)
    if len(self.covered) != len(self.neurons):
      raise ValueError('The number of neurons in network does not matches to the setting.')  

    return self

  def update(self, covered, **kwargs):
    '''Update the variable of the strategy.

    Args:
      covered: A list of coverage vectors that a current input covers.
      kwargs: Not used. Present for the compatibility with the super class.

    Returns:
      Self for possible call chains.
    '''

    # Flatten coverage vectors.
    covered = np.concatenate(covered)

    # Update coverage vectors.
    self.covered = np.bitwise_or(self.covered, covered)

    return self
