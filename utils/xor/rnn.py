#!/usr/bin/env python

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import array_ops, nn_ops, init_ops, math_ops
from tensorflow.python.ops import variable_scope as vs


class MyLSTMCell(RNNCell):
    
    """
    LSTMCell implementation that is compatible with TensorFlow. To solve the compatibility issue, this
    class inherits TensorFlow RNNCell class.
    """

    def __init__(self, num_units, num_proj, forget_bias=1.0, activation=None):
        
        """
        Initialize a class instance.

        In this function the following is done:

        1. Store the input parameters and calculate other ones that you think necessary.

        2. Initialize some trainable variables which will be used during the calculation.

        :param num_units: The number of units in the LSTM cell.
        :param num_proj: The output dimensionality. For example, if you expect your output of the cell at each time step to be a 10-element vector, then num_proj = 10.
        :param forget_bias: The bias term used in the forget gate. By default we set it to 1.0.
        :param activation: The activation used in the inner states. By default we use tanh.

        There are biases used in other gates, but since TensorFlow doesn't have them, we don't implement them either.
        """
        
        super(MyLSTMCell, self).__init__(_reuse=None)
        self.num_units = num_units
        self.num_proj = num_proj
        self.forget_bias = forget_bias
        self.activation = activation or tf.nn.tanh
            
    # The following 2 properties are required when defining a TensorFlow RNNCell.
    @property
    def state_size(self):
        
        """
        Overrides parent class method. Returns the state size of of the cell.

        state size = num_units + output_size

        :return:'' An integer.
        """
        
        return self.num_units + self.num_proj

    @property
    def output_size(self):
        
        """
        Overrides parent class method. Returns the output size of the cell.

        :return: An integer.
        """
        
        return self.num_proj

    def call(self, inputs, state):
        
        """
        Run one time step of the cell. Given the current inputs and the state from the last time step, calculate the current state and cell output.

        :param inputs: The input at the current time step. The last dimension of it should be 1.
        :param state:  The state value of the cell from the last time step. The state size can be found from function state_size(self).
        :return: A tuple containing (output, new_state). For details check TensorFlow LSTMCell class.
        """
        
        c = tf.slice(state, [0, 0], [-1, self.num_units])
        h = tf.slice(state, [0, self.num_units], [-1, self.num_proj])

        gates  = tf.contrib.layers.fully_connected(tf.concat([inputs, h], 1),num_outputs=4 * self.num_units, activation_fn = None)                                 
        # f = forget_gate, i = input_gate, j = new_input, o = output_gate
        i, j, f, o = tf.split(gates,4, 1)

        forget_bias = 1.0
        new_c = (c * tf.nn.sigmoid(f + forget_bias)+ tf.nn.sigmoid(i) * tf.nn.tanh(j))
        new_h =  tf.nn.tanh(new_c) * tf.nn.sigmoid(o)
        new_h_dash = tf.contrib.layers.fully_connected(new_h, num_outputs=2, activation_fn = None)                                 
        new_state = tf.concat([new_c, new_h_dash], 1)
        
        return new_h_dash, new_state
