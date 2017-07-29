import numpy as np
import itertools

def plot_net(num_input, num_hidden, height, width):
    input_height_margin = 100
    hidden_height_margin = 50
    width_margin = 100
    xs = np.linspace(width_margin, width - width_margin, 3)
    input_x = xs[0]
    hidden_x = xs[1]
    output_x = xs[2]
    input_ys = np.linspace(input_height_margin,
                           height - input_height_margin,
                           num_input)
    hidden_ys = np.linspace(hidden_height_margin,
                            height - hidden_height_margin,
                            num_hidden)
    output_y = height / 2.0

    input_neurons = [{'x': input_x, 'y': y} for y in input_ys]
    hidden_neurons = [{'x': hidden_x, 'y': y} for y in hidden_ys]
    output_neuron = [{'x': output_x, 'y': output_y}]

    all_neurons = input_neurons + hidden_neurons + output_neuron

    return all_neurons
