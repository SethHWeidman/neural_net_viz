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

    weight_matrix_1 = np.ones(num_input * num_hidden)
    list_of_weights = list(itertools.product(range(num_input), range(num_hidden)))
    weights_1 = []
    for el1, el2 in zip(weight_matrix_1, list_of_weights):
        d = {'value': el1, 'coordinates': el2}
        weights_1.append(d)

    weight_matrix_2 = np.ones(num_hidden * 1)
    list_of_weights = list(itertools.product(range(num_hidden+1), range(1)))
    weights_2 = []
    for el1, el2 in zip(weight_matrix_2, list_of_weights):
        d = {'value': el1, 'coordinates': el2}
        weights_2.append(d)

    input_neurons = [{'x': input_x, 'y': y, 'layer': 0, 'neuron': i} for i, y in enumerate(input_ys)]
    hidden_neurons = [{'x': hidden_x, 'y': y, 'layer': 1, 'neuron': i} for i, y in enumerate(hidden_ys)]
    output_neuron = [{'x': output_x, 'y': output_y}]

    all_neurons = input_neurons + hidden_neurons + output_neuron

    all_weights = {'input_to_hidden': weights_1, 'hidden_to_output': weights_2}

    return all_neurons, all_weights
