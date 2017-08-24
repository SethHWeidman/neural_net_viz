import numpy as np
import itertools

def plot_net(num_input, num_hidden, height, width):
    '''
    This function returns the coordinates and other information needed to plot the
    points on the net in space.
    '''

    input_height_margin = 100
    hidden_height_margin = 50
    width_margin = 100

    # X coordinates for neurons
    xs = np.linspace(width_margin, width - width_margin, 3)
    input_x = xs[0]
    hidden_x = xs[1]
    output_x = xs[2]

    # Y coordinates for neurons
    input_ys = np.linspace(input_height_margin,
                           height - input_height_margin,
                           num_input)
    hidden_ys = np.linspace(hidden_height_margin,
                            height - hidden_height_margin,
                            num_hidden)

    output_y = height / 2.0

    # Coordinates for weight lines
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

    # Other information about the neurons
    input_neurons = [{'x': input_x, 'y': y, 'layer': 0, 'neuron': i, 'value': 1} for i, y in enumerate(input_ys)]
    hidden_neurons = [{'x': hidden_x, 'y': y, 'layer': 1, 'neuron': i, 'value': 1} for i, y in enumerate(hidden_ys)]
    output_neuron = [{'x': output_x, 'y': y, 'layer': 2, 'neuron': i, 'value': 1} for i, y in enumerate([output_y])]

    #
    all_neurons = input_neurons + hidden_neurons + output_neuron

    # A "weight dictionary"
    all_weights = [weights_1] + [weights_2]

    return all_neurons, all_weights

def update_coordinate_values(layers, coords):
    '''
    Updates the values contained in the "coordinates" dictionary, based on the
    values returned from the neural net, which are in a list of lists.
    '''
    for i in range(len(layers)):
        layer = layers[i]
        for item in coords:
            for j, element in enumerate(layer):
                if item['layer'] == i and item['neuron'] == j:
                    item['value'] = element
    return coords


def update_weight_values(weights, weight_values):
    '''
    Updates the values contained in the "weights" dictionary, based on the
    values in the weight_values array of arrays.
    '''
    for k in range(len(weights)):
        for i, weight_col in enumerate(weight_values[k]):
            for j, weight_val in enumerate(weight_col):
                for el in weights[k]:
                    if el['coordinates'][0] == i and el['coordinates'][1] == j:
                        el['value'] = weight_val
    return weights
