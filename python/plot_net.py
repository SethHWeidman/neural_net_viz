import numpy as np
import itertools

def plot_net(num_input, num_hidden, height, width):
    '''
    This function returns the coordinates and other information needed to plot the
    points on the net in space.
    '''

    # Set up the margins
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

    weights_1 = initialize_weight_values(num_input, num_hidden)
    weights_2 = initialize_weight_values(num_hidden, 1)

    # Other information about the neurons
    input_neurons = [{'x': input_x, 'y': y, 'layer': 0, 'neuron': i, 'value': 1} for i, y in enumerate(input_ys)]
    hidden_neurons = [{'x': hidden_x, 'y': y, 'layer': 1, 'neuron': i, 'value': 1} for i, y in enumerate(hidden_ys)]
    output_neuron = [{'x': output_x, 'y': y, 'layer': 2, 'neuron': i, 'value': 1} for i, y in enumerate([output_y])]

    # All the neurons
    all_neurons = [input_neurons] + [hidden_neurons] + [output_neuron]

    # A "weight dictionary"
    all_weights = [weights_1] + [weights_2]

    return all_neurons, all_weights

def initialize_weight_values(num_input, num_hidden):
    '''
    Takes in a number of input and hidden neurons and returns a list of dictionaries,
    of the form:
    [{'value': 1, 'coordinates': [0,0]}, {'value': 1, 'coordinates': [0,1]}, ...]
    '''
    # Coordinates for weight lines
    weight_matrix = np.ones(num_input * num_hidden)
    list_of_weights = [[x, y] for x in range(num_input) for y in range(num_hidden)]
    weights = []
    for el1, el2 in zip(weight_matrix, list_of_weights):
        d = {'value': el1, 'coordinates': el2}
        weights.append(d)
    return weights


def update_coordinate_values(layers, coords):
    '''
    Updates the values contained in the "coordinates" dictionary, based on the
    values returned from the neural net, which are in a list of lists.

    layers: a list of lists
    coords: a list of coordinates objects
    '''
    for i in range(len(layers)):
        layer = layers[i]
        for item in coords[i]:
            for j, element in enumerate(layer):
                if item['layer'] == i and item['neuron'] == j:
                    item['value'] = element
    return coords


def update_weight_values(weights, weight_values):
    '''
    Updates the values contained in the "weights" dictionary, based on the
    values in the weight_values array of arrays.

    weights: a list of lists of key value pairs, each of which has a "value" and
    "coordinates".
    weight_values: a list of lists of lists simple containing the weight values,
    each of which is a matrix
    '''
    for k in range(len(weights)):
        for i, weight_col in enumerate(weight_values[k]):
            for j, weight_val in enumerate(weight_col):
                for el in weights[k]:
                    if el['coordinates'][0] == i and el['coordinates'][1] == j:
                        el['value'] = weight_val
    return weights
