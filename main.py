from flask import Flask, render_template, request, jsonify
# from neural_net import NeuralNetwork, Linear, sigmoid
from python.plot_net import plot_net
from python.plot_net import update_coordinate_values
from python.plot_net import update_weight_values
from python.generate_data import generate_data
from python.generate_data import read_x_row
from python.neural_net import TwoLayerNetwork, Linear, sigmoid
import python.database as db
import pandas as pd
import pickle
import numpy as np
import time

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


def clean_layer(layer):
    '''
    Layer is a numpy array
    '''

    if len(layer) == 1:
        return [layer[0] * 10.0 + 10.0]
    else:
        mi = np.amin(layer)
        ma = np.amax(layer)
        layer = ((layer - mi) / (ma - mi) * 10.0) + 10.0
        layer = [np.round(el, 3) for el in layer]
        return layer


def clean_weights(weight_values):
    '''
    Scale the weights matrix
    '''
    def _scale_weight_matrix(matrix):
        mi = np.amin(matrix)
        ma = np.amax(matrix)
        return ((matrix - mi) / (ma - mi) * 4.0) + 1.0

    return [_scale_weight_matrix(weight_matrix).tolist() for weight_matrix in weight_values]


@app.route('/visualize_data/', methods=['GET', 'POST'])
def visualize_data():
    # Get in number of input neurons
    num_input = int(request.form['input_neurons'])
    num_hidden = int(request.form['hidden_neurons'])

    # Generate data - updates database with data.
    df = generate_data(num_input)

    # Inside plot net, get one row from the dataFrame
    coords, weights = plot_net(num_input, num_hidden, 480, 600)

    # Add something to the values so it doesn't plot zeros.
    layer1 = Linear(n_in=num_input, n_out=num_hidden, activation_function=sigmoid)
    layer2 = Linear(n_in=num_hidden, n_out=1, activation_function=sigmoid)
    nn = TwoLayerNetwork(layers=[layer1, layer2], random_seed=823)

    # Sample one row of x data
    df_x = db.read_key('dataframe')[db.read_key('x_cols')]
    samp = np.random.randint(0, db.read_key('num_obs'))
    x_row = np.array(df_x.iloc[samp, :], ndmin=2)

    # Sample one y observation
    # import pdb; pdb.set_trace()
    df_y = db.read_key('dataframe')["Y"]
    y_row = np.array(df_y.iloc[samp], ndmin=2)

    # Take this single observation and run it through the network to generate
    # values for the layers
    layer_values, prediction = nn.return_forwardpass_info(x_row)
    print("layer_values", layer_values)
    layers_clean = [clean_layer(layer) for layer in layer_values]

    # Simply extract the weight values
    weight_values = nn.return_weights()
    weight_values = clean_weights(weight_values)

    # In the coords and weights "objects", update the weights and coordinates
    # values
    coords = update_coordinate_values(layers_clean, coords)
    weights = update_weight_values(weights, weight_values)

    # Compute loss
    loss = nn.loss(prediction, y_row)

    db.create_key("coordinates", coords)
    db.create_key("weights", weights)
    db.create_key("x_data", x_row)
    db.create_key("layer_values", layer_values)
    db.create_key("prediction", prediction)
    db.create_key("next_grad", loss)
    db.create_key("neural_net", nn)
    db.create_key("next_layer", len(nn.layers))

    # Render the template!
    return render_template('visualize.html',
                           data=db.read_key("coordinates"),
                           weights=db.read_key("weights"))


@app.route('/update_next_neuron_layer/', methods=['GET', 'POST'])
def update_next_neurons():
    print("Next layer:", db.read_key('next_layer'))
    if db.read_key('next_layer') > len(db.read_key('neural_net').layers):
        coords = db.read_key('coordinates')
        return jsonify(data=coords)

    nn = db.read_key("neural_net")
    if db.read_key('next_layer') == 0:
        x_row = read_x_row()
        db.update_key("x_data", x_row)
    else:
        x_row = db.read_key("x_data")
    layers = nn.return_num_layers(x_row)
    layers_clean = [clean_layer(layer) for layer in layers]
    print("layers_clean", layers_clean)
    coords = update_coordinate_values(layers_clean, db.read_key('coordinates'))
    print("Coordinates:", coords)
    db.update_key('coordinates', coords)
    time.sleep(5)
    print("Updated neurons in layer", db.read_key('next_layer')-1)
    return jsonify(data=coords)


@app.route('/update_next_weight_layer/', methods=['GET', 'POST'])
def update_next_weights():
    print("Next layer:", db.read_key('next_layer'))
    if db.read_key('next_layer') < 1:
        weights = db.read_key('weights')
        return jsonify(data=weights)
    nn = db.read_key("neural_net")
    next_grad = db.read_key("next_grad")
    weights_before = db.read_key('weights')
    weight_values_before_clean = nn.update_num_layer_weights(next_grad, num=1)
    weight_values_after_clean = clean_weights(weight_values_before_clean)
    weights = update_weight_values(db.read_key('weights'), weight_values_after_clean)
    db.update_key('weights', weights)
    time.sleep(1)
    print("Updating weights before layer", db.read_key('next_layer')+1)
    return jsonify(data=weights)



@app.route('/visualize_neurons/', methods=['GET', 'POST'])
def visualize_neurons():
    input_neurons = request.form['input_neurons']
    hidden_neurons = request.form['hidden_neurons']
    num_input = int(input_neurons)
    num_hidden = int(hidden_neurons)
    xs_ys, weights = plot_net(num_input, num_hidden, 480, 600)
    db.create_key('data', xs_ys)
    db.create_key('weights', weights)
    db.delete_key('next_weight_layer')
    return render_template('visualize_neurons.html',
                           data=db.read_key('data'),
                           weights=db.read_key('weights'))


@app.route('/reset_data/', methods=['GET', 'POST'])
def reset_data():
    print("The reset data button was clicked")
    db.delete_all_keys()
    if not db.read_key("data"):
        db.create_key("data", [])
    if not db.read_key("weights"):
        db.create_key("weights", [])
    if not db.read_key("coordinates"):
        db.create_key("coordinates", [])
    if db.read_key("next_weight_layer"):
        db.delete_key("next_weight_layer")
    return render_template('index.html',
                           coordinates=db.read_key('coordinates'),
                           data=db.read_key('data'),
                           weights=db.read_key('weights'))


if __name__ == '__main__':
    app.run(debug=True, port=5001)
