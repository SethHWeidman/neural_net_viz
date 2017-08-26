from flask import Flask, render_template, request, jsonify
# from neural_net import NeuralNetwork, Linear, sigmoid
from python.plot_net import plot_net
from python.plot_net import update_coordinate_values
from python.plot_net import update_weight_values
from python.generate_data import generate_data
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

    if type(layer) == np.float64:
        return [15.0]
    else:
        mi = np.amin(layer)
        ma = np.amax(layer)
        layer = ((layer - mi) / (ma - mi) * 10.0) + 10.0
        layer = [np.round(el, 3) for el in layer]
        return layer


def clean_weights(weights):
    '''
    Scale the weights matrix
    '''
    mi = np.amin(weights)
    ma = np.amax(weights)
    return (weights - mi) / (ma - mi) + 0.5


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
    df_x = pd.read_json(db.find_key('dataframe'))[db.find_key('x_cols')]
    samp = np.random.randint(0, db.find_key('num_obs'))
    x_row = df_x.iloc[samp, :]

    # Sample one y observation
    # import pdb; pdb.set_trace()
    df_y = pd.read_json(db.find_key('dataframe'))["Y"]
    y_row = df_y.iloc[samp]

    # Take this single observation and run it through the network to generate
    # values for the layers
    layers, prediction = nn.return_forwardpass_info(x_row)
    layers_clean = [clean_layer(layer) for layer in layers]

    # Simply extract the weight values
    weight_values = nn.return_weights()
    weight_values = [clean_weights(weight_matrix).tolist() for weight_matrix in weight_values]

    # In the coords and weights "objects", update the weights and coordinates
    # values
    coords = update_coordinate_values(layers_clean, coords)
    weights = update_weight_values(weights, weight_values)

    # Add these to the database for further use
    db.add_key("coordinates", coords)
    db.add_key("weights", weights)

    # Compute loss
    loss = nn.loss(prediction, y_row)

    pickle.dump(nn, open("nn_data.p", "wb"))
    
    db.add_key("coordinates", coords)
    db.add_key("weights", weights)
    db.add_key("loss", loss.tolist())
    db.add_key("neural_net", )

    # Render the template!
    return render_template('visualize.html',
                           data=db.find_key("coordinates"),
                           weights=db.find_key("weights"))


@app.route('/update_one/', methods=['GET', 'POST'])
def update_one():
    coords = db.find_key("coordinates")
    for item in coords:
        if item['neuron'] == 0  and item['layer'] == 0:
            item['value'] = 30.0
    time.sleep(3)
    return jsonify(data=coords)


@app.route('/update_next_weight_layer/', methods=['GET', 'POST'])
def update_next_layer():
    loss = db.find_key("loss")
    weight_values = nn.update_num_layer_weights(loss, num=1)
    weights = update_weight_values(db.find_key('weights'), weight_values)
    time.sleep(3)
    return jsonify(data=weights)



@app.route('/visualize_neurons/', methods=['GET', 'POST'])
def visualize_neurons():
    input_neurons = request.form['input_neurons']
    hidden_neurons = request.form['hidden_neurons']
    num_input = int(input_neurons)
    num_hidden = int(hidden_neurons)
    xs_ys, weights = plot_net(num_input, num_hidden, 480, 600)
    db.add_key('data', xs_ys)
    db.add_key('weights', weights)
    return render_template('visualize_neurons.html',
                           data=db.find_key('data'),
                           weights=db.find_key('weights'))



# @app.route('/update_neurons/', methods=['GET', 'POST'])
# def initialize_weights():
    # Take the weights from the Network and change the thicknesses of the lines
    # accordingly


# def update_neurons():
    # Add a button that loads some SkLearn data into the database.

    # use the current weights of the network to update the sizes of the neurons.
    # height = db.find_key('dimensions')['height']
    # width = db.find_key('dimensions')['width']
    # xs_ys, weights = plot_net(num_input, num_hidden, height, width)
    # db.update_key('data', xs_ys)
    # db.update_key('weights', weights)
    # print(weights)
    # return render_template('index.html',
    #                        dimensions=db.find_key('dimensions'),
    #                        data=db.find_key('data'),
    #                        weights=db.find_key('weights'))


@app.route('/reset_data/', methods=['GET', 'POST'])
def reset_data():
    print("The reset data button was clicked")
    db.delete_all_docs()
    if not db.find_key("data"):
        db.add_key("data", [])
    if not db.find_key("weights"):
        db.add_key("weights", [])
    if not db.find_key("coordinates"):
        db.add_key("coordinates", [])
    return render_template('index.html',
                           coordinates=db.find_key('coordinates'),
                           data=db.find_key('data'),
                           weights=db.find_key('weights'))


if __name__ == '__main__':
    app.run(debug=True)
