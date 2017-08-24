from flask import Flask, render_template, request, jsonify
from neural_net import NeuralNetwork, Linear, sigmoid
from python.plot_net import plot_net
from python.plot_net import update_coordinate_values
from python.plot_net import update_weight_values
from python.generate_data import generate_data
from neural_net import TwoLayerNetwork, Linear
import python.database as db
import pandas as pd
import numpy as np
import time

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


def clean_layers(layers):
    layers = [np.round(layer, 3) for layer in layers]
    return [layer.tolist() for layer in layers]


def clean_weights(weights):

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

    df_x = pd.read_json(db.find_key('dataframe'))[db.find_key('x_cols')]
    x_row = df_x.sample(random_state=823)

    layers = nn.return_layer_outputs(x_row)
    layers_clean = clean_layers(layers)

    weight_values = nn.return_weights(x_row)
    weight_values = [clean_weights(weight_matrix).tolist() for weight_matrix in weight_values]

    coords = update_coordinate_values(layers, coords)
    # import pdb; pdb.set_trace()
    weights = update_weight_values(weights, weight_values)


    db.add_key("coordinates", coords)
    db.add_key("weights", weights)

    return render_template('visualize.html',
                           data=db.find_key("coordinates"),
                           weights=db.find_key("weights"))

@app.route('/update_one/', methods=['GET', 'POST'])
def update_one():
    coords = db.find_key("coordinates")
    for item in coords:
        if item['neuron'] == 0  and item['layer'] == 0:
            item['value'] = 1
    time.sleep(3)
    return jsonify(data=coords)



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
    if not db.find_key("dimensions"):
        db.add_key("dimensions", {'height': 480, 'width': 600})
    if not db.find_key("data"):
        db.add_key("data", [])
    if not db.find_key("weights"):
        db.add_key("weights", [])
    return render_template('index.html',
                           dimensions=db.find_key('dimensions'),
                           data=db.find_key('data'),
                           weights=db.find_key('weights'))


if __name__ == '__main__':
    app.run(debug=True)
