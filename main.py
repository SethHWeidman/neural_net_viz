from flask import Flask, render_template, request, jsonify
from neural_net import NeuralNetwork, Linear, sigmoid
from plot_net import plot_net
import database as db

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/visualize_data/', methods=['GET', 'POST'])
def visualize_data():
    # Get in number of input neurons

    # Generate data

    # Add something to the values so it doesn't plot zeros.

    # Feed these values into the input layer
        # On hover, show the value of the whole data point, and just the neuron value.

    # Add the weights back in so the thicknesses are adjusted based on their values.

    # Have a button on the page that you can click that updates the next layer in the
    # net

    # Have a button you can click that is enabled after the last layer is updated that
    # lets you backpropogate the weights

    # Do all of this with transitions


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
