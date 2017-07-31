from flask import Flask, render_template, request
from neural_net import NeuralNetwork, Linear, sigmoid
from plot_net import plot_net

app = Flask(__name__)

HEIGHT = 400
WIDTH = 600

data = {}
data['xs_ys'] = None
data['mult_num'] = 1

@app.route('/')
def hello_world():
    layer1 = Linear(n_in=3, n_out=4, activation_function=sigmoid)
    layer2 = Linear(n_in=4, n_out=1, activation_function=sigmoid)
    nn = NeuralNetwork(layers=[layer1, layer2], random_seed=1)
    data['xs_ys'] = None
    data['mult_num'] = 1
    return render_template('index.html',
                           height=HEIGHT,
                           width=WIDTH)

@app.route('/visualize_neurons/', methods=['GET', 'POST'])
def visualize_neurons():
    print("The button has been clicked")
    input_neurons = request.form['input_neurons']
    hidden_neurons = request.form['hidden_neurons']
    num_input = int(input_neurons)
    num_hidden = int(hidden_neurons)
    xs_ys = plot_net(num_input, num_hidden, HEIGHT, WIDTH)
    data['xs_ys'] = xs_ys
    return render_template('index.html',
                           height=HEIGHT,
                           width=WIDTH,
                           data=xs_ys)

@app.route('/multiply_number/', methods=['GET', 'POST'])
def multiply():
    data['mult_num'] *= 2
    return render_template('index.html',
                           height=HEIGHT,
                           width=WIDTH,
                           data=data['xs_ys'],
                           mult_num=data['mult_num'])


if __name__ == '__main__':
    app.run(debug=True)
