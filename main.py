from flask import Flask, render_template, request, jsonify
from neural_net import NeuralNetwork, Linear, sigmoid
from plot_net import plot_net
import database as db

app = Flask(__name__)


@app.route('/')
def hello_world():

    if not db.find_key("dimensions"):
        db.add_key("dimensions", {'height': 480, 'width': 600})
    if not db.find_key("data"):
        db.add_key("data", [])
    return render_template('index.html',
                           dimensions=db.find_key('dimensions'),
                           data=db.find_key('data'))


@app.route('/visualize_neurons/', methods=['GET', 'POST'])
def visualize_neurons():
    print("Data upon starting visual:")
    print(db.find_all_docs())
    print()
    input_neurons = request.form['input_neurons']
    hidden_neurons = request.form['hidden_neurons']
    num_input = int(input_neurons)
    num_hidden = int(hidden_neurons)
    height = db.find_key('dimensions')['height']
    width = db.find_key('dimensions')['width']
    xs_ys = plot_net(num_input, num_hidden, height, width)
    db.update_key('data', xs_ys)
    print("Data after update:")
    print(db.find_key('data'))
    return render_template('index.html',
                           dimensions=db.find_key('dimensions'),
                           data=db.find_key('data'))


@app.route('/reset_data/', methods=['GET', 'POST'])
def multiply():
    print("The reset data button was clicked")
    db.delete_all_docs()
    if not db.find_key("dimensions"):
        db.add_key("dimensions", {'height': 480, 'width': 600})
    if not db.find_key("data"):
        db.add_key("data", [])
    return render_template('index.html',
                           dimensions=db.find_key('dimensions'),
                           data=db.find_key('data'))

if __name__ == '__main__':
    app.run(debug=True)
