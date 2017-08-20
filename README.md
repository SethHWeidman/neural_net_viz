# neural_net_viz
A simple neural net visualization using Flask and D3

## Instructions
Run `python main.py` and navigate to `http://127.0.0.1:5000/` to see the app in action.

## To-do
Make a function that visualizes the weights:

1. Start with values of the initial neurons.
N circles on an SVG.
2. Start with values of the weights: the 3 x 4 weights, for example.
3. Visualize the 4 x 1 weights.
4. Visualize the output.
5. Compare to the actual value.

### Updating the weights

Load in a neural net object.
Hard part: define a function to step just one step forward in the neural net.
No! Define different functions for different steps:
1. A function for computing inputs to neurons.
2. A function for computing values of neurons. If the values have already been computed, don't do anything.
  Make a button underneath each layer that computes the values for that layer.
3.
The way to visualize:

Size of neurons: size of circles.
Lines connecting: thickness = weights.
Hover over neuron, see weights. Hover over neuron, see prior layer too.
Need to store how neurons are connected. 
