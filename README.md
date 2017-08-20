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

In the `plot_net.py` function, the lines are now calculated that connect each neuron.

To-do:
1. Adjust the thickness based on the values (easy).
2. Hook all this up with the neural_net function.  
