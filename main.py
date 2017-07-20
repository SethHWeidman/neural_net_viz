from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/visualize_neurons/', methods=['GET', 'POST'])
def visualize_neurons():
    number = request.form['number']
    number_final = int(number) * 5
    ys = [50, 150, 250, 350]
    xs = [number_final for y in ys]
    zip_x_y = [dict({"x": i, "y": j}) for i, j in zip(xs, ys)]
    return render_template('index.html',
                           data=zip_x_y)

if __name__ == '__main__':
    app.run(debug=True)
