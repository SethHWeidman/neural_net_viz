from flask import Flask, render_template, request
from check_number import check_number

app = Flask(__name__)

@app.route('/visualize/')
def hello_world():
    return render_template('index_data_imputed.html')

@app.route('/visualize_neurons/', methods=['GET', 'POST'])
def visualize_neurons():
    number = request.form['number']
    number_final = int(number) * 5
    ys = [50, 150, 250, 350]
    xs = [number_final for y in ys]
    zip_x_y = [dict({"x": i, "y": j}) for i, j in zip(xs, ys)]
    return render_template('index_data_imputed.html',
                           data=zip_x_y)

@app.route('/check_number/', methods=['POST'])
def check_number_page():
    number = request.form['number_to_check']
    number_message = check_number(number)
    return render_template('number_answer.html', number_message=number_message)

if __name__ == '__main__':
    app.run(debug=True)
