from flask import Flask, render_template, request
from check_number import check_number

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/check_number/', methods=['POST'])
def check_number_page():
    number = request.form['number']
    number_message = check_number(number)
    return render_template('number_answer.html', number_message=number_message)

if __name__ == '__main__':
    app.run(debug=True)
