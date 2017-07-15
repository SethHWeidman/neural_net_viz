from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/check_number/', methods=['POST'])
def check_number():
    return "The form was submitted"

if __name__ == '__main__':
    app.run(debug=True)
