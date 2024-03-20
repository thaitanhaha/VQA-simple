from flask import Flask, render_template, request
from prediction import test_with_custom_data

app = Flask("VQA")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        question = request.form['text']

        if upload_file == None or question == None:
            return render_template('index.html')
        
        message = test_with_custom_data('custom_data/0.png', str(question))
        return render_template('index.html', message = message)

    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def return_to_home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
