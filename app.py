from flask import Flask, render_template, request
from prediction import test_with_custom_data
from PIL import Image
import numpy as np
from io import BytesIO

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
        
        img_bytes = uploaded_file.read()
        img = Image.open(BytesIO(img_bytes))
        message = test_with_custom_data(img, str(question))
        # message = test_with_custom_data('custom_data/0.png', str(question))

        return render_template('index.html', message = message)

    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def return_to_home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
