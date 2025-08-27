from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "Empty file uploaded"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Read and clean the uploaded CSV
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    df.dropna(inplace=True)

    # Drop the label if it exists (in case the user uploads training-like data)
    if 'Label' in df.columns:
        df.drop('Label', axis=1, inplace=True)

    # Predict using the model
    predictions = model.predict(df)

    # Count predictions
    total = len(predictions)
    benign = (predictions == 0).sum()
    attack = (predictions == 1).sum()

    return render_template('result.html', total=total, benign=benign, attack=attack)

if __name__ == '__main__':
    app.run(debug=True)
