from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename
import os
import librosa
import numpy as np
from keras.models import load_model
import mlflow.keras
import numpy as np
from io import BytesIO

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_audio(audio, sr):
    target_sr = 44100
    if sr != target_sr:
        audio = librosa.resample(audio, sr, target_sr)
        sr = target_sr
    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio)
    return audio, sr

def load_trained_model(model_path):
    model = load_model(model_path)
    return model

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            model = mlflow.keras.load_model("mlartifacts/0/b5988715bc0346a1b644dcc31695b762/artifacts/models/trained_model.h5")
            mixture_audio, sr = librosa.load(file_path, sr=None)
            mixture_audio, sr = preprocess_audio(mixture_audio, sr)
            
            separated_sources = model.predict(np.expand_dims(mixture_audio, axis=0))
            
            # Create an in-memory file
            output = BytesIO()
            np.savez(output, sources=separated_sources, sr=sr)
            output.seek(0)
            
            return send_file(output, as_attachment=True, attachment_filename='output.npz')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.run(port=5002, debug=True)
