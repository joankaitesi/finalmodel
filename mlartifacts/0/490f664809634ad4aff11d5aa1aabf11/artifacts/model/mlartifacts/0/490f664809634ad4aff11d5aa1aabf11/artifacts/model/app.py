from flask import Flask, request, jsonify
import mlflow.keras
import numpy as np

app = Flask(__name__)

# Load the trained model using MLflow
model = mlflow.keras.load_model("file:///C/Users/DELL/Desktop/model/trained_model.h5")

# Define a route for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Get the spectrogram image from the request
    spectrogram_image = request.files['spectrogram']

    # Preprocess the image
    img = image.load_img(spectrogram_image, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make prediction
    prediction = model.predict(img_array)

    # Decode prediction
    predicted_class = np.argmax(prediction)
    
    # Return the predicted class
    return jsonify({"predicted_class": predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
