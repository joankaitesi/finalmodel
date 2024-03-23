import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import mlflow.keras
import mlflow
from sklearn.preprocessing import LabelEncoder

# Set the tracking URI to the local directory
mlflow.set_tracking_uri("http://127.0.0.1:5000")
dataset_path = "C:/Users/DELL/Desktop/model/DSD100subset"
img_height, img_width = 128, 128  # Spectrogram image dimensions

def preprocess_audio(audio, sr):
    target_sr = 44100
    if sr != target_sr:
        audio = librosa.resample(audio, sr, target_sr)
        sr = target_sr

    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio)

    return audio, sr

def save_spectrogram_as_image(audio, sr, output_path):
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_spectrogram_images(song_name):
    output_dir = f"{song_name}_spectrograms"
    os.makedirs(output_dir, exist_ok=True)

    mixture_path = os.path.join(dataset_path, "Mixtures", "Dev", song_name, "mixture.wav")
    mixture, sr = librosa.load(mixture_path, sr=None)
    mixture, sr = preprocess_audio(mixture, sr)
    save_spectrogram_as_image(mixture, sr, os.path.join(output_dir, "mixture_spectrogram.png"))

    sources_path = os.path.join(dataset_path, "Sources", "Dev", song_name)
    source_files = os.listdir(sources_path)
    for source_file in source_files:
        source_name = os.path.splitext(source_file)[0]
        source_path = os.path.join(sources_path, source_file)
        source, sr = librosa.load(source_path, sr=None)
        source, sr = preprocess_audio(source, sr)
        save_spectrogram_as_image(source, sr, os.path.join(output_dir, f"{source_name}_spectrogram.png"))

def load_data(song_names):
    X, y = [], []
    for song_name in song_names:
        spectrogram_dir = f"{song_name}_spectrograms"
        spectrogram_paths = [os.path.join(spectrogram_dir, f) for f in os.listdir(spectrogram_dir) if f.endswith(".png")]
        for spectrogram_path in spectrogram_paths:
            spectrogram = plt.imread(spectrogram_path)
            spectrogram_resized = np.resize(spectrogram, (img_height, img_width, 3))
            X.append(spectrogram_resized)
            y.append(int(song_name[-1]))
    return np.array(X), np.array(y)

def build_model(img_height, img_width, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')
    
    datagen.fit(X_train)
    
    history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))
    
    return model, history

def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    return test_loss, test_acc

def save_model(model, model_save_path):
    model.save(model_save_path + ".keras")
    mlflow.keras.save_model(model, model_save_path)

if __name__ == "__main__":
    # Define parameters
    num_classes = 2  # Update with actual number of classes
    song_names = ["song1", "song2"]  # Update with actual song names

    # Preprocess and save spectrogram images
    for song_name in song_names:
        save_spectrogram_images(song_name)

    # Load data
    X, y = load_data(song_names)
    y = y - 1  # Map labels to start from 0

    # Check unique labels
    unique_labels = np.unique(y)
    print("Unique labels in y_train after mapping:", unique_labels)

    # Update num_classes based on the number of unique labels
    num_classes = len(unique_labels)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess data
    X_train = X_train.astype('float32') / 255
    X_val = X_val.astype('float32') / 255
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)

    # Load testing data
    X_test, y_test = load_data(song_names)
    y_test = y_test - 1  # Map labels to start from 0

    # Check unique labels in testing data
    unique_labels_test = np.unique(y_test)

    # Build model
    model = build_model(img_height, img_width, num_classes)

    # Train model
    model, history = train_model(model, X_train, y_train, X_val, y_val)

    # Save model
    # model_save_path = "C:/Users/DELL/Desktop/model/trained_model_v2"
    # save_model(model, model_save_path)

    # Evaluate model
    X_test = X_test.astype('float32') / 255
    y_test_encoded = LabelEncoder().fit_transform(y_test).reshape(-1)  # Encode and reshape y_test
    test_loss, test_acc = evaluate_model(model, X_test, to_categorical(y_test_encoded, num_classes))
    print(f'Test accuracy: {test_acc}')
# MLflow Tracking
    mlflow.keras.log_model(model, "models/trained_model.h5")

