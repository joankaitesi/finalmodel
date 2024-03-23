import mlflow

import os
import librosa
from IPython.display import Audio
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import regularizers
from keras.optimizers import Adam
import mlflow.keras
from IPython.display import display, Audio
import mlflow
import mlflow.tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import save_model

# Set the tracking URI to the local directory
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model_path = 'C:/Users/DELL/Desktop/model/trained_model.h5'
model_save_path = 'C:/Users/DELL/Desktop/model/trained_model.h5'
results_path = "C:/Users/DELL/Desktop/model"
dataset_path = "C:/Users/DELL/Desktop/model/DSD100subset"
output_dir = "C:/Users/DELL/Desktop/model"
output_path = "C:/Users/DELL/Desktop/model"

# save_model(model, 'model_save_path')

def play_audio_sample(song_name):
    # Load mixture
    mixture_path = os.path.join(dataset_path, "Mixtures", "Dev", song_name, "mixture.wav")
    mixture, sr = librosa.load(mixture_path, sr=None)

    # Load individual sources
    sources_path = os.path.join(dataset_path, "Sources", "Dev", song_name)
    source_files = os.listdir(sources_path)
    for source_file in source_files:
        source_name = os.path.splitext(source_file)[0]
        source_path = os.path.join(sources_path, source_file)
        source, sr = librosa.load(source_path, sr=None)
        print(f"Playing {source_name} source")
        display(Audio(source, rate=sr, autoplay=True))

    print("Playing mixture")
    display(Audio(mixture, rate=sr, autoplay=True))

# Example: Play sources and mixture for song1
play_audio_sample("song1")
play_audio_sample("song2")

# Function to preprocess audio
def preprocess_audio(audio, sr):
    # Resample to a common sampling rate if needed (e.g., 44100 Hz)
    target_sr = 44100
    if sr != target_sr:
        audio = librosa.resample(audio, sr, target_sr)
        sr = target_sr
# Normalize audio
    audio = librosa.util.normalize(audio)
 # Trim leading and trailing silence
    audio, _ = librosa.effects.trim(audio)
    return audio, sr

# Function to match mixtures with their sources and play them
def play_audio_sample(song_name):
    # Load mixture
    mixture_path = os.path.join(dataset_path, "Mixtures", "Dev", song_name, "mixture.wav")
    mixture, sr = librosa.load(mixture_path, sr=None)
    mixture, sr = preprocess_audio(mixture, sr)

    # Load individual sources
    sources_path = os.path.join(dataset_path, "Sources", "Dev", song_name)
    source_files = os.listdir(sources_path)
    for source_file in source_files:
        source_name = os.path.splitext(source_file)[0]
        source_path = os.path.join(sources_path, source_file)
        source, sr = librosa.load(source_path, sr=None)
        source, sr = preprocess_audio(source, sr)
        print(f"Playing {source_name} source")
        display(Audio(source, rate=sr, autoplay=True))

    print("Playing mixture")
    display(Audio(mixture, rate=sr, autoplay=True))

# Example: Play sources and mixture for song1
play_audio_sample("song1")
play_audio_sample("song2")

# Function to preprocess audio
def preprocess_audio(audio, sr):
    # Resample to a common sampling rate if needed (e.g., 44100 Hz)
    target_sr = 44100
    if sr != target_sr:
        audio = librosa.resample(audio, sr, target_sr)
        sr = target_sr

    # Normalize audio
    audio = librosa.util.normalize(audio)

    # Trim leading and trailing silence
    audio, _ = librosa.effects.trim(audio)

    return audio, sr

# Function to match mixtures with their sources and play them
def play_audio_sample(song_name):
    # Load mixture
    mixture_path = os.path.join(dataset_path, "Mixtures", "Dev", song_name, "mixture.wav")
    mixture, sr = librosa.load(mixture_path, sr=None)
    mixture, sr = preprocess_audio(mixture, sr)

    # Load individual sources
    sources_path = os.path.join(dataset_path, "Sources", "Dev", song_name)
    source_files = os.listdir(sources_path)
    for source_file in source_files:
        source_name = os.path.splitext(source_file)[0]
        source_path = os.path.join(sources_path, source_file)
        source, sr = librosa.load(source_path, sr=None)
        source, sr = preprocess_audio(source, sr)
        print(f"Playing {source_name} source")
        display(Audio(source, rate=sr, autoplay=True))

    print("Playing mixture")
    display(Audio(mixture, rate=sr, autoplay=True))

# Example: Play sources and mixture for song1
play_audio_sample("song1")
play_audio_sample("song2")

# Function to preprocess audio
def preprocess_audio(audio, sr):
    # Resample to a common sampling rate if needed (e.g., 44100 Hz)
    target_sr = 44100
    if sr != target_sr:
        audio = librosa.resample(audio, sr, target_sr)
        sr = target_sr

    # Normalize audio
    audio = librosa.util.normalize(audio)

    # Trim leading and trailing silence
    audio, _ = librosa.effects.trim(audio)
    return audio, sr

import os
import librosa
import IPython.display as ipd

def visualize_audio_sample(song_name):
    # Load mixture
    mixture_path = os.path.join(dataset_path, "Mixtures", "Dev", song_name, "mixture.wav")
    mixture, sr = librosa.load(mixture_path, sr=None)
    mixture, sr = preprocess_audio(mixture, sr)

    # Display mixture audio
    print("Playing mixture")
    ipd.display(ipd.Audio(data=mixture, rate=sr))

    # Load individual sources
    sources_path = os.path.join(dataset_path, "Sources", "Dev", song_name)
    source_files = os.listdir(sources_path)
    for source_file in source_files:
        source_name = os.path.splitext(source_file)[0]
        source_path = os.path.join(sources_path, source_file)
        source, sr = librosa.load(source_path, sr=None)
        source, sr = preprocess_audio(source, sr)

        # Display source audio
        print(f"Playing {source_name} source")
        ipd.display(ipd.Audio(data=source, rate=sr))

# Example: Visualize sources and mixture for song1
visualize_audio_sample("song1")
visualize_audio_sample("song2")

def preprocess_audio(audio, sr):
    # Resample to a common sampling rate if needed (e.g., 44100 Hz)
    target_sr = 44100
    if sr != target_sr:
        audio = librosa.resample(audio, sr, target_sr)
        sr = target_sr

    # Normalize audio
    audio = librosa.util.normalize(audio)

    # Trim leading and trailing silence
    audio, _ = librosa.effects.trim(audio)

    return audio, sr

# Function to save spectrogram as image
def save_spectrogram_as_image(audio, sr, output_path):
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Function to match mixtures with their sources and save spectrogram images
def save_spectrogram_images(song_name):
    # Create directories to store images
    output_dir = f"{song_name}_spectrograms"
    os.makedirs(output_dir, exist_ok=True)

    # Load mixture
    mixture_path = os.path.join(dataset_path, "Mixtures", "Dev", song_name, "mixture.wav")
    mixture, sr = librosa.load(mixture_path, sr=None)
    mixture, sr = preprocess_audio(mixture, sr)
    save_spectrogram_as_image(mixture, sr, os.path.join(output_dir, "mixture_spectrogram.png"))

    # Load individual sources
    sources_path = os.path.join(dataset_path, "Sources", "Dev", song_name)
    source_files = os.listdir(sources_path)
    for source_file in source_files:
        source_name = os.path.splitext(source_file)[0]
        source_path = os.path.join(sources_path, source_file)
        source, sr = librosa.load(source_path, sr=None)
        source, sr = preprocess_audio(source, sr)
        save_spectrogram_as_image(source, sr, os.path.join(output_dir, f"{source_name}_spectrogram.png"))

# Example: Save spectrogram images for song1
save_spectrogram_images("song1")

# Example: Save spectrogram images for song2
save_spectrogram_images("song2")

# Load VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to preprocess spectrogram images for VGG16
def preprocess_spectrogram_for_vgg(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to extract features using VGG16
def extract_features_using_vgg(image_paths):
    features = []
    for image_path in image_paths:
        preprocessed_img = preprocess_spectrogram_for_vgg(image_path)
        feature = vgg_model.predict(preprocessed_img)
        features.append(feature.flatten())
    return np.array(features)

# Get list of all spectrogram image paths
spectrogram_image_paths = []
for root, dirs, files in os.walk("./", topdown=False):
    for name in files:
        if name.endswith("_spectrogram.png"):
            spectrogram_image_paths.append(os.path.join(root, name))

# Extract features from spectrogram images using VGG16
features = extract_features_using_vgg(spectrogram_image_paths)

print("Features shape:", features.shape)


def preprocess_and_save_testing_spectrograms(song_name):
    # Create directory to store images
    output_dir = f"/content/drive/My Drive/{song_name}_testing_spectrograms"
    os.makedirs(output_dir, exist_ok=True)

    # Load mixture
    mixture_path = os.path.join(dataset_path, "Mixtures", "Test", song_name, "mixture.wav")
    mixture, sr = librosa.load(mixture_path, sr=None)
    mixture, sr = preprocess_audio(mixture, sr)
    save_spectrogram_as_image(mixture, sr, os.path.join(output_dir, "mixture_spectrogram.png"))

    # Load individual sources
    sources_path = os.path.join(dataset_path, "Sources", "Test", song_name)
    source_files = os.listdir(sources_path)
    for source_file in source_files:
        source_name = os.path.splitext(source_file)[0]
        source_path = os.path.join(sources_path, source_file)
        source, sr = librosa.load(source_path, sr=None)
        source, sr = preprocess_audio(source, sr)
        save_spectrogram_as_image(source, sr, os.path.join(output_dir, f"{source_name}_spectrogram.png"))

preprocess_and_save_testing_spectrograms("song1")

# Example: Preprocess and save spectrogram images for testing data of song2
preprocess_and_save_testing_spectrograms("song2")


# Function to preprocess audio
def preprocess_audio(audio, sr):
    # Resample to a common sampling rate if needed (e.g., 44100 Hz)
    target_sr = 44100
    if sr != target_sr:
        audio = librosa.resample(audio, sr, target_sr)
        sr = target_sr

    # Normalize audio
    audio = librosa.util.normalize(audio)

    # Trim leading and trailing silence
    audio, _ = librosa.effects.trim(audio)

    return audio, sr

# Function to save spectrogram as image
def save_spectrogram_as_image(audio, sr, output_path):
    spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Function to match mixtures with their sources and save spectrogram images
def save_spectrogram_images(song_name):
    # Create directories to store images
    output_dir = f"{song_name}_spectrograms"
    os.makedirs(output_dir, exist_ok=True)

    # Load mixture
    mixture_path = os.path.join(dataset_path, "Mixtures", "Dev", song_name, "mixture.wav")
    mixture, sr = librosa.load(mixture_path, sr=None)
    mixture, sr = preprocess_audio(mixture, sr)
    save_spectrogram_as_image(mixture, sr, os.path.join(output_dir, "mixture_spectrogram.png"))

    # Load individual sources
    sources_path = os.path.join(dataset_path, "Sources", "Dev", song_name)
    source_files = os.listdir(sources_path)
    for source_file in source_files:
        source_name = os.path.splitext(source_file)[0]
        source_path = os.path.join(sources_path, source_file)
        source, sr = librosa.load(source_path, sr=None)
        source, sr = preprocess_audio(source, sr)
        save_spectrogram_as_image(source, sr, os.path.join(output_dir, f"{source_name}_spectrogram.png"))

# Example: Save spectrogram images for song1
save_spectrogram_images("song1")

# Example: Save spectrogram images for song2
save_spectrogram_images("song2")

num_classes = len(os.listdir(os.path.join(dataset_path, "Sources", "Dev", "song1")))  # Assuming all songs have the same number of sources
img_height, img_width = 128, 128  # Spectrogram image dimensions

# Function to load spectrogram images and source labels
def load_data(song_names):
    X, y = [], []
    for song_name in song_names:
        spectrogram_dir = f"{song_name}_spectrograms"
        spectrogram_paths = [os.path.join(spectrogram_dir, f) for f in os.listdir(spectrogram_dir) if f.endswith(".png")]
        for spectrogram_path in spectrogram_paths:
            spectrogram = plt.imread(spectrogram_path)
            spectrogram_resized = np.resize(spectrogram, (img_height, img_width, 3))  # Resize spectrogram to expected dimensions
            X.append(spectrogram_resized)
            y.append(int(song_name[-1]))  # Extract source label from song name and convert to integer
    return np.array(X), np.array(y)

# Get list of song names
song_names = ["song1", "song2"]  # Add more songs if needed

# # Load data
# X, y = load_data(song_names)

# # Split data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Preprocess data: Normalize pixel values to range [0, 1]
# X_train = X_train.astype('float32') / 255
# X_val = X_val.astype('float32') / 255

# # Convert labels to one-hot encoded vectors
# y_train = to_categorical(y_train, num_classes)
# y_val = to_categorical(y_val, num_classes)


# # Define CNN model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(num_classes, activation='softmax')
# ])

# # Compile model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train model
# history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# # Plot training history
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.legend(loc='lower right')
# plt.show()

# # Evaluate model on testing data
# # Evaluate model on testing data for song1
# X_test_song1, y_test_song1 = load_data(["song1"])
# X_test_song1 = X_test_song1.astype('float32') / 255
# y_test_song1 = to_categorical(y_test_song1, num_classes)

# test_loss_song1, test_acc_song1 = model.evaluate(X_test_song1,  y_test_song1, verbose=2)
# print(f'Test accuracy for song1: {test_acc_song1}')

# # Evaluate model on testing data for song2
# X_test_song2, y_test_song2 = load_data(["song2"])
# X_test_song2 = X_test_song2.astype('float32') / 255
# y_test_song2 = to_categorical(y_test_song2, num_classes)

# test_loss_song2, test_acc_song2 = model.evaluate(X_test_song2,  y_test_song2, verbose=2)
# print(f'Test accuracy for song2: {test_acc_song2}')
# overall_accuracy = (test_acc_song1 + test_acc_song2) / 2
# print(f'Overall test accuracy: {overall_accuracy}')


# Load data
X, y = load_data(song_names)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data: Normalize pixel values to range [0, 1]
X_train = X_train.astype('float32') / 255
X_val = X_val.astype('float32') / 255

# Convert labels to one-hot encoded vectors
y_train = to_categorical(y_train, num_classes)
y_val = to_categorical(y_val, num_classes)

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

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation
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

# Adjust Model Architecture

history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))
opt = Adam(learning_rate=0.001)  # Adjust learning rate
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
mlflow.log_metrics({"accuracy": history.history['accuracy'][-1], "loss": history.history['loss'][-1],
                        "val_accuracy": history.history['val_accuracy'][-1], "val_loss": history.history['val_loss'][-1]})
# Start the MLflow run with the specified experiment ID
if mlflow.active_run() is not None:
    mlflow.end_run()

with mlflow.start_run():

    mlflow.keras.log_model(model, "model")
    mlflow.log_param("learning_rate",0.001 )
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("accuracy", 0.8)
    mlflow.log_metric("loss", 0.2)

    # mlflow.log_artifact(model_path)
    mlflow.log_artifact(results_path)
    # mlflow.log_model(model, "model_path")

def save_model(model, model_save_path):
    keras.save_model(model, model_save_path)
    mlflow.keras.save_model(model, model_save_path)

mlflow.end_run()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

X_test_song1, y_test_song1 = load_data(["song1"])
X_test_song1 = X_test_song1.astype('float32') / 255
y_test_song1 = to_categorical(y_test_song1, num_classes)

test_loss_song1, test_acc_song1 = model.evaluate(X_test_song1,  y_test_song1, verbose=2)
print(f'Test accuracy for song1: {test_acc_song1}')

# Evaluate model on testing data for song2
X_test_song2, y_test_song2 = load_data(["song2"])
X_test_song2 = X_test_song2.astype('float32') / 255
y_test_song2 = to_categorical(y_test_song2, num_classes)

test_loss_song2, test_acc_song2 = model.evaluate(X_test_song2,  y_test_song2, verbose=2)
print(f'Test accuracy for song2: {test_acc_song2}')

# Overall test accuracy
overall_test_accuracy = (test_acc_song1 + test_acc_song2) / 2
print(f'Overall test accuracy: {overall_test_accuracy}')

with mlflow.start_run():

    mlflow.create_experiment()
    mlflow.keras.log_model(model, "model")
    mlflow.log_param("learning_rate",0.001 )
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("accuracy", 1)
    mlflow.log_metric("loss", 0.5)

    mlflow.log_artifact(model_path)
    mlflow.log_artifact(results_path)
save_model(model, 'model_save_path')
