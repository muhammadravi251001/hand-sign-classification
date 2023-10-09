import zipfile
import os
import sklearn
import shutil
import requests
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from google.colab import files
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

# Download the dataset zip file
url = "https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip"
response = requests.get(url)
with open("rockpaperscissors.zip", 'wb') as f:
    f.write(response.content)

# Extract the zip file
with zipfile.ZipFile("rockpaperscissors.zip", 'r') as zip_ref:
    zip_ref.extractall()

def create_directory_with_validation(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

base_dir = "rockpaperscissors"
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
rock_img_dir = os.path.join(base_dir, 'rock')
paper_img_dir = os.path.join(base_dir, 'paper')
scissors_img_dir = os.path.join(base_dir, 'scissors')

create_directory_with_validation(train_dir)
create_directory_with_validation(val_dir)

def create_directory_with_validation(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def move_all_files_to_dir(img_source, source_dir, destination_dir):
    for file in img_source:
        shutil.move(os.path.join(source_dir, file), os.path.join(destination_dir, file))

VAL_SIZE = 0.4

# Splitting images into training and validation sets
train_rock_img, val_rock_img = train_test_split(os.listdir(rock_img_dir), test_size=VAL_SIZE)
train_paper_img, val_paper_img = train_test_split(os.listdir(paper_img_dir), test_size=VAL_SIZE)
train_scissors_img, val_scissors_img = train_test_split(os.listdir(scissors_img_dir), test_size=VAL_SIZE)

# Creating directories for training and validation sets
train_rock_img_dir = os.path.join(train_dir, 'rock_img')
train_paper_img_dir = os.path.join(train_dir, 'paper_img')
train_scissors_img_dir = os.path.join(train_dir, 'scissors_img')

val_rock_img_dir = os.path.join(val_dir, 'rock_img')
val_paper_img_dir = os.path.join(val_dir, 'paper_img')
val_scissors_img_dir = os.path.join(val_dir, 'scissors_img')

create_directory_with_validation(train_rock_img_dir)
create_directory_with_validation(train_paper_img_dir)
create_directory_with_validation(train_scissors_img_dir)

create_directory_with_validation(val_rock_img_dir)
create_directory_with_validation(val_paper_img_dir)
create_directory_with_validation(val_scissors_img_dir)

# Moving files to the respective directories
move_all_files_to_dir(train_rock_img, rock_img_dir, train_rock_img_dir)
move_all_files_to_dir(train_paper_img, paper_img_dir, train_paper_img_dir)
move_all_files_to_dir(train_scissors_img, scissors_img_dir, train_scissors_img_dir)

move_all_files_to_dir(val_rock_img, rock_img_dir, val_rock_img_dir)
move_all_files_to_dir(val_paper_img, paper_img_dir, val_paper_img_dir)
move_all_files_to_dir(val_scissors_img, scissors_img_dir, val_scissors_img_dir)

# Data Augmentation for Training and Validation
train_img_data_generator = ImageDataGenerator(
    rotation_range=20,
    rescale=1./225,
    shear_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest',
)

val_img_data_generator = ImageDataGenerator(
    rotation_range=20,
    rescale=1./225,
    shear_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest',
)

train_img_generator = train_img_data_generator.flow_from_directory(
    train_dir,
    batch_size=32,
    target_size=(128, 128),
    class_mode='categorical'
)

val_img_generator = val_img_data_generator.flow_from_directory(
    val_dir,
    batch_size=32,
    target_size=(128, 128),
    class_mode='categorical'
)

# Create a Convolutional Neural Network (CNN) Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax', 
        kernel_regularizer=regularizers.l1_l2(l1=0.005, l2=0.005)),
])

# Compile the Model
model.compile(
    metrics=['accuracy'],
    loss=tf.keras.losses.KLDivergence(),
    optimizer=tf.optimizers.Nadam()
)

# Create Callbacks for Early Stopping
callbacks = [tf.keras.callbacks.EarlyStopping(patience=3)]

# Train the Model
model.fit(
    train_img_generator,
    validation_data=val_img_generator,
    steps_per_epoch=32,
    epochs=16,
    validation_steps=4,
    callbacks=callbacks,
    verbose=2,
    shuffle=True,
    workers=2,
    use_multiprocessing=True
)

# Define class names consistently with directory order
class_names = ['paper', 'rock', 'scissors']

# Function to predict and display the result
def predict_and_display(file_name, model):
    img = image.load_img(file_name, target_size=(128, 128))
    imgplot = plt.imshow(img)
    x = np.expand_dims(image.img_to_array(img), axis=0)
    images = np.vstack([x])
    result = model.predict(images)[0]
    predicted_class_index = np.argmax(result)
    print(f'This is: {class_names[predicted_class_index]}')

# Upload an image and make a prediction
uploaded_files = files.upload()
for file_name in uploaded_files.keys():
    predict_and_display(file_name, model)
