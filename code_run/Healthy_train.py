import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
import tensorflow as tf

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)

# Check for GPU
device_name = tf.test.gpu_device_name()
if device_name:
    print(f'GPU detected: {device_name}')
else:
    print('No GPU detected, training on CPU.')

# Paths
BASE_DIR = r'data\plant_metadata\Healthy'  # <-- Replace with your data folder path
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
EPOCHS = 70  # Adjust as needed

# Data Loading and Generators
def create_generators(base_dir):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=VALIDATION_SPLIT
    )

    train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=seed
    )

    val_generator = datagen.flow_from_directory(
        base_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True,
        seed=seed
    )

    return train_generator, val_generator

# Model Creation
def create_cnn_model(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(*IMG_SIZE, 3)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Load data and create generators
train_gen, val_gen = create_generators(BASE_DIR)
class_indices = train_gen.class_indices
classes = list(class_indices.keys())
num_classes = len(classes)

# Create model
model = create_cnn_model(num_classes)
model.summary()

# Save best model as .keras format
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, 'healthy_model.keras'), 
    monitor='val_accuracy', 
    save_best_only=True, 
    verbose=1
)

# Training
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

# Save the final trained model as .keras format
model.save(os.path.join(MODEL_DIR, 'healthy_model.keras'))
print(f"Final model saved as 'healthy_model.keras'")

# Load best model
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'healthy_model.keras'))

# Evaluation
loss, accuracy = model.evaluate(val_gen)
print(f'Validation Loss: {loss:.4f}')
print(f'Validation Accuracy: {accuracy:.4f}')

# Utility functions
def predict_plant(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds)
    predicted_class = classes[predicted_index]
    confidence = preds[0][predicted_index]
    print(f'Predicted Plant: {predicted_class} (Confidence: {confidence:.2f})')
    return predicted_class, confidence

def fetch_sample_images(plant_name, num_samples=3):
    # Find folder path
    folder_path = None
    for class_name in classes:
        if class_name == plant_name:
            folder_path = os.path.join(BASE_DIR, class_name)
            break
    if folder_path and os.path.exists(folder_path):
        images = os.listdir(folder_path)
        selected_images = random.sample(images, min(num_samples, len(images)))
        for img_name in selected_images:
            img_path = os.path.join(folder_path, img_name)
            img = load_img(img_path, target_size=IMG_SIZE)
            plt.imshow(img)
            plt.title(f'{plant_name}')
            plt.axis('off')
            plt.show()
    else:
        print(f'No images found for {plant_name}')

# Example Usage:
# predict_plant('path_to_some_image.jpg')
# fetch_sample_images('Tulsi')