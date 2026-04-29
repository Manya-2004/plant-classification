import os
import glob
import shutil
import random
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Check for GPU
device_name = tf.test.gpu_device_name()
if device_name:
    print(f"GPU detected: {device_name}")
else:
    print("No GPU detected, using CPU.")

# Constants
BATCH_SIZE = 32
IMG_SIZE = (128, 128)
EPOCHS = 100  # Adjust as needed
MODEL_DIR = 'models'

# Create models directory if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(data_dir, validation_split=0.2):
    """
    Load data with folder-based labels, create train/validation generators.
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    class_indices = train_generator.class_indices
    print("Class labels:", class_indices)
    return train_generator, val_generator, class_indices

def create_cnn_model(num_classes):
    """
    Create a simple CNN model.
    """
    model = models.Sequential([
        layers.InputLayer(input_shape=(*IMG_SIZE, 3)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_save_model(train_gen, val_gen, num_classes, save_path):
    """
    Train the CNN model and save it as .keras format.
    """
    model = create_cnn_model(num_classes)
    print(model.summary())
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )
    # Save as Keras v3 format (.keras)
    model.save(save_path)
    print(f"Model saved to {save_path} (Keras v3 format)")
    return model, history

def evaluate_model(model, val_gen):
    """
    Evaluate the model on validation data.
    """
    loss, acc = model.evaluate(val_gen)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")

def get_class_labels(generator):
    """
    Return list of class labels based on generator class indices.
    """
    labels = list(generator.class_indices.keys())
    return labels

def predict_image(model, image_path, class_labels):
    """
    Predict the class of a single image.
    """
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds[0])
    return class_labels[pred_idx], preds[0]

def fetch_sample_images(data_dir, class_name, num_samples=3):
    """
    Fetch sample images paths for a given class name.
    """
    class_dir = os.path.join(data_dir, class_name)
    images = glob.glob(os.path.join(class_dir, '*'))
    return random.sample(images, min(num_samples, len(images)))

def load_saved_model(model_path):
    """
    Load the saved Keras model for inference.
    """
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model

# Main pipeline
if __name__ == "__main__":
    # Path to the dataset folder
    data_folder = input("Enter the path to the 'diseases' folder: ").strip()

    # Load data
    train_gen, val_gen, class_indices = load_data(data_folder)

    # Train model for disease classification - SAVED AS .keras
    disease_model_path = os.path.join(MODEL_DIR, 'disease_model.keras')
    disease_model, history = train_and_save_model(train_gen, val_gen, len(class_indices), disease_model_path)

    # Evaluate
    evaluate_model(disease_model, val_gen)

    # Save class labels for inference (save as JSON for easy loading)
    class_labels = get_class_labels(train_gen)
    import json
    with open(os.path.join(MODEL_DIR, 'class_labels.json'), 'w') as f:
        json.dump(class_labels, f)
    print(f"Class labels saved to {MODEL_DIR}/class_labels.json")

    # Inference functions
    def predict_disease(image_path):
        label, probs = predict_image(disease_model, image_path, class_labels)
        print(f"Predicted Disease: {label}")
        print(f"Probabilities: {dict(zip(class_labels, probs))}")

    def predict_plant(image_path):
        # For this pipeline, assume plant health classification is same as disease classification
        label, probs = predict_image(disease_model, image_path, class_labels)
        print(f"Plant health status: {label}")
        print(f"Probabilities: {dict(zip(class_labels, probs))}")

    # Utility function
    def fetch_samples(plant_name, num_samples=3):
        samples = fetch_sample_images(data_folder, plant_name, num_samples)
        print(f"Sample images for {plant_name}:")
        for s in samples:
            print(s)

    print("\n=== Training Complete ===")
    print(f"Model saved as: {disease_model_path}")
    print("Class labels:", class_labels)
    print("\nExample usage:")
    print("predict_disease('path_to_image.jpg')")
    print("fetch_samples('Tomato Blight')")