import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30

def create_generators(data_dir):
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        validation_split=0.2,
        rotation_range=25,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset='training'
    )

    val = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        subset='validation'
    )

    return train, val


def build_model(num_classes):
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(data_dir, model_name):
    train_gen, val_gen = create_generators(data_dir)
    model = build_model(train_gen.num_classes)

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3),
        ModelCheckpoint(f"models/{model_name}.keras", save_best_only=True)
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    return model

BASE_DIR = os.path.join(os.getcwd(), "Data")
train_model(os.path.join(BASE_DIR, "trees_fruits"), "trees_model")
train_model(os.path.join(BASE_DIR, "herbs_medicinal"), "herbs_model")
train_model(os.path.join(BASE_DIR, "vegetables"), "vegetables_model")
train_model(os.path.join(BASE_DIR, "flowers"), "flowers_model")
train_model(os.path.join(BASE_DIR, "weeds"), "weeds_model")
train_model(os.path.join(BASE_DIR, "climbers"), "climbers_model")