import os
from train import train_model

BASE_DIR = os.path.join(os.getcwd(), "Data")

train_model(os.path.join(BASE_DIR, "level1"), "level1_model")

train_model(os.path.join(BASE_DIR, "trees_fruits"), "trees_model")
train_model(os.path.join(BASE_DIR, "herbs_medicinal"), "herbs_model")
train_model(os.path.join(BASE_DIR, "vegetables"), "vegetables_model")
train_model(os.path.join(BASE_DIR, "flowers"), "flowers_model")
train_model(os.path.join(BASE_DIR, "weeds"), "weeds_model")
train_model(os.path.join(BASE_DIR, "climbers"), "climbers_model")