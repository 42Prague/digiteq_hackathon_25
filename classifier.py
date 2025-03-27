import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def train(dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        batch_size=32
        )
    print(train_ds.class_names)
    

def validate(dir):
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        batch_size=32
        )
    print(val_ds.class_names)

def main():
    data_dir = 'data/basic/dataset'
    data_dir = pathlib.Path(data_dir)
    print(f"files: {len(list(data_dir.glob('*.jpg')))}")

    train(data_dir)
    validate(data_dir)


if __name__ == "__main__":
    main()
