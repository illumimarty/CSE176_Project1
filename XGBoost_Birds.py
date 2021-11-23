
import os
import keras
import keras_preprocessing as kp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

def main():

    # download dataset from Kaggle, unzip and place in /datasets folder
    train_data_dir = "./datasets/100-bird-species/train"
    valid_data_dir = "./datasets/100-bird-species/valid"
    test_data_dir = "./datasets/100-bird-species/valid"

    train_generator = image_dataset_from_directory(
        train_data_dir,
        labels="inferred",
        color_mode="rgb",
        image_size=(224,224),
        shuffle="True",
        seed=42    
    )
    valid_generator = image_dataset_from_directory(
        valid_data_dir,
        labels="inferred",
        color_mode="rgb",
        image_size=(224,224),
        shuffle="True",
        seed=42    
    )
    test_generator = image_dataset_from_directory(
        test_data_dir,
        labels="inferred",
        color_mode="rgb",
        image_size=(224,224),
        shuffle="True",
        seed=42    
    )

    for images, labels in train_generator:
        print(images.shape)
        print(labels.shape)
        break

    print(type(train_generator))
    class_names = train_generator.class_names

    plt.figure(figsize=(10,10))
    for images, labels in train_generator.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            print(labels[i])
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

if __name__== "__main__":
    main()
