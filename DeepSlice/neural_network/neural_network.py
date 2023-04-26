from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import pandas as pd
import numpy as np
import os
from skimage.color import rgb2gray
import warnings
import imghdr
import struct


def gray_scale(img: np.ndarray) -> np.ndarray:
    """
    Convert the image to grayscale

    :param img: The image to convert
    :type img: numpy.ndarray
    :return: The converted image
    :rtype: numpy.ndarray
    """
    img = rgb2gray(img).reshape(299, 299, 1)
    return img


def initialise_network(xception_weights: str, weights: str, species: str) -> Sequential:
    """
    Initialise a neural network with the given weights

    :param weights: The weights for the network
    :type weights: list
    :param species: The species of the animal, this is necessary because of a previous error where the models are slightly different for different species
    :return: The initialised neural network
    :rtype: keras.models.Sequential
    """
    base_model = Xception(include_top=True, weights=xception_weights)
    base_model._layers.pop()
    base_model._layers.pop()
    if species == "rat":
        inputs = Input(shape=(299, 299, 3))
        base_model_layer = base_model(inputs, training=True)
        dense1_layer = Dense(256, activation="relu")(base_model_layer)
        dense2_layer = Dense(256, activation="relu")(dense1_layer)
        output_layer = Dense(9, activation="linear")(dense2_layer)
        model = Model(inputs=inputs, outputs=output_layer)
    else:
        model = Sequential()
        model.add(base_model)
        model.add(Dense(256, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(9, activation="linear"))

    if weights != None:
        model.load_weights(weights)
    return model




def load_images_from_path(image_path: str) -> np.ndarray:
    """
    Load the images from the given path
    :param image_path: The path to the images
    :type image_path: str
    :return: an Image generator for the found images
    :rtype: keras.preprocessing.image.ImageDataGenerator
    """
    if not os.path.isdir(image_path):
        raise ValueError("The path provided is not a directory")
    valid_formats = [".jpg", ".jpeg", ".png"]
    images = glob(image_path + "/*")

    images = [i for i in images if os.path.splitext(i)[1].lower() in valid_formats]
    sizes = [get_image_size(i) for i in images]
    width = [i[0] for i in sizes]
    height = [i[1] for i in sizes]
    if len(images) == 0:
        raise ValueError(
            f"No images found in the directory, please ensure image files are one of the following formats: {', '.join(valid_formats)}"
        )
    image_df = pd.DataFrame({"Filenames": images})
    with warnings.catch_warnings():
        ##throws warning about samplewise_std_normalization conflicting with samplewise_center which we don't use.
        warnings.simplefilter("ignore")
        image_generator = ImageDataGenerator(
            preprocessing_function=gray_scale, samplewise_std_normalization=True
        ).flow_from_dataframe(
            image_df,
            x_col="Filenames",
            y_col=None,
            target_size=(299, 299),
            batch_size=1,
            colormode="rgb",
            shuffle=False,
            class_mode=None,
        )
    return image_generator, width, height
    
def load_images_from_list(image_list: list) -> np.ndarray:
    """
    Load the images from the given list
    :param image_list: The list of images
    :type image_list: list
    :return: an Image generator for the found images
    :rtype: keras.preprocessing.image.ImageDataGenerator
    """
    valid_formats = [".jpg", ".jpeg", ".png"]
    images = [i for i in image_list if os.path.splitext(i)[1].lower() in valid_formats]
    sizes = [get_image_size(i) for i in images]
    width = [i[0] for i in sizes]
    height = [i[1] for i in sizes]
    if len(images) == 0:
        raise ValueError(
            f"No images found in the directory, please ensure image files are one of the following formats: {', '.join(valid_formats)}"
        )
    image_df = pd.DataFrame({"Filenames": images})
    with warnings.catch_warnings():
        ##throws warning about samplewise_std_normalization conflicting with samplewise_center which we don't use.
        warnings.simplefilter("ignore")
        image_generator = ImageDataGenerator(
            preprocessing_function=gray_scale, samplewise_std_normalization=True
        ).flow_from_dataframe(
            image_df,
            x_col="Filenames",
            y_col=None,
            target_size=(299, 299),
            batch_size=1,
            colormode="rgb",
            shuffle=False,
            class_mode=None,
        )
    return image_generator, width, height

def predictions_util(
    model: Sequential,
    image_generator: ImageDataGenerator,
    primary_weights: str,
    secondary_weights: str,
    ensemble: bool = False,
):
    """
    Predict the image alignments
    
    :param model: The model to use for prediction
    :param image_generator: The image generator to use for prediction
    :type model: keras.models.Sequential
    :type image_generator: keras.preprocessing.image.ImageDataGenerator
    :return: The predicted alignments
    :rtype: list
    """
    model.load_weights(primary_weights)
    predictions = model.predict(
        image_generator,
        steps=image_generator.n // image_generator.batch_size,
        verbose=1,
    )
    predictions = predictions.astype(np.float64)
    if ensemble:
        image_generator.reset()
        model.load_weights(secondary_weights)
        secondary_predictions = model.predict(
            image_generator,
            steps=image_generator.n // image_generator.batch_size,
            verbose=1,
        )
        predictions = np.mean([predictions, secondary_predictions], axis=0)
        model.load_weights(primary_weights)
    filenames = image_generator.filenames
    filenames = [i.split("/")[-1] for i in filenames]
    predictions_df = pd.DataFrame(
        {
            "Filenames": filenames,
            "ox": predictions[:, 0],
            "oy": predictions[:, 1],
            "oz": predictions[:, 2],
            "ux": predictions[:, 3],
            "uy": predictions[:, 4],
            "uz": predictions[:, 5],
            "vx": predictions[:, 6],
            "vy": predictions[:, 7],
            "vz": predictions[:, 8],
        }
    )

    return predictions_df


def get_image_size(fname):
    # https://stackoverflow.com/questions/8032642/how-to-obtain-image-size-using-standard-python-class-without-using-external-lib
    """Determine the image type of fhandle and return its size.
    from draco"""
    with open(fname, "rb") as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            raise Exception("Invalid header")

        ext = imghdr.what(fname)
        if imghdr.what(fname) == "png":
            check = struct.unpack(">i", head[4:8])[0]
            if check != 0x0D0A1A0A:
                raise Exception("png checksum failed")
            width, height = struct.unpack(">ii", head[16:24])
        elif imghdr.what(fname) == "gif":
            width, height = struct.unpack("<HH", head[6:10])
        elif imghdr.what(fname) == "jpeg":
            fhandle.seek(0)  # Read 0xff next
            size = 2
            ftype = 0
            while not 0xC0 <= ftype <= 0xCF:
                fhandle.seek(size, 1)
                byte = fhandle.read(1)
                while ord(byte) == 0xFF:
                    byte = fhandle.read(1)
                ftype = ord(byte)
                size = struct.unpack(">H", fhandle.read(2))[0] - 2
            # We are at a SOFn block
            fhandle.seek(1, 1)  # Skip `precision' byte.
            height, width = struct.unpack(">HH", fhandle.read(4))
        else:
            raise Exception(f"Invalid filetype: {head}")
        return width, height
