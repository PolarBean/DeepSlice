from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import pandas as pd
import numpy as np
import os
from skimage.color import rgb2gray
import warnings


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


def initialise_network(xception_weights: str, weights: str) -> Sequential:
    """
    Initialise a neural network with the given weights
    :param weights: The weights for the network
    :type weights: list
    :return: The initialised neural network
    :rtype: keras.models.Sequential
    """
    base_model = Xception(include_top=True, weights=xception_weights)
    base_model._layers.pop()
    base_model._layers.pop()
    model = Sequential()
    model.add(base_model)
    model.add(Dense(256, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(9, activation="linear"))
    if weights != None:
        model.load_weights(weights)
    return model


def load_images(image_path: str) -> np.ndarray:
    """
    Load the images from the given path
    :param image_path: The path to the images
    :type image_path: str
    :return: an Image generator for the found images
    :rtype: keras.preprocessing.image.ImageDataGenerator
    """
    if not os.path.isdir(image_path):
        raise ValueError("The path provided is not a directory")
    valid_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    images = glob(image_path + "/*")
    images = [i for i in images if os.path.splitext(i)[1].lower() in valid_formats]
    if len(images) == 0:
        raise ValueError(f"No images found in the directory, please ensure image files are one of the following formats: {', '.join(valid_formats)}")
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
    return image_generator


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

