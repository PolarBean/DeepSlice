from glob import glob
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.color import rgb2gray

from .network_architecture import (
    DeepSliceNetwork,
    load_deepslice_weights,
    load_imagenet_classifier,
)

IMAGE_SIZE = 299
VALID_FORMATS = [".jpg", ".jpeg", ".png"]
BATCH_SIZE = 8


def gray_scale(img: np.ndarray) -> np.ndarray:
    """
    Convert the image to grayscale

    :param img: The image to convert
    :type img: numpy.ndarray
    :return: The converted image
    :rtype: numpy.ndarray
    """
    img = rgb2gray(img).reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
    return img


def preprocess_image(path: str) -> np.ndarray:
    """
    Load an image and prepare it for the network: resize to 299x299 with nearest
    neighbour sampling, convert to grayscale, standardise to zero mean and unit variance and
    repeat the grayscale channel three times. This matches the Keras
    ImageDataGenerator the network was trained with.

    :param path: The path to the image
    :type path: str
    :return: The image, of shape (299, 299, 3)
    :rtype: numpy.ndarray
    """
    with Image.open(path) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        img = np.asarray(image, dtype=np.float32)
    img = gray_scale(img)
    # Keras turned on samplewise_center along with samplewise_std_normalization
    img -= np.mean(img, keepdims=True)
    img /= np.std(img, keepdims=True) + 1e-6
    return np.repeat(img, 3, axis=2)


class ImageLoader:
    """Loads and preprocesses images in batches for prediction"""

    def __init__(self, filenames: list, batch_size: int = BATCH_SIZE):
        self.filenames = list(filenames)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        for start in range(0, len(self.filenames), self.batch_size):
            batch = self.filenames[start : start + self.batch_size]
            yield np.stack([preprocess_image(f) for f in batch])


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialise_network(xception_weights: str, weights: str, species: str) -> DeepSliceNetwork:
    """
    Initialise a neural network with the given weights

    :param xception_weights: The ImageNet Xception weights, which provide the ImageNet classifier
    :param weights: The DeepSlice weights for the network
    :param species: The species of the animal, this is necessary because of a previous error where the models are slightly different for different species
    :return: The initialised neural network
    :rtype: DeepSliceNetwork
    """
    model = DeepSliceNetwork()
    load_imagenet_classifier(model, xception_weights)
    if weights is not None:
        model = load_xception_weights(model, weights, species)
    model.eval()
    return model.to(get_device())


def load_xception_weights(model, weights, species="mouse"):
    return load_deepslice_weights(model, weights)


def _check_images(images: list) -> list:
    images = [i for i in images if os.path.splitext(i)[1].lower() in VALID_FORMATS]
    if len(images) == 0:
        raise ValueError(
            f"No images found in the directory, please ensure image files are one of the following formats: {', '.join(VALID_FORMATS)}"
        )
    return images


def load_images_from_path(image_path: str):
    """
    Load the images from the given path
    :param image_path: The path to the images
    :type image_path: str
    :return: an image loader for the found images, and their widths and heights
    :rtype: tuple(ImageLoader, list, list)
    """
    if not os.path.isdir(image_path):
        raise ValueError("The path provided is not a directory")
    return load_images_from_list(glob(image_path + "/*"))


def load_images_from_list(image_list: list):
    """
    Load the images from the given list
    :param image_list: The list of images
    :type image_list: list
    :return: an image loader for the found images, and their widths and heights
    :rtype: tuple(ImageLoader, list, list)
    """
    images = _check_images(image_list)
    sizes = [get_image_size(i) for i in images]
    width = [i[0] for i in sizes]
    height = [i[1] for i in sizes]
    return ImageLoader(images), width, height


def _predict(model: DeepSliceNetwork, image_loader: ImageLoader) -> np.ndarray:
    device = next(model.parameters()).device
    predictions = []
    with torch.no_grad():
        for batch in image_loader:
            output = model(torch.from_numpy(batch).to(device))
            predictions.append(output.cpu().numpy())
    return np.concatenate(predictions).astype(np.float64)


def predictions_util(
    model: DeepSliceNetwork,
    image_loader: ImageLoader,
    primary_weights: str,
    secondary_weights: str,
    ensemble: bool = False,
    species: str = "mouse",
):
    """
    Predict the image alignments

    :param model: The model to use for prediction
    :param image_loader: The images to predict
    :type model: DeepSliceNetwork
    :type image_loader: ImageLoader
    :return: The predicted alignments
    :rtype: pandas.DataFrame
    """
    model = load_xception_weights(model, primary_weights, species)
    predictions = _predict(model, image_loader)
    if ensemble:
        model = load_xception_weights(model, secondary_weights, species)
        secondary_predictions = _predict(model, image_loader)
        predictions = np.mean([predictions, secondary_predictions], axis=0)
        model = load_xception_weights(model, primary_weights, species)
    filenames = [os.path.basename(i) for i in image_loader.filenames]
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
    with Image.open(fname) as image:
        width, height = image.size
    return width, height
