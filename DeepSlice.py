from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from skimage import color, transform
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utilities.QuickNII_functions import pd_to_quickNII
from utilities import plane_alignment
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from scipy.stats import trim_mean
from statistics import mean
import re

def ideal_spacing(pred_oy, section_numbers, section_thickness_um):
    pred_oy = np.float64(pred_oy)
    section_numbers = np.float64(section_numbers.values)
    section_thickness_um = np.float64(section_thickness_um)
    pred_um = pred_oy * 25
    section_um = section_numbers * section_thickness_um
    avg_dist = np.mean(pred_um - section_um)
    return ((section_um+avg_dist))/25    

def calculate_brain_center_depth(section):
    cross, k = plane_alignment.find_plane_equation(section)
    translated_volume = np.array((456, 0, 320))
    linear_point = (
        ((translated_volume[0] / 2) * cross[0]) + ((translated_volume[2] / 2) * cross[2])) + k
    depth = -(linear_point / cross[1])
    return depth

class DeepSlice:
    def __init__(self, weights='NN_weights/Synthetic_data_final.hdf5', web=False, folder_name=None):
        self.weights = weights
        self.web = web
        self.folder_name = folder_name

    def Build(self, xception_weights='NN_weights/xception_weights_tf_dim_ordering_tf_kernels.h5'):
        # Download Xception architecture with weights pretrained on imagenet
        DenseModel = Xception(
            include_top=True, weights=xception_weights)
        # remove the Dense Softmax layer and average pooling layer from the pretrained model
        DenseModel._layers.pop()
        DenseModel._layers.pop()
        # Build Deepslice
        model = Sequential()
        model.add(DenseModel)
        # we tested various sizes for these last two layers but consistently found that 256 performed best for some unknown reason.
        # theoretically larger layers should be better able to fit the training set but this is not what we saw.
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        # as we are predicting continuous values, here we define 9 output neurons with linear activation functions,
        # each corresponding to one of the QuickNII alignment variables Oxyz, Uxyz, Vxyz.
        model.add(Dense(9, activation='linear'))
        if self.weights != None:
            # load weights
            model.load_weights(self.weights)
        self.model = model

    def gray_scale(self, img):
        # Downsamples images too 299 x 299
        # converts images to grayscale
        img = color.rgb2gray(img).reshape(299, 299, 1)
        return (img)
    
    def predict(self, image_dir, huber=False, prop_angles=True):  # input
        # define_image_generator
        self.Image_generator = (ImageDataGenerator(preprocessing_function=self.gray_scale, samplewise_std_normalization=True)
                                .flow_from_directory(image_dir,
                                                     target_size=(299, 299),
                                                     batch_size=1,
                                                     color_mode='rgb',
                                                     shuffle=False))
        # reset the image generator to ensure it starts from the first image
        self.Image_generator.reset()
        # feed images to the model and store the predicted parameters
        preds = self.model.predict(self.Image_generator,
                                   steps=self.Image_generator.n // self.Image_generator.batch_size, verbose=1)
        # convert the parameter values to floating point digits
        preds = preds.astype(float)
        # define the column names
        self.columns = ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]
        # create a pandas DataFrame of the parameter values
        results = pd.DataFrame(preds, columns=self.columns)
        # insert the section filenames into the pandas DataFrame
        results["Filenames"] = self.Image_generator.filenames[:results.shape[0]]
        ordered_cols = ["Filenames"] + self.columns
        self.results = results[ordered_cols]  # To get the same column order
        if prop_angles:
            self.propagate_angles(huber)


    def even_spacing(self, section_number_pattern, section_thickness_um):
        ###This function takes a dataset with section numbers and spaces those sections based on their numbers
        section_numbers = []
        for Filename in self.results.Filenames.values:
            section_number = re.search(str(section_number_pattern), Filename)
            ###find the first appearance of the specified pattern
            section_number = section_number.group(0)
            ###remove non-numeric characters
            section_number = re.sub("[^0-9]", "", section_number)
            section_numbers.append(section_number)



        self.results['section_ID'] = section_numbers
        self.results.section_ID = self.results.section_ID.astype(np.float64)
        depth = []
        for section in self.results[self.columns].values:
            depth.append((calculate_brain_center_depth(section)))
        ideal = ideal_spacing(depth, self.results['section_ID'], section_thickness_um)
        self.results.oy-=(depth-ideal)

    def propagate_angles(self, huber=True):
        DV = []
        ML = []
        oy = []
        for prediction in self.results.iterrows():
            m = prediction[1][['ox', 'oy', 'oz', 'ux', 'uy',
                               'uz', 'vx', 'vy', 'vz']].values.astype(np.float64)
            oy.append(m[1])
            cross, k = plane_alignment.find_plane_equation(m)
            DV.append(plane_alignment.get_angle(m, cross, k, 'DV'))
            ML.append(plane_alignment.get_angle(m, cross, k, 'ML'))
        if huber == True:
            oy = np.array(oy).reshape(-1, 1)
            # we use a huberised linear regressor as it is more robust to outliers
            huber_regressor = HuberRegressor().fit(oy, DV)
            # for our predictions we multiple the depth by the coefficient and add the y intercept
            prop_DV = (huber_regressor.coef_ * oy) + huber_regressor.intercept_
            huber_regressor = HuberRegressor().fit(oy, ML)
            prop_ML = (huber_regressor.coef_ * oy) + huber_regressor.intercept_
        else:
            length = len(DV)
            weights = plane_alignment.make_gaussian_weights(0, 528)
            weights = [weights[int(y)] for y in oy]
            # DV = sorted(DV, key=abs)[int(length*0.75):]
            # ML = sorted(ML, key=abs)[int(length*0.75):]
            print(weights)
            prop_DV = [np.average(DV, weights=np.array(weights))] * length
            prop_ML = [np.average(ML, weights=np.array(weights))] * length

        rotated_sections = []
        count = 0
        for section in self.results.iterrows():
            section = section[1][self.columns].values
            for i in range(10):
                cross, k = plane_alignment.find_plane_equation(section)

                section = plane_alignment.Section_adjust(
                    section, mean=prop_DV[count], direction='DV')

                section = plane_alignment.Section_adjust(
                    section, mean=prop_ML[count], direction='ML')

            rotated_sections.append(section)
            cross, k = plane_alignment.find_plane_equation(section)

            count += 1
        results = pd.DataFrame(rotated_sections, columns=self.columns)
        # insert the section filenames into the pandas DataFrame
        results["Filenames"] = self.Image_generator.filenames[:results.shape[0]]
        ordered_cols = ["Filenames"] + self.columns
        self.results = results[ordered_cols]  # To get the same column order

    def Save_Results(self, filename):
        pd_to_quickNII(results=self.results,
                       orientation='coronal', filename=str(filename), web=self.web, folder_name=self.folder_name)
