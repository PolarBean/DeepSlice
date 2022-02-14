
##very janky way to solve relative import problem
import os

from pathlib import Path
##set path to be the DeepSlice directory
path = str(Path(__file__).parent) 

os.chdir(path)
print(path)
import warnings
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from skimage import color, transform
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utilities.QuickNII_functions import pd_to_quickNII, XML_to_csv
from utilities import plane_alignment
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from scipy.stats import trim_mean
from statistics import mean
import re

def ideal_thickness(results, depth, detect_bad_sections=False):
    number_spacing = np.float64(results['section_ID'].values[1:]) - np.float64(results['section_ID'].values[:-1])
    depth_spacing = (np.array(depth)[:-1] - np.array(depth)[1:])
    norm_depth = depth_spacing/number_spacing
 
    if detect_bad_sections==True:
        predicted_thickness = np.mean(norm_depth) 
        thickness_variability = np.std(norm_depth)
        good_indexes = (norm_depth<(predicted_thickness+(thickness_variability*3))) & (norm_depth>(predicted_thickness-(thickness_variability*3)))
        print(f"sections {results.Filenames[1:][~good_indexes]} appeared to be outliers and were not incorporated into the spacing analysis")
        norm_depth = norm_depth[good_indexes]
    predicted_thickness = np.mean(norm_depth) * 25
    thickness_variability = np.std(norm_depth) * 25

    print("Your sections appear to be sectioned at {0:.4f} micron thickness".format(np.abs(predicted_thickness)))
    print("the variability of thickness is {0:.4f} microns".format(thickness_variability))
    return predicted_thickness

def ideal_spacing(pred_oy, section_numbers, section_thickness_um, bad_section_indexes=None):
    if bad_section_indexes is None:
        bad_section_indexes = np.array([False] * len(pred_oy))

    pred_oy = np.float64(pred_oy[~bad_section_indexes])
    section_numbers = np.float64(section_numbers.values)
    section_thickness_um = np.float64(section_thickness_um)
    pred_um = pred_oy * 25
    section_um = section_numbers * section_thickness_um
    print(f"section_um: {pred_um/25}")
    avg_dist = np.mean(pred_um - section_um[~bad_section_indexes])
    print("ideal: ",  (section_um+avg_dist)/25)
    print("pred_UM: ")
    print(f"average distance: {avg_dist}")
    # print((pred_um-section_um) - avg_dist)
    return ((section_um+avg_dist))/25    

def calculate_brain_center_depth(section):
    cross, k = plane_alignment.find_plane_equation(section)
    translated_volume = np.array((456, 0, 320))
    linear_point = (
        ((translated_volume[0] / 2) * cross[0]) + ((translated_volume[2] / 2) * cross[2])) + k
    depth = -(linear_point / cross[1])
    return depth

class DeepSlice:
    def __init__(self, web=False, folder_name=None):
        self.web = web
        self.folder_name = folder_name

    def init_model(self, DS_weights, xception_weights):
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
        if DS_weights != None:
            # load weights
            model.load_weights(DS_weights)
        return model

    def Build(self,DS_weights = path+'/NN_weights/Allen_Mixed_Best.h5',\
         xception_weights=path+'/NN_weights/xception_weights_tf_dim_ordering_tf_kernels.h5',wise_weights=path+'/NN_weights/Synthetic_data_final.hdf5'):
        self.wise_weights = wise_weights
        self.DS_weights = DS_weights
        self.model = self.init_model(DS_weights=self.DS_weights, xception_weights=xception_weights)

 


    def gray_scale(self, img):
        # Downsamples images too 299 x 299
        # converts images to grayscale
        img = color.rgb2gray(img).reshape(299, 299, 1)
        return (img)
    
    def predict(self, image_dir, prop_angles=True, huber=False, wise=False):  # input
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
        if wise:
            self.Image_generator.reset()
            self.model.load_weights(self.wise_weights)
            wise_preds = self.model.predict(self.Image_generator,
                                   steps=self.Image_generator.n // self.Image_generator.batch_size, verbose=1)

            preds = np.mean((preds, wise_preds), axis=0)
            self.model.load_weights(self.DS_weights)

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

    def reorder_indexes(self):
        # reorders values so they are in the same order as the indexes with minimal swaps
        self.results.oy = sorted(self.results.oy) 
    

        

    def even_spacing(self, section_thickness_um=None, no_correction = False, order_only = False, bad_sections = [], detect_bad_sections = False, ignore_final_chars = 0):
        print("Section Numbers must have been included as the last three digits of the Filename")
        ###This function takes a dataset with section numbers and spaces those sections based on their numbers
        section_numbers = []
        depth = []
        count=1
        for Filename in self.results.Filenames.str.split('\\', expand=True).iloc[:,-1].values:
            if ignore_final_chars > 0:
                temp_filename = Filename[:-ignore_final_chars]
            else:
                temp_filename = Filename
            ##this removes all non-numeric characters
            section_number = re.sub("[^0-9]", "", temp_filename)
            ###this gets the three numbers closest to the end
            section_number = section_number[-3:]
            ind = [Filename in i for i in self.results.Filenames.values]
            d = self.results[ind]
            d = calculate_brain_center_depth(d[self.columns].values[0])
            #print(f"Filename: {Filename} section_number: {section_number} depth: {d}")

            ###find the first appearancex of the specified pattern
            ###remove non-numeric characters
            if len(section_number)<3:
                warnings.warn(f"could not find three digit section number for file \"{Filename}\", it should be the last three digits of the filenames.")
            if len(section_number)==0:
                warnings.warn(f"could not find any section number for file \"{Filename}\, using {count} instead")
                section_numbers.append(count)
            count+=1
            section_numbers.append(section_number)

        self.results['section_ID'] = section_numbers
        self.results.section_ID = self.results.section_ID.astype(np.float64)
        self.results = self.results.sort_values('section_ID', ascending=False).reset_index(drop=True)


        if order_only:
            self.reorder_indexes()
        depth = []
        for section in self.results[self.columns].values:
            depth.append((calculate_brain_center_depth(section)))
        depth = np.array(depth)
        self.results['depth'] = pd.Series(depth)
        self.results.to_csv("quickCheck.csv")
        if len(bad_sections)>0:
            bad_section_indexes = np.sum([self.results.Filenames.str.contains(bs) for bs in bad_sections], axis=0, dtype=bool)
            print(f" we found {np.sum(bad_section_indexes)} out of {len(bad_sections)} bad sections")
            estimate_thickness = ideal_thickness(self.results[~bad_section_indexes], depth[~bad_section_indexes], detect_bad_sections=detect_bad_sections)
        else:
            estimate_thickness = ideal_thickness(self.results, depth, detect_bad_sections=detect_bad_sections)

        print("\n", estimate_thickness, "\n")


        if estimate_thickness<0:
            print("the sections are not numbered rostrocaudaly")

        else:
            print("the sections are numbered rostrocaudaly")
            if section_thickness_um is not None:
                section_thickness_um*=-1

        if section_thickness_um is None:
            section_thickness_um = -estimate_thickness

        if no_correction:
            return 

        print(len(depth))
        if len(bad_sections)>0:
            ideal = ideal_spacing(depth, self.results['section_ID'], section_thickness_um, bad_section_indexes = bad_section_indexes)
        else:
            ideal = ideal_spacing(depth, self.results['section_ID'], section_thickness_um)


        self.results.oy-=(depth-ideal)
        depth = []
        for section in self.results[self.columns].values:
            depth.append((calculate_brain_center_depth(section)))
        self.results.section_ID = np.abs(self.results.section_ID)


    def load_QuickNII(self, filename):
        self.results = XML_to_csv(filename)

    def set_angles(self, DV=None, ML=None):
        rotated_sections = []
        count = 0
        for section in self.results.iterrows():
            section = section[1][self.columns].values
            original_depth = calculate_brain_center_depth(section)
            for i in range(10):
                cross, k = plane_alignment.find_plane_equation(section)
                if DV is not None:
                    section = plane_alignment.Section_adjust(
                        section, mean=DV, direction='DV')
                if ML is not None:
                    section = plane_alignment.Section_adjust(
                        section, mean=ML, direction='ML')
            rotated_depth = calculate_brain_center_depth(section)
            movement = rotated_depth - original_depth
            # section[1] -= movement
            rotated_sections.append(section)
            cross, k = plane_alignment.find_plane_equation(section)
            final_depth = calculate_brain_center_depth(section)
            # print(" original: {} \n rotated {} \n corrected {} \n".format(original_depth, rotated_depth, final_depth))

            count += 1
        self.results[self.columns] = rotated_sections
        # insert the section filenames into the pandas DataFrame


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
            oy=np.array(oy)
            oy[oy<0] = 0
            oy[oy>527]=527
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
            original_depth = calculate_brain_center_depth(section)
            for i in range(10):
                cross, k = plane_alignment.find_plane_equation(section)

                section = plane_alignment.Section_adjust(
                    section, mean=prop_DV[count], direction='DV')

                section = plane_alignment.Section_adjust(
                    section, mean=prop_ML[count], direction='ML')
            rotated_depth = calculate_brain_center_depth(section)
            movement = rotated_depth - original_depth
            # section[1] -= movement
            rotated_sections.append(section)
            cross, k = plane_alignment.find_plane_equation(section)
            final_depth = calculate_brain_center_depth(section)
            # print(" original: {} \n rotated {} \n corrected {} \n".format(original_depth, rotated_depth, final_depth))

            count += 1
        results = pd.DataFrame(rotated_sections, columns=self.columns)
        # insert the section filenames into the pandas DataFrame
        results["Filenames"] = self.Image_generator.filenames[:results.shape[0]]
        ordered_cols = ["Filenames"] + self.columns
        self.results = results[ordered_cols]  # To get the same column order

    def Save_Results(self, filename):
        if 'section_ID' in  self.results:
            section_numbers = np.abs(self.results['section_ID'])
        else:
            section_numbers = None
        pd_to_quickNII(results=self.results,
                       orientation='coronal', filename=str(filename), web=self.web, folder_name=self.folder_name, aligner = 'DeepSlice_ver_3.0_python')
