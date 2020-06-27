from tensorflow.keras.applications.xception import Xception
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from skimage import color, transform
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from QuickNII_functions import pd_to_quickNII
import plane_alignment
import pandas as pd
import numpy as np

class DeepSlice:
    def __init__(self, weights=None, web=True):
        self.weights = weights
        self.web = web



    def Build(self):
        # Download Xception architecture with weights pretrained on imagenet
        DenseModel = Xception(include_top=True, weights='xception_weights_tf_dim_ordering_tf_kernels.h5')
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




    def predict(self,image_dir):##input
        ##define_image_generator
        self.Image_generator = (ImageDataGenerator(preprocessing_function=self.gray_scale, samplewise_std_normalization=True)
                           .flow_from_directory(image_dir,
                                                target_size=(299, 299),
                                                batch_size=1,
                                                color_mode='rgb',
                                                shuffle=False))
        ##reset the image generator to ensure it starts from the first image
        self.Image_generator.reset()
        ##feed images to the model and store the predicted parameters
        preds = self.model.predict(self.Image_generator,
                              steps=self.Image_generator.n // self.Image_generator.batch_size, verbose=1)
        # convert the parameter values to floating point digits
        preds = preds.astype(float)
        # define the column names
        self.columns = ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]
        ##create a pandas DataFrame of the parameter values
        results = pd.DataFrame(preds, columns=self.columns)
        ##insert the section filenames into the pandas DataFrame
        results["Filenames"] = self.Image_generator.filenames[:results.shape[0]]
        ##This line is for compatibility with the website
        if self.web == True:
            results["Filenames"] = results["Filenames"].str.split('/')[1]
        ordered_cols = ["Filenames"] + self.columns
        self.results = results[ordered_cols]  # To get the same column order
        self.propagate_angles()

    def propagate_angles(self):
        DV = []
        ML = []
        for prediction in self.results.iterrows():
            m = prediction[1][['ox', 'oy', 'oz', 'ux', 'uy', 'uz', 'vx', 'vy', 'vz']].values.astype(np.float64)
            cross, k = plane_alignment.find_plane_equation(m)
            DV.append(plane_alignment.get_angle(m, cross, k, 'DV'))
            ML.append(plane_alignment.get_angle(m, cross, k, 'ML'))
        DV = sorted(DV, key=abs)
        ML = sorted(ML, key=abs)
        len_75 = int(len(ML) * 0.75)
        DV_mean = np.mean(DV[-len_75:])
        ML_mean = np.mean(ML[-len_75:])
        rotated_sections = []
        for section in self.results.iterrows():
            section = section[1][self.columns].values
            for i in range(4):
                section = plane_alignment.Section_adjust(section, mean=DV_mean, direction='DV')
                section = plane_alignment.Section_adjust(section, mean=ML_mean, direction='ML')

            rotated_sections.append(section)
            cross, k = plane_alignment.find_plane_equation(section)
            print(plane_alignment.get_angle(section, cross, k, 'DV'))
            print(plane_alignment.get_angle(section, cross, k, 'ML'))
        self.results = rotated_sections

    def Save_Results(self, filename):
        pd_to_quickNII(results=self.results, orientation='coronal', filename=str(filename))
