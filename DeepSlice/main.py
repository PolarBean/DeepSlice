from typing import Union
from .coord_post_processing import spacing_and_indexing, angle_methods
from .read_and_write import QuickNII_functions
from .neural_network import neural_network
from .metadata import metadata_loader


class Model:
    def __init__(self, species: str):
        """
        Initialise a DeepSlice model for a given species.
        :param species: the name of the species
        :type species: str
        """
        # The config file contains information about the DeepSlice version, and neural network weights for each species.
        self.config, self.weight_path = metadata_loader.load_config()
        self.species = species
        self.model = neural_network.initialise_network(
            self.weight_path + self.config["weight_file_paths"]["xception_imagenet"],
            self.weight_path + self.config["weight_file_paths"][species]["primary"],
        )

    def predict(
        self,
        image_directory: str,
        ensemble: bool = None,
        section_numbers: bool = True,
        legacy_section_numbers=False,
    ):
        """
        predicts the atlas position for a folder full of histological brain sections
        :param image_directory: the directory containing the brain sections
        :param ensemble: whether to use multiple models, this will default to True when available
        :param section_numbers: whether this dataset includes section numbers (as the last three digits of the filename). 
        :type image_directory: str
        :type ensemble: bool    
        :type section_numbers: bool
        """
        # We set this to false as predict is the entry point for a new brain and therefore we need to reset all values which may persist from a previous animal.
        self.bad_sections_present = False
        # Different species may or may not have an ensemble model, so we need to check for this before defaulting to True
        if ensemble is None:
            ensemble = self.config["ensemble_status"][self.species]
            ensemble = eval(ensemble)

        image_generator = neural_network.load_images(image_directory)
        primary_weights = (
            self.weight_path + self.config["weight_file_paths"][self.species]["primary"]
        )
        secondary_weights = (
            self.weight_path
            + self.config["weight_file_paths"][self.species]["secondary"]
        )
        if secondary_weights is "None":
            print(f"ensemble is not available for {self.species}")
            ensemble = False
        predictions = neural_network.predictions_util(
            self.model, image_generator, primary_weights, secondary_weights, ensemble
        )
        if section_numbers:
            predictions["nr"] = spacing_and_indexing.number_sections(
                predictions["Filenames"], legacy_section_numbers
            )
            predictions["nr"] = predictions["nr"].astype(int)
        #: pd.DataFrame: Filenames and predicted QuickNII coordinates of the input sections.
        self.predictions = predictions

    def set_bad_sections(self, bad_sections: list):
        """
        sets the bad sections for a given brain. Must be run after predict()
        :param bad_sections: A list of bad sections to ignore when calculating angles and spacing
        :type bad_sections: list
        """
        self.predictions = spacing_and_indexing.set_bad_sections_util(
            self.predictions, bad_sections
        )

    def enforce_index_order(self):
        """
        reorders the section depths (oy) in the predictions such that they align with the section indexes 
        """
        self.predictions = spacing_and_indexing.enforce_section_ordering(
            self.predictions
        )

    def enforce_index_spacing(self):
        self.predictions = spacing_and_indexing.space_according_to_index(
            self.predictions
        )

    def adjust_angles(self, ML: Union[int, float], DV: Union[int, float]):
        """
        Adjusts the Mediolateral (ML) and Dorsoventral (DV) angles of all sections to the specified values.
        :param ML: the Mediolateral angle to set
        :param DV: the Dorsoventral angle to set
        :type ML: [int, float]
        :type DV: [int, float]
        """
        self.predictions = angle_methods.set_angles(self.predictions, ML, DV)

    def propagate_angles(self, method="weighted_mean"):
        """
        Calculates the average Mediolateral and Dorsoventral angles for all sections.
        """
        ##needs to be run twice as adjusting the angle in one plane bumps the other out slightly.
        for i in range(2):
            self.predictions = angle_methods.propagate_angles(
                self.predictions, method, self.species
            )

    def save_predictions(self, filename):
        """
        Save the predictions to a QuickNII compatible JSON file.
        :param filename: the name of the file to save to
        :type filename: str
        """
        target = self.config["target_volumes"][self.species]
        aligner = self.config["DeepSlice_version"]["prerelease"]
        self.predictions.to_csv(filename + ".csv", index=False)
        QuickNII_functions.write_QUINT_JSON(
            df=self.predictions, filename=filename, aligner=aligner, target=target
        )
        QuickNII_functions.write_QuickNII_XML(
            df=self.predictions, filename=filename, aligner=aligner
        )

