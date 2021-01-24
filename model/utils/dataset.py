import pandas as pd
from pathlib import Path
    

def load_data_from_dir(directory, data_properties=None, pattern="**/*.csv"):
    """Reads data from an entire directory.

    Args:
        directory (string): Directory to read the data from.
        data_properties (DataProperties, optional): Filter files by 
            metadata info. Defaults to None.
        pattern (str, optional): Pattern to search for (usging glob). 
            Defaults to "**/*.csv".

    Returns:
        pandas.DataFrame: Single dataframe containing all data 
        read from directory
    """

    dataframe = pd.DataFrame()

    for file_name in Path(directory).glob("**/*.csv"):
        if data_properties in None or \
                data_properties == DataProperties.from_file_header(
                        str(file_name)):
            file_dataframe = pd.read_csv(file_name)

            if dataframe.empty():
                dataframe = file_dataframe
            else:
                dataframe = pd.concat([dataframe, file_dataframe])
    
    return dataframe


class DataProperties:
    def __init__(
            self, window_size=0, 
            initial_cutoff=0, questionnaire_method=""):
        # Sliding window averaging size for rppg 
        self.window_size = window_size
        # Amount of seconds removed from start of measurement 
        self.initial_cutoff = initial_cutoff
        # Method used for turning questionnaire into labels  
        self.questionnaire_method = questionnaire_method

    @classmethod
    def create_from_file_name(
            cls, file_name, window_size_prefix="rppg", 
            initial_cutoff_prefix="igcali"):
        """[summary]

        Args:
            file_name ([type]): [description]
            window_size_prefix (str, optional): [description]. Defaults to "rppg".
            initial_cutoff_prefix (str, optional): [description]. Defaults to "igcali".

        Returns:
            [type]: [description]
        """

        _window_size_prefix = cls.window_size_prefix
        window_index = file_name.find(_window_size_prefix)

        if window_index != -1:
            window_index += len(_window_size_prefix)
            window_size = int(file_name[window_index: window_index + 2])

            _initial_cutoff_prefix = cls.initial_cutoff_prefix
            initial_cutoff_index = file_name.find(_initial_cutoff_prefix) 
                    + len(_initial_cutoff_prefix)
            initial_cutoff = int(
                    file_name[initial_cutoff_index : initial_cutoff_index + 2])

            questionnaire_method = "design" 
                    if file_name.find("design") != -1 
                    else "self"

            return cls(window_size, initial_cutoff, questionnaire_method)

        else:
            return None

    def __eq__(self, obj):
        return isinstance(obj, data_properties) and \
            self.window_size == obj.window_size and \
            self.initial_cutoff == obj.initial_cutoff and \
            self.questionnaire_method == obj.questionnaire_method