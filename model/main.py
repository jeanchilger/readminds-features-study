"""General Pipeline from notebook.
- 1. `compile_data_recursive` (L 119);
- 2. `create_preprocessed` (L 427);
- 3. `tune_model` (L 428);
- 4. `create_preprocessed` (L 434);
- 5. `get_testable_subjects` (L 436);
- 6. `subject_cross_validation` (L 440);
- 7. `save_results` (L 473);
- 8. Train and evaluate for different measurements (L 479-492);
- 9. Filter features and measurements (L 517-542);
- 10. Train eliminating 1 feature at a time (L 545-553);
"""

from utils.dataset import (
    DataProperties,
    load_data_from_dir,
    normalize_and_encode,
    split_features_label,
)

# Don't change these values
# Facial features stored in the data, combine facial measurement type and 
# base facial feature to get a string index
facial_measurement_types = ["mean_", "std_", "sum_"]

base_facial_features = [
    "face_activity_eyebrow",
    "face_activity_eyebrow_right",
    "face_activity_eyebrow_left",
    "face_activity_mouth_corner",
    "face_activity_mouth_outer",
    "face_area",
    "face_blinks_1s",
    "face_blinks_5s",
    "face_com_distance",
    "face_eye_area",
    "face_motion_instability",
    "face_mouth_area",
    "face_mouth_edges_change",
    "face_mouth_histogram_change",
]

# Other indices
HR_ground_header = "mean_HR_ground"
rppg_header = "HR_poh2011"
subject_no_header = "subject"
game_type_header = "game"


# Define label and feature names to use
label_header = "emotional_state"

feature_headers = [
    "mean_HR_ground",
    "mean_face_activity_mouth_outer",
    "mean_face_activity_mouth_corner",
    "mean_face_eye_area",
    "mean_face_activity_eyebrow",
    "mean_face_area",
    "mean_face_motion_instability",
    "mean_face_com_distance",
]

# DON'T CHANGE THESE VALUES, IMPORTANT
validation_subjects = [408, 402, 418, 934, 960, 908]

# Training hyperparameters
max_epochs = 50

# Early stop patience is a measure of how many epochs
# of training the network goes through without a positive
# performance impact before stopping training.
early_stop_patience = 100


def calibration_validation_split(
        dataset, game_type_header="game", 
        validation_game_substr="Mario"):
    """Splits a dataset into two.

    Splits a dataset in calibration and validation data. The split
    is made based on a game containing `validation_game_substr` in its
    name. `game_type_header` is the column used to search this pattern.

    Args:
        dataset (pandas.DataFrame): Input dataset
        game_type_header (str, optional): [description]. Defaults to "game".
        validation_game_substr (str, optional): [description]. 
            Defaults to "Mario".

    Returns:
        (pandas.DataFrame, pandas.DataFrame): [description]
    """
    
    calibration_data = dataset[~dataset[game_type_header].str.contains(
            validation_game_substr)]

    validation_data = dataset[dataset[game_type_header].str.contains(
            validation_game_substr)]

    return calibration_data, validation_data


if __name__ == "__main__":
    dataset_path = "data/dataset/dagibs"

    # Load dataset
    compiled_data = load_data_from_dir(
            dataset_path, 
            DataProperties(
                    window_size=60, 
                    initial_cutoff=45,
                    questionnaire_method="self"))

    print(compiled_data.shape)

    # Normalize and encode labels
    compiled_data = compiled_data[compiled_data[label_header] != "neutral"]

    dataset = normalize_and_encode(compiled_data, feature_headers, label_header)

    # Remove validation subjects from dataset
    dataset = dataset[~dataset[subject_no_header].isin(validation_subjects)]

    # Split dataset into calibration and validation 
    # data (aka training and testing data)
    calibration_data, validation_data = calibration_validation_split(dataset)

    calibration_X, calibration_y = split_features_label(
            calibration_data, feature_headers, label_header)
    
    validation_X, validation_y = split_features_label(
            validation_data, feature_headers, label_header)