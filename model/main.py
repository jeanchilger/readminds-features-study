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

import numpy as np

from evaluation.metrics import recall as recall_m, precision, f1_score
from kerastuner.tuners import RandomSearch
# from models.tunning import TunableModel
from models.nn import create_model
from preprocessing import normalize_and_encode
from tensorflow import keras
from utils import set_random_state
from utils import console
from utils.dataset import (
    DataProperties,
    load_data_from_dir,
    split_features_label,
)

# Set python hash and frameworks' random seed to a comon value
set_random_state(seed=0)

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

    train_X, train_y = split_features_label(
            calibration_data, feature_headers, label_header)

    test_X, test_y = split_features_label(
            validation_data, feature_headers, label_header)

    return train_X, train_y, test_X, test_y


def run_subject_model_experiment(
        compiled_data, feature_headers, label_header, testable_subjects,
        subject_no_header="subject", early_stop_patience=100, verbose=0):
    """Runs "Subject Model" experiment.

    The training process is as follows: First, a generic model is
    obtained by training with data from all subjects, removing examples
    were the game is "Mario". Second, the generic model is specialized
    using subject-specific data. Lastly, the validation is made with
    "Mario" data from particular subjects (each model is validated
    individually).

    Args:
        compiled_data (pandas.DataFrame): Raw dataset.
        feature_headers (list of str): Features to be used.
        label_header (str): Label header.
        testable_subjects (list of int): Ids of subjects that has Mario data.
        subject_no_header (str, optional): Subject column name in dataset.
            Defaults to "subject".
        early_stop_patience (int, optional): Parameter for EarlyStopping.
            Defaults to 100.
        verbose (int, optional): Verbosity level passed to keras.
            Defaults to 2.
    """

    dataset = normalize_and_encode(
            compiled_data, feature_headers, label_header)

    # Split dataset into calibration and validation
    # data (aka training and testing data)
    train_X, train_y, test_X, test_y = calibration_validation_split(dataset)

    early_stop = keras.callbacks.EarlyStopping(patience=early_stop_patience)

    # Creates the generic, unspecialized, model
    generic_model = create_model(
            input_size=len(feature_headers),
            output_size=len(dataset[label_header].unique()))

    generic_model.fit(
            train_X, train_y, epochs=1000, batch_size=500,
            validation_data=(test_X, test_y), verbose=verbose)
    generic_model.save("best_model")

    print(generic_model.summary())

    # NOTE: Training the generic model every time in the loop
    # has given an accuracy of 61.65% while using a single training for
    # the generic model granted 61.53%. We should check if this is due
    # to keras lack of weight copying (a bug in our setup then) or if
    # is really a improvement. Also, training only with subject data
    # (same as in Bevilacqua's work) has reported an accuracy of 61.95%.

    # Specialize model for each subject
    scores = []
    for subject_id in testable_subjects:
        console.warning("Subject: " + str(subject_id))

        subject_model = create_model(
                input_size=len(feature_headers),
                output_size=len(dataset[label_header].unique()))

        # Get subject specific data
        train_sub_X, train_sub_y, test_sub_X, test_sub_y = \
            calibration_validation_split(
                    dataset[dataset[subject_no_header] == subject_id])

        subject_model.fit(
                train_sub_X, train_sub_y, epochs=1000,
                batch_size=50, validation_data=(test_sub_X, test_sub_y),
                callbacks=[early_stop], verbose=verbose)

        score = subject_model.evaluate(test_sub_X, test_sub_y, verbose=0)
        scores.append(score)

        print("\tAccuracy: {:.3f}%".format(score[1] * 100))

    scores = np.array(scores)
    console.error("ACC @ Mean: {:.3f}%".format(
            np.mean(scores, axis=0)[1] * 100))


def run_feature_group_experiment(
        compiled_data, label_header, testable_subjects,
        subject_no_header="subject", early_stop_patience=100, verbose=2):
    """Runs "Feature Group" model experiment.

    The training occurs similarly as in "Subject Model" experiment.
    The only difference is that in "Feature Group", the features used
    are different: every measurement type (sum, mean or std) is
    combined and used along with `base_facial_features`. The dataset taken
    from these features are used in "Subject Model" training.

    Args:
        compiled_data (pandas.DataFrame): Raw dataset.
        label_header (str): Label header.
        testable_subjects (list of int): Ids of subjects that has Mario data.
        subject_no_header (str, optional): Subject column name in dataset.
            Defaults to "subject".
        early_stop_patience (int, optional): Parameter for EarlyStopping.
            Defaults to 100.
        verbose (int, optional): Verbosity level passed to keras.
            Defaults to 2.
    """

    measurement_groups = [
        ["sum_", "mean_", "std_"],
        ["sum_", "mean_"],
        ["sum_", "std_"],
        ["mean_", "std_"],
        ["sum_"],
        ["mean_"],
        ["std_"],
    ]

    for measurement_group in measurement_groups:
        _feature_headers = [measurement + feature
                            for measurement in measurement_group
                            for feature in base_facial_features]

        console.info(
                "Using measurements: [" + ", ".join(measurement_group) + "]")

        run_subject_model_experiment(
                compiled_data, _feature_headers, label_header,
                testable_subjects, subject_no_header,
                early_stop_patience, verbose)


if __name__ == "__main__":
    dataset_path = "data/dataset/dagibs"

    # Load dataset
    compiled_data = load_data_from_dir(
            dataset_path,
            DataProperties(
                    window_size=60,
                    initial_cutoff=45,
                    questionnaire_method="self"))

    # Remove validation subjects from dataset
    compiled_data = compiled_data[~compiled_data[subject_no_header].isin(
            validation_subjects)]

    compiled_data = compiled_data[compiled_data[label_header] != "neutral"]

    print(compiled_data.shape)

    # Get subjects that has Mario entries
    testable_subjects = [
            subject_id
            for subject_id in compiled_data[subject_no_header].unique()
            if any(compiled_data[
                compiled_data[subject_no_header] == subject_id][game_type_header]
                .str.contains("Mario"))]

    testable_subjects = sorted(testable_subjects)

    # NOTE: I noticed swedish guys have pre-trained on all subjects
    # including those without Mario game entry. Later they've specialized
    # the model for each subject that has Mario entry. In other words we
    # have: len(train_data) + len(test_data) < len(dataset).

    #######################################################
    # Model experiment 1
    #######################################################
    # print("@"*70)
    # print("Model Experiment 1")
    # print("@"*70)

    # run_subject_model_experiment(
    #         compiled_data, feature_headers, label_header,
    #         testable_subjects)

    #######################################################
    # Model experiment 2
    # Feature group
    #######################################################
    # print("@"*70)
    # print("Model Experiment 2")
    # print("@"*70)

    # run_feature_group_experiment(
    #         compiled_data, label_header, testable_subjects)
