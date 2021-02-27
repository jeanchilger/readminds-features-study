from ..features.hr.video_wrapper import VideoWrapper
from ..features.hr.ica_estimator import ICAEstimator
import argparse
import pandas as pd


AMOUNT_OF_FEATURES = 7


def get_args():
    parser = argparse.ArgumentParser(
        description="Main script for ica method testing.")

    parser.add_argument(
        "-v", "--video-path",
        type=str,
        required=True,
        help="Path to the video input.")

    parser.add_argument(
        "-o", "--output-path",
        type=str,
        required=False,
        help="Path to output csv file where heart rate will be appended.")

    parser.add_argument(
        "-d", "--detector",
        type=str,
        required=True,
        help="Face detector method.",
        choices=["dlib", "mtcnn", "mtcnn_kalman"])

    parser.add_argument(
        "-e", "--extractor",
        type=str,
        required=True,
        help="Video extractor.",
        choices=["opencv", "skvideo"])

    parser.add_argument(
        "-tr", "--type-roi",
        type=str,
        required=True,
        help="Method for ROI extraction.",
        choices=["skin_adapt", "skin_fix", "rect"])

    parser.add_argument(
        "-sta", "--skin-thresh-adapt",
        type=float,
        help="Skin threshold needed when \
        type-roi=skin_adapt.")

    parser.add_argument(
        "-stf", "--skin-thresh-fix",
        type=float,
        help="Skin threshold needed when \
        type-roi=skin_fix.")

    parser.add_argument(
        "-rc", "--rect_coords",
        type=list,
        help="ROI coordinates, needed when \
            type-roi=skin_fix and no \
            rect-regions is provided.")

    parser.add_argument(
        "-rr", "--rect-regions",
        type=list,
        help="ROI regions, needed when type-roi=rect \
            and no rect-coords is provided.",
        choices=["forehead", "lcheek", "rcheek", "nose"])

    return parser.parse_args()


def ica_estimator(args):
    """Method that generates heart rate estimations for a given
    input video, based in ICA rPPG method (check ica_estimator.py
    for more details).

    Args:
        args (argparse.Namespace): object containing Heart Rate
            estimator parameters (more details in video_wrapper.py).

    Returns:
        list: column-like array containing heart rate estimations,
            amount of rows matches the input video duration in
            seconds.

    """
    face_extractor = VideoWrapper(
        video_path=args.video_path,
        detector=args.detector,
        extractor=args.extractor,
        type_roi=args.type_roi,
        skin_thresh_adapt=args.skin_thresh_adapt,
        skin_thresh_fix=args.skin_thresh_fix,
        rect_coords=args.rect_coords,
        rect_regions=args.rect_regions)

    face_extractor.extract_faces(verbose=True)
    face_extractor.show_faces()
    ica = ICAEstimator(face_extractor)
    ica.run_offline()

    return ica.bpm_estimation


def append_heart_rate(output_file, bpm_estimations):
    """Append heart rate estimations in the already
    existing dataset of features (F1-F7).

    Args:
        output_file (str): path to csv file containing the
            features F1-F7.
        bpm_estimations (list): column-like array containing
            heart rate estimations.

    Raises:
        Exception: if the dataset amount of samples differ
            from the amount of heart rate estimations an
            exception is raised.
    """
    dataset = pd.read_csv(
        output_file, float_precision="round_trip",
        usecols=range(0, AMOUNT_OF_FEATURES))

    dataset_rows_amount = dataset.shape[0]
    dataset_columns_amount = dataset.shape[1]
    hr_estimation_amount = len(bpm_estimations)

    if dataset_rows_amount != hr_estimation_amount:
        raise Exception(
            "Dataset samples amount and heart rate " +
            "estimations must match in lenght")

    dataset.insert(dataset_columns_amount, "hr", bpm_estimations)

    dataset.to_csv(output_file, float_format="%.4f", index=False)


def show_estimator_config(args):
    d = args.__dict__
    max_str_len = max([len(val) for val in d.values() if isinstance(val, str)])
    print("=========================={}".format("="*max_str_len))
    print(" File name            : ", args.video_path)
    print(" Output CSV file      : ", args.output_path)
    print(" Detector             : ", args.detector)
    print(" Extractor            : ", args.extractor)
    print(" Type ROI             : ", args.type_roi)
    print(" Skin threshold adapt : ", args.skin_thresh_adapt)
    print(" Skin threshold fix   : ", args.skin_thresh_fix)
    print(" Rectangle coordinates: ", args.rect_coords)
    print(" Rectangle regions    : ", args.rect_regions)
    print("=========================={}".format("="*max_str_len))


def dummy_test(args):
    """
    To run this test a csv with features
    F1-F7 must be generated before hand.

    bazel-bin/src/feature_extractor/feature_extractor_video \
    --frame_width=320 --frame_height=240 --frame_rate=50 \
    --video_source=data/videos/fernando-320x240-1min.mp4

    python3 -m src.feature_extractor.hr_to_csv --video-path \
    fernando-320x240-1min.mp4 --output-path data/dataset/test.csv \
    --detector mtcnn_kalman --extractor skvideo --type-roi skin_adapt \
    --skin-thresh-adapt 0.4

    PS: this test can be removed when everything is working properly
    """
    # Heart rate estimations for fernando-320x240-1min.mp4
    bpm_estimations = [
        82.03125,
        82.03125,
        84.9609375,
        96.6796875,
        112.79296875,
        80.56640625,
        48.33984375,
        52.734375,
        54.19921875,
        115.72265625,
        49.8046875,
        54.19921875,
        49.8046875,
        52.734375,
        54.19921875,
        55.6640625,
        58.59375,
        77.63671875,
        87.890625,
        83.49609375,
        51.26953125,
        49.8046875,
        52.734375,
        57.12890625,
        65.91796875,
        71.77734375,
        67.3828125,
        74.70703125,
        49.8046875,
        51.26953125,
        54.19921875,
        65.91796875,
        62.98828125,
        61.5234375,
        55.6640625,
        54.19921875,
        54.19921875,
        70.3125,
        62.98828125,
        54.19921875,
        55.6640625,
        57.12890625,
        67.3828125,
        62.98828125,
        70.3125,
        93.75,
        51.26953125,
        52.734375,
        57.12890625,
        57.12890625,
        52.734375,
        86.42578125,
        84.9609375,
        83.49609375,
        65.91796875,
        57.12890625,
        54.19921875,
        82.03125,
        83.49609375]

    append_heart_rate(args.output_path, bpm_estimations)


if __name__ == "__main__":
    args = get_args()
    show_estimator_config(args)
    bpm_estimations = ica_estimator(args)
    append_heart_rate(args.output_path, bpm_estimations)
