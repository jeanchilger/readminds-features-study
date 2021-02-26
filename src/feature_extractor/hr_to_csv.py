from ..features.hr.video_wrapper import VideoWrapper
from ..features.hr.ica_estimator import ICAEstimator
import argparse
import pandas as pd


# python3 -m src.feature_extractor.hr_to_csv --video-path fernando-320x240-1min.mp4 --output-path data/dataset/test.csv --detector mtcnn_kalman --extractor skvideo --type-roi skin_adapt --skin-thresh-adapt 0.4


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
        help="Path to output csv file.")

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
    dataset = pd.read_csv(output_file, float_precision="round_trip")

    dataset_rows_amount = dataset.shape[0]
    dataset_columns_amount = dataset.shape[1]
    hr_estimation_amount = len(bpm_estimations)

    if dataset_rows_amount != hr_estimation_amount:
        print(dataset_rows_amount, hr_estimation_amount)
        raise Exception(
            "Dataset amount of rows " +
            "and heart rate estimations lenght must match")

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


if __name__ == "__main__":
    args = get_args()
    show_estimator_config(args)
    # bpm_estimations = ica_estimator(args)
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
