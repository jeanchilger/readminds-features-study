from video_wrapper import VideoWrapper
from ica_estimator import ICAEstimator
import argparse


"""
python src/features/hr/main.py --video-path fernando-320x240-1min.mp4 \
--output-file-path data/dataset/test.csv --detector mtcnn_kalman \
--extractor skvideo --type-roi skin_adapt --skin-thresh-adapt 0.4
"""


def get_args():
    parser = argparse.ArgumentParser(
        description="Main script for ica method testing.")

    parser.add_argument(
        "-v", "--video-path",
        type=str,
        required=True,
        help="Path to the video input.")

    parser.add_argument(
        "-o", "--output-file-path",
        type=str,
        required=True,
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


def test_face_extractor(args):
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


def test_ica_estimator(args):
    face_extractor = VideoWrapper(
        video_path=args.video_path,
        detector=args.detector,
        extractor=args.extractor,
        type_roi=args.type_roi,
        skin_thresh_adapt=args.skin_thresh_adapt,
        skin_thresh_fix=args.skin_thresh_fix,
        rect_coords=args.rect_coords,
        rect_regions=args.rect_regions)

    output_file_path = args.output_file_path
    face_extractor.extract_faces(verbose=True)
    face_extractor.show_faces()
    ica = ICAEstimator(face_extractor)
    ica.run_offline()
    ica.save(output_file_path)


def show_estimator_config(args):
    d = args.__dict__
    max_str_len = max([len(val) for val in d.values() if isinstance(val, str)])
    print("=========================={}".format("="*max_str_len))
    print(" File name            : ", args.video_path)
    print(" Output CSV file      : ", args.output_file_path)
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
    # You may what to run one test at
    # the time, just in case a problem occur
    # test_face_extractor(args)
    # test_ica_estimator(args)
