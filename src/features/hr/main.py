from video_wrapper import VideoWrapper
from ica_estimator import ICAEstimator
import argparse

#
# python src/features/hr/main.py --video_path miaaau.mp4 \
# --detector mtcnn_kalmann --extractor skvideo --type_roi \
# skin_adapt --skin_thresh_adapt 0.4
#


def create_parser():
    parser = argparse.ArgumentParser(description="Main script for ica \
                                                  method testing.")

    parser.add_argument("--video_path",
                        type=str,
                        required=True,
                        help="Path to the video input.")

    parser.add_argument("--detector",
                        type=str,
                        required=True,
                        help="Face detector method.",
                        choices=["dlib", "mtcnn", "mtcnn_kalmann"])

    parser.add_argument("--extractor",
                        type=str,
                        required=True,
                        help="Video extractor.",
                        choices=["opencv", "skvideo"])

    parser.add_argument("--type_roi",
                        type=str,
                        required=True,
                        help="Method for ROI extraction.",
                        choices=["skin_adapt", "skin_fix", "rect"])

    parser.add_argument("--skin_thresh_adapt",
                        type=float,
                        help="Skin threshold needed when \
                              type_roi=skin_adapt.")

    parser.add_argument("--skin_thresh_fix",
                        type=float,
                        help="Skin threshold needed when \
                              type_roi=skin_fix.")

    parser.add_argument("--rect_coords",
                        type=list,
                        help="ROI coordinates, needed when \
                              type_roi=skin_fix and no \
                              rect_regions is provided.")

    parser.add_argument("--rect_regions",
                        type=list,
                        help="ROI regions, needed when type_roi=rect \
                              and no rect_coords is provided.",
                        choices=["forehead", "lcheek", "rcheek", "nose"])

    parser.add_argument("--end_time",
                        type=float,
                        help="End time in seconds, \
                              needed when pyVHR is not able to compute \
                              video duration like amount_of_frames / fps.")

    return parser


def test_face_extractor(args):
    face_extractor = VideoWrapper(
        video_path=args.video_path,
        detector=args.detector,
        extractor=args.extractor,
        type_roi=args.type_roi,
        skin_thresh_adapt=args.skin_thresh_adapt,
        skin_thresh_fix=args.skin_thresh_fix,
        rect_coords=args.rect_coords,
        rect_regions=args.rect_regions,
        end_time="INF" if args.end_time is None else args.end_time)

    face_extractor.extract_faces()
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
        rect_regions=args.rect_regions,
        end_time="INF" if args.end_time is None else args.end_time)

    ica = ICAEstimator(face_extractor)
    ica.run_offline()
    ica.save("miaau.csv")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # You may what to run one test at
    # the time, just in case a problem occur
    # test_face_extractor(args)
    test_ica_estimator(args)
