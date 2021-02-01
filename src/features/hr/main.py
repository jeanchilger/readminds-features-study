from video_wrapper import VideoWrapper
from ica_estimator import ICAEstimator
import argparse


def test_ica_estimator():
    parser = argparse.ArgumentParser(description="Main script for ica \
                                                  method testing. Please \
                                                  use check --help")

    parser.addArgument("--video_path",
                       type=str,
                       help="Path to the video input.")

    parser.addArgument("--detector",
                       type=str,
                       help="Face detector method \
                             dlib, mtcnn, mtcnn_kalmann).")

    parser.addArgument("--extractor",
                       type=str,
                       help="Video extractor (opencv, skvideo).")

    parser.addArgument("--type_roi",
                       type=str,
                       help="Method for ROI extraction \
                            (skin_adapt, skin_fix, rect).")

    parser.addArgument("--type_roi",
                       type=str,
                       help="Method for ROI extraction \
                             (skin_adapt, skin_fix, rect).")

    parser.addArgument("--skin_thresh_adapt",
                       type=float,
                       help="Skin threshold needed when \
                             type_roi=skin_adapt.")

    parser.addArgument("--skin_thresh_fix",
                       type=float,
                       help="Skin threshold needed when \
                             type_roi=skin_fix.")

    parser.addArgument("--rect_coords",
                       type=list,
                       help="ROI coordinates, needed when \
                             type_roi=skin_fix and no \
                             rect_regions is provided.")

    parser.addArgument("--rect_regions",
                       type=list,
                       help="ROI regions, needed when type_roi=rect \
                             and no rect_coords is provided.")

    parser.addArgument("--end_time",
                       type=int,
                       help="End time in seconds, \
                             needed when pyVHR is not able to compute \
                             video duration with amount_of_frame / fps.")

    args = parser.parse_args()

    video_path = args.video_path
    detector = args.detector
    extractor = args.extractor
    type_roi = args.type_roi
    skin_thresh_adapt = args.skin_thresh_adapt
    skin_thresh_fix = args.skin_thresh_fix
    rect_coords = args.rect_coords
    rect_regions = args.rect_regions
    end_time = args.end_time
    face_extractor = VideoWrapper(video_path, detector, extractor, type_roi,
                                  skin_thresh_adapt=skin_thresh_adapt,
                                  skin_thresh_fix=skin_thresh_fix,
                                  rect_coords=rect_coords,
                                  rect_regions=rect_regions,
                                  end_time=59)
    face_extractor.extract_faces()
    face_extractor.show_faces()

    ica = ICAEstimator(face_extractor)
    ica.run_offline()
    ica.save("miaau.csv")


if __name__ == "__main__":
    test_ica_estimator()
