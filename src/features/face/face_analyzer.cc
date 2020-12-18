#include <vector>

#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "src/features/face/face_analyzer.h"

FaceAnalyzer::FaceAnalyzer(int img_width, int img_height)
    : GenericAnalyzer(img_width, img_height) {}

FaceAnalyzer::FaceAnalyzer(mediapipe::NormalizedLandmarkList landmarks, 
              int img_width, int img_height)
                    : GenericAnalyzer{ landmarks, img_width, img_height } {
    Update();
}

void FaceAnalyzer::SetLandmarks(mediapipe::NormalizedLandmarkList landmarks) {
    GenericAnalyzer::SetLandmarks(landmarks);

    Update();
}

double FaceAnalyzer::GetFaceArea() {
    return f_face_area_;
}

// TODO: This may be a ineffective approach.
// Selecting only the outer landmarks could be faster.
void FaceAnalyzer::UpdateFaceArea() {
    std::vector<cv::Point> all_points;
    std::vector<cv::Point> hull;

    int x, y;
    for (int i=0; i < 468; i ++) {
        mediapipe::NormalizedLandmark landmark = landmarks_.landmark(i);

        x = (int) floor(landmark.x() * img_width_);
        y = (int) floor(landmark.y() * img_height_);
        
        all_points.push_back(cv::Point(x, y));
    }

    cv::convexHull(all_points, hull);

    f_face_area_ = cv::contourArea(hull) / norm_factor_;

}

void FaceAnalyzer::Update() {
    UpdateFaceArea();
}