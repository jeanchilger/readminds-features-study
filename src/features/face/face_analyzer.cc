#include "src/features/face/face_analyzer.h"

#include <vector>

#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

// FaceAnalyzer::FaceAnalyzer() : GenericAnalyzer() {}

FaceAnalyzer::FaceAnalyzer(int img_width, int img_height)
    : GenericAnalyzer(img_width, img_height) {}

FaceAnalyzer::FaceAnalyzer(mediapipe::NormalizedLandmarkList landmarks, 
              int img_width, int img_height)
                    : GenericAnalyzer(landmarks, img_width, img_height) {}


double FaceAnalyzer::GetFaceArea() {
    return face_area_;
}

// TODO: This may be a ineffective approach.
// Selecting only the outer landmarks could be faster.
void FaceAnalyzer::CalculateFaceArea() {
    std::vector<cv::Point> all_points;
    std::vector<cv::Point> hull;

    int x, y;
    for (int i=0; i < 468; i ++) {
        mediapipe::NormalizedLandmark landmark = landmarks_.landmark(i);
        
        all_points.push_back(CvtNormIntoCvPoint(landmark));
    }

    cv::convexHull(all_points, hull);

    face_area_ = cv::contourArea(hull) / norm_factor_;

}

void FaceAnalyzer::CalculateFaceMotion() {

}

void FaceAnalyzer::Update() {
    CalculateFaceArea();
    CalculateFaceMotion();
}