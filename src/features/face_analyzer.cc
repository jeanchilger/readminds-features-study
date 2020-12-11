#include "src/features/face_analyzer.h"

FaceAnalyzer::FaceAnalyzer(int img_width, int img_height) {
    img_width_ = img_width;
    img_height_ = img_height;
}

FaceAnalyzer::FaceAnalyzer(mediapipe::NormalizedLandmarkList landmarks, 
              int img_width, int img_height) {
    img_width_ = img_width;
    img_height_ = img_height;

    SetLandmarks(landmarks);
}

void FaceAnalyzer::SetLandmarks(mediapipe::NormalizedLandmarkList landmarks) {
    landmarks_ = landmarks;
}

double FaceAnalyzer::EuclideanDistance(cv::Point a, cv::Point b) {

}