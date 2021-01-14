#include <cmath>
#include <iostream>

#include "src/features/generic_analyzer.h"

// PUBLIC

// GenericAnalyzer::GenericAnalyzer() {

// }

GenericAnalyzer::GenericAnalyzer(int img_width, int img_height) {
    Initialize(img_width, img_height);
}

GenericAnalyzer::GenericAnalyzer(mediapipe::NormalizedLandmarkList landmarks, 
                                 int img_width, int img_height) {
    Initialize(landmarks, img_width, img_height);
}

void GenericAnalyzer::SetLandmarks(mediapipe::NormalizedLandmarkList landmarks) {
    landmarks_ = landmarks;

    CalculateNormFactor();
    Update();
}

void GenericAnalyzer::Initialize(int img_width, int img_height) {
    img_width_ = img_width;
    img_height_ = img_height;
}
        
void GenericAnalyzer::Initialize(mediapipe::NormalizedLandmarkList landmarks, 
                                 int img_width, int img_height) {
    img_width_ = img_width;
    img_height_ = img_height;

    SetLandmarks(landmarks);
}

// PROTECTED

double GenericAnalyzer::EuclideanDistance(cv::Point a, cv::Point b) {
    cv::Point diff = a - b;

    return std::sqrt(diff.x * diff.x + diff.y * diff.y);
}

double GenericAnalyzer::EuclideanDistance(
        double x1, 
        double y1, 
        double x2, 
        double y2) {

    return std::sqrt(x1 * x2 + y1 * y2);
}

void GenericAnalyzer::CalculateNormFactor() {
    mediapipe::NormalizedLandmark first_anchor = 
            landmarks_.landmark(ANCHOR_LANDMARKS[0]);

    int last_anchor_idx = sizeof(ANCHOR_LANDMARKS) / 
            sizeof(*ANCHOR_LANDMARKS) - 1;
    mediapipe::NormalizedLandmark last_anchor = 
            landmarks_.landmark(ANCHOR_LANDMARKS[last_anchor_idx]);

    norm_factor_ = EuclideanDistance(
            cv::Point(
                    (int) floor(first_anchor.x() * img_width_),
                    (int) floor(first_anchor.y() * img_height_)),
            cv::Point(
                    (int) floor(last_anchor.x() * img_width_),
                    (int) floor(last_anchor.y() * img_height_)));
}

cv::Point GenericAnalyzer::CvtNormIntoCvPoint(mediapipe::NormalizedLandmark landmark) {
    int x = (int) floor(landmark.x() * img_width_);
    int y = (int) floor(landmark.y() * img_height_);

    return cv::Point(x, y);
}

double GenericAnalyzer::EuclideanNorm(cv::Point landmark) {
    return std::sqrt(pow(landmark.x, 2) + pow(landmark.y, 2));
}