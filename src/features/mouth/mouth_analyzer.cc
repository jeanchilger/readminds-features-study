#include "src/features/mouth/mouth_analyzer.h"

#include <vector>

#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

MouthAnalyzer::MouthAnalyzer(int img_width, int img_height)
    : GenericAnalyzer(img_width, img_height) {}

MouthAnalyzer::MouthAnalyzer(mediapipe::NormalizedLandmarkList landmarks, 
              int img_width, int img_height)
                    : GenericAnalyzer{ landmarks, img_width, img_height } {
    Update();
}

void MouthAnalyzer::SetLandmarks(mediapipe::NormalizedLandmarkList landmarks) {
    GenericAnalyzer::SetLandmarks(landmarks);

    Update();
}

double MouthAnalyzer::GetMouthOuter() {
    return m_mouth_outer_;
}

double MouthAnalyzer::GetMouthCorner() {
    return m_mouth_corner_;
}

double MouthAnalyzer::Area() {
    return m_area_;
}

void MouthAnalyzer::Update() {
    CalculateMouthArea();
    CalculateMouthOuter();
    CalculateMouthCorner();
}

void MouthAnalyzer::CalculateMouthArea() {
    std::vector<cv::Point> mouth_contour;

    int x, y;
    for (int i : MOUTH_UPPER_LIP) {
        mediapipe::NormalizedLandmark landmark = landmarks_.landmark(i);

        x = (int) floor(landmark.x() * img_width_);
        y = (int) floor(landmark.y() * img_height_);
        
        mouth_contour.push_back(cv::Point(x, y));
    }

    m_area_ = cv::contourArea(mouth_contour);
}

void MouthAnalyzer::CalculateMouthOuter() {
    double distances_sum = 0;

    double anchor_x, anchor_y, x, y;
    for (int a : ANCHOR_LANDMARKS) {
        mediapipe::NormalizedLandmark anchor_landmark = landmarks_.landmark(a);
        
        anchor_x = anchor_landmark.x() * img_width_;
        anchor_y = anchor_landmark.y() * img_height_;

        for (int i : MOUTH_UPPER_LIP) {
            mediapipe::NormalizedLandmark landmark = landmarks_.landmark(i);

            x = landmark.x() * img_width_;
            y = landmark.y() * img_height_;

            distances_sum += EuclideanDistance(anchor_x, anchor_y, x, y);

        }

        for (int i : MOUTH_LOWER_LIP) {
            mediapipe::NormalizedLandmark landmark = landmarks_.landmark(i);

            x = landmark.x() * img_width_;
            y = landmark.y() * img_height_;

            distances_sum += EuclideanDistance(anchor_x, anchor_y, x, y);
        }
    }

    m_mouth_outer_ = distances_sum / norm_factor_;

}

void MouthAnalyzer::CalculateMouthCorner() {
    double distances_sum = 0;

    double anchor_x, anchor_y, x, y;
    for (int a : ANCHOR_LANDMARKS) {
        mediapipe::NormalizedLandmark anchor_landmark = landmarks_.landmark(a);
        
        anchor_x = anchor_landmark.x() * img_width_;
        anchor_y = anchor_landmark.y() * img_height_;

        for (int i : MOUTH_CORNERS) {
            mediapipe::NormalizedLandmark landmark = landmarks_.landmark(i);

            x = landmark.x() * img_width_;
            y = landmark.y() * img_height_;

            distances_sum += EuclideanDistance(anchor_x, anchor_y, x, y);

        }
    }

    m_mouth_corner_ = distances_sum / norm_factor_;
}