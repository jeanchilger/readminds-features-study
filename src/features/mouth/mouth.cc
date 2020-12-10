#include "src/features/mouth/mouth.h"

#include <iostream>

#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

Mouth::Mouth(int img_width, int img_height) {
    img_width_ = img_width;
    img_height_ = img_height;
}

Mouth::Mouth(mediapipe::NormalizedLandmarkList landmarks, 
              int img_width, int img_height) {
    img_width_ = img_width;
    img_height_ = img_height;

    SetLandmarks(landmarks);
    UpdateMouthArea();
}

void Mouth::SetLandmarks(mediapipe::NormalizedLandmarkList landmarks) {
    landmarks_ = landmarks;
    UpdateMouthArea();
}

double Mouth::Area() {
    return m_area_;
}

void Mouth::UpdateMouthArea() {

    std::vector<cv::Point> mouth_contour;

    int x, y;
    for (int i : MOUTH_UPPER_LIP) {
        mediapipe::NormalizedLandmark landmark = landmarks_.landmark(i);

        x = (int) floor(landmark.x() * img_width_);
        y = (int) floor(landmark.y() * img_height_);
        
        mouth_contour.push_back(cv::Point(x, y));
    }

    // for (cv::Point c : mouth_contour) {
    //     std::cout << "(" << c.x << ", " << c.y << ")\n";
    // }

    m_area_ = cv::contourArea(mouth_contour);
}