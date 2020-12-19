#include "src/features/eye/eye_analyzer.h"

#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

EyeAnalyzer::EyeAnalyzer(mediapipe::NormalizedLandmarkList list,
             int width, int height) {
    img_width_  = width;
    img_height_ = height;
    lmark_list_ = list;
}

EyeAnalyzer::EyeAnalyzer(int width, int height) {
    img_width_  = width;
    img_height_ = height;
}

void EyeAnalyzer::SetLandmarks(mediapipe::NormalizedLandmarkList list) {
    lmark_list_ = list;
}

/*
    Computes the eye area for landmarks which are
    closer to the eye.
    Relative to feature F3 in Fernando's paper.
*/
double EyeAnalyzer::GetEyeInnerArea() {
    GenerateEyeContours_();
    double eyes_area = ComputeEyesContoursArea_();
    return eyes_area;
}

/*
    Creates two vectors of cv::Points that describes eyes
    contours for further area calculation.
*/
void EyeAnalyzer::GenerateEyeContours_() {
    
    mediapipe::NormalizedLandmark landmark;
    int eye_right_lmark, eye_left_lmark, eye_area_;
    int lmarks_amount = sizeof(EYE_RIGHT_INNER_LMARKS) / sizeof(int);
    cv::Point lmark_cv_point;

    for (int i=0; i < lmarks_amount; i++) {
         
        eye_right_lmark = EYE_RIGHT_INNER_LMARKS[i];
        landmark = lmark_list_.landmark(eye_right_lmark);
        lmark_cv_point = CvtNormIntoCvPoint_(landmark);
        eye_right_contour_.push_back(lmark_cv_point);

        eye_left_lmark  = EYE_LEFT_INNER_LMARKS[i];
        landmark = lmark_list_.landmark(eye_left_lmark);
        lmark_cv_point = CvtNormIntoCvPoint_(landmark);
        eye_left_contour_.push_back(lmark_cv_point);
    }
}

/*
    Calculates the eyes area from contours createad in 
    generate_eyes_contours_()
*/
double EyeAnalyzer::ComputeEyesContoursArea_() {
    double right_eye_area = cv::contourArea(eye_right_contour_);
    double left_eye_area  = cv::contourArea(eye_left_contour_);
    return right_eye_area + left_eye_area;
}

/*
    Convert normalized landmarks coordinates into
    a OpenCV point, depth (z) is not been taken
    into account (yet).
*/
cv::Point EyeAnalyzer::CvtNormIntoCvPoint_(mediapipe::NormalizedLandmark lmark) {
    int x = (int) floor(lmark.x() * img_width_);
    int y = (int) floor(lmark.y() * img_height_);
    return cv::Point(x, y);
}