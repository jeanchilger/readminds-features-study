#include "src/features/eye/eye.h"

#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

Eye::Eye(int width, int height) {
    img_width_  = width;
    img_height_ = height;
}

Eye::~Eye() {
}

void Eye::SetLandmarks(mediapipe::NormalizedLandmark list) {
    lmark_list_ = list;
}

/*
    Computes the eye area for landmarks which are
    closer to the eye.
    Relative to feature F3 in Fernando's paper.
*/
int Eye::EyeInnerArea() {
    GenerateEyeContours_(EYE_RIGHT_INNER_LMARKS, EYE_LEFT_INNER_LMARKS);
    eyes_area = ComputeEyesContoursArea_();
    return eyes_area;
}

/*
    Creates two vectors of cv::Points that describes eyes
    contours for further area calculation.

*/
void Eye::GenerateEyeContours_(int right_eye_lmarks[], int left_eye_lmarks) {
    
    mediapipe::NormalizedLandmark landmark;
    int eye_right_lmark, eye_left_lmark, eye_area_;
    int lmarks_amount = sizeof(right_eye_lmarks) / sizeof(int);
    cv::Point lmark_cv_point;

    for (int i=0; i < lmarks_amount; i++) {

        eye_right_lmark = right_eye_lmarks[i];
        landmark = lmark_list_.landmark(eye_right_lmark);
        lmark_cv_point = CvtNormIntoCvPoint_(landmark);
        eye_right_contour_.push_back(lmark_cv_point);

        eye_left_lmark  = left_eye_lmarks[i];
        landmark = lmark_list_.landmark(eye_left_lmark);
        lmark_cv_point = CvtNormIntoCvPoint_(landmark);
        eye_left_contour_.push_back(lmark_cv_point);
    }
}

/*
    Calculates the eyes area from contours createad in 
    generate_eyes_contours_()
*/
int ComputeEyesContoursArea_() {
    int right_eye_area = cv::contourArea(eye_right_contour_);
    int left_eye_area  = cv::contourArea(eye_left_contour_);
    return right_eye_area + left_eye_area;
}

/*
    Convert normalized landmarks coordinates into
    a OpenCV point, depth (z) is not been taken
    into account.
*/
cv::Point Eye::CvtNormIntoCvPoint_(mediapipe::NormalizedLandmark lmark) {
    int x = (int) floor(lmark.x() * img_width_);
    int y = (int) floor(lmark.y() * img_height_)
    return cv::Point(x, y);
}