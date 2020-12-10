#include "eye.h"

Eye::Eye() {
}

Eye::~Eye() {
}

void Eye::set_image_dimentions(int width, int height) {
    img_width_  = width;
    img_height_ = height;
}

void Eye::set_lmark_list(mediapipe::NormalizedLandmark list) {
    lmark_list_ = list;
}

/*
    Computes the eye area for landmarks which are
    closer to the eye.
    Relative to feature F3 in Fernando's paper.
*/
int Eye::eye_inner_area() {
    generate_eye_contours_(EYE_RIGHT_INNER_LMARKS, EYE_LEFT_INNER_LMARKS);
    eyes_area = compute_eyes_contours_area_();
    return eyes_area;
}

/*
    Creates two vectors of cv::Points that describes eyes
    contours for further area calculation.

*/
void Eye::generate_eye_contours_(int right_eye_lmarks[], int left_eye_lmarks) {
    
    mediapipe::NormalizedLandmark landmark;
    int eye_right_lmark, eye_left_lmark, eye_area_;
    int lmarks_amount = sizeof(right_eye_lmarks) / sizeof(int);
    cv::Point lmark_cv_point;

    for (int i=0; i < lmarks_amount; i++) {

        eye_right_lmark = right_eye_lmarks[i];
        landmark = lmark_list_.landmark(eye_right_lmark);
        lmark_cv_point = cvt_norm_into_cv_point_(landmark);
        eye_right_contour_.push_back(lmark_cv_point);

        eye_left_lmark  = left_eye_lmarks[i];
        landmark = lmark_list_.landmark(eye_left_lmark);
        lmark_cv_point = cvt_norm_into_cv_point_(landmark);
        eye_left_contour_.push_back(lmark_cv_point);
    }
}

/*
    Calculates the eyes area from contours createad in 
    generate_eyes_contours_()
*/
int compute_eyes_contours_area_() {
    int right_eye_area = cv::contourArea(eye_right_contour_);
    int left_eye_area  = cv::contourArea(eye_left_contour_);
    return right_eye_area + left_eye_area;
}

/*
    Convert normalized landmarks coordinates into
    a OpenCV point, depth (z) is not been taken
    into account.
*/
cv::Point Eye::cvt_norm_into_cv_point_(mediapipe::NormalizedLandmark lmark) {
    int x = (int) floor(lmark.x() * img_width_);
    int y = (int) floor(lmark.y() * img_height_)
    return cv::Point(x, y);
}