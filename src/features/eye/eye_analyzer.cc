#include "src/features/eye/eye_analyzer.h"

#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

EyeAnalyzer::EyeAnalyzer(int img_width, int img_height)
    : GenericAnalyzer(img_width, img_height) {}

EyeAnalyzer::EyeAnalyzer(mediapipe::NormalizedLandmarkList landmarks,
            int img_width, int img_height) 
                : GenericAnalyzer(landmarks, img_width, img_height) {
    Update();
}   

void EyeAnalyzer::SetLandmarks(mediapipe::NormalizedLandmarkList landmarks) {
    GenericAnalyzer::SetLandmarks(landmarks);

    Update();
}

void EyeAnalyzer::Initialize(int img_width, int img_height) {
    GenericAnalyzer::Initialize(img_width, img_height);

    Update();
}

void EyeAnalyzer::Initialize(mediapipe::NormalizedLandmarkList landmarks, 
                               int img_width, int img_height) {
    GenericAnalyzer::Initialize(landmarks, img_width, img_height);

    Update();
}

void EyeAnalyzer::Update() {
    UpdateEyeInnerArea();
    UpdateEyebrow();
}

double EyeAnalyzer::GetEyeInnerArea() {
    return eye_area_;
}

double EyeAnalyzer::GetEyebrow() {
    return eyebrow_anchor_dist_sum_;
}

void EyeAnalyzer::UpdateEyeInnerArea() {
    GenerateEyeContours_();
    eye_area_ = ComputeEyesContoursArea_();
}

double EyeAnalyzer::ComputeEyesContoursArea_() {
    double right_eye_area = cv::contourArea(eye_right_contour_);
    double left_eye_area  = cv::contourArea(eye_left_contour_);

    return right_eye_area + left_eye_area;
}

void EyeAnalyzer::GenerateEyeContours_() {
    mediapipe::NormalizedLandmark landmark;
    int eye_right_lmark, eye_left_lmark, eye_area_;
    int lmarks_amount = sizeof(EYE_RIGHT_INNER_LMARKS) / sizeof(int);
    cv::Point lmark_cv_point;

    for (int i=0; i < lmarks_amount; i++) {
         
        eye_right_lmark = EYE_RIGHT_INNER_LMARKS[i];
        landmark = landmarks_.landmark(eye_right_lmark);
        lmark_cv_point = CvtNormIntoCvPoint_(landmark);
        eye_right_contour_.push_back(lmark_cv_point);
        
        eye_left_lmark = EYE_LEFT_INNER_LMARKS[i];
        landmark = landmarks_.landmark(eye_left_lmark);
        lmark_cv_point = CvtNormIntoCvPoint_(landmark);
        eye_left_contour_.push_back(lmark_cv_point);
    }
}

void EyeAnalyzer::UpdateEyebrow() {
    double distances_sum = 0;

    double anchor_x, anchor_y, x, y;
    for (int a : F4_ANCHORS) {
        mediapipe::NormalizedLandmark anchor_landmark = landmarks_.landmark(a);
        
        anchor_x = anchor_landmark.x() * img_width_;
        anchor_y = anchor_landmark.y() * img_height_;

        for (int i : EYE_BROW_RIGHT_UPPER) {
            mediapipe::NormalizedLandmark landmark = landmarks_.landmark(i);

            x = landmark.x() * img_width_;
            y = landmark.y() * img_height_;

            distances_sum += EuclideanDistance(anchor_x, anchor_y, x, y);
        }

        for (int i : EYE_BROW_LEFT_UPPER) {
            mediapipe::NormalizedLandmark landmark = landmarks_.landmark(i);

            x = landmark.x() * img_width_;
            y = landmark.y() * img_height_;

            distances_sum += EuclideanDistance(anchor_x, anchor_y, x, y);
        }
    }

    eyebrow_anchor_dist_sum_ = distances_sum / norm_factor_;
}