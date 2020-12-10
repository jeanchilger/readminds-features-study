#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

const int EYE_RIGHT[] = {
    7, 163, 144, 145, 153, 130, 154,
    133, 173, 157, 158, 159, 160, 161,
    190, 246, 243,
};

const int EYE_LEFT[] =  {
    249, 263, 362, 382, 381, 380, 374,
    373, 388, 387, 386, 385, 384, 390,
    398, 466, 463
};

const int EYE_LMARK_SIZE = 17;

cv::Point cvt_norm_into_scalar(mediapipe::NormalizedLandmark, int w, int h){
    int x = (int) floor(landmark.x() * w);
    int y = (int) floor(landmark.y() * h)
    return cv::Point(x, y);
}

/*
    Gets a normalized landmarks list, image width
    and height and returns the eye area.
*/
int eye_area(mediapipe::NormalizedLandmarkList face_landmarks,
            int img_width, int img_height) {

    std::vector<cv::Point> eye_right_contour;
    std::vector<cv::Point> eye_left_contour;
    mediapipe::NormalizedLandmark landmark;
    int eye_right_lmark, eye_left_lmark, area;
    cv::Point lmark_point;

    for (int i=0; i < EYE_LMARK_SIZE; i++) {

        eye_right_lmark = EYE_RIGHT[i];
        landmark = face_landmarks.landmark(eye_right_lmark);
        lmark_point = cvt_norm_into_scalar(landmark, img_width, img_height);
        eye_right_contour.push_back(lmark_point);

        eye_left_lmark  = EYE_LEFT[i];
        landmark = face_landmarks.landmark(eye_left_lmark);
        lmark_point = cvt_norm_into_scalar(landmark, img_width, img_height);
        eye_left_contour.push_back(lmark_point);
    }

    area = cv::contourArea(eye_right_contour) + cv::contourArea(eye_left_contour);

    return area;
}
