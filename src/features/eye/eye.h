#ifndef _READMINDS_EYE_H_
#define _READMINDS_EYE_H_

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

/*
    Eye related features, by now the implemented feature
    is inner most eye area.
*/

const int EYE_RIGHT_INNER_LMARKS[] = {
    7, 163, 144, 145, 153, 130, 154,
    133, 173, 157, 158, 159, 160, 161,
    190, 246, 243,
};

const int EYE_LEFT_INNER_LMARKS[] =  {
    249, 263, 362, 382, 381, 380, 374,
    373, 388, 387, 386, 385, 384, 390,
    398, 466, 463
};

class Eye
{
public:
    Eye(int width, int height);
    Eye(mediapipe::NormalizedLandmarkList list, int width, int height);
    void SetLandmarks(mediapipe::NormalizedLandmarkList list);
    double GetEyeInnerArea();

private:
    
    mediapipe::NormalizedLandmarkList lmark_list_;
    std::vector<cv::Point> eye_right_contour_;
    std::vector<cv::Point> eye_left_contour_;
    int img_width_, img_height_;

    cv::Point CvtNormIntoCvPoint_(mediapipe::NormalizedLandmark lmark);
    void GenerateEyeContours_();
    double ComputeEyesContoursArea_();
};
#endif