#ifndef _READMINDS_FEATURES_EYE_EYE_ANALYZER_H_
#define _READMINDS_FEATURES_EYE_EYE_ANALYZER_H_

#include "src/features/generic_analyzer.h"

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

/*
    Eye related features, by now the implemented feature
    is inner most eye area.
*/

const int EYE_LEFT_INNER_LMARKS[] = {
    7, 163, 144, 145, 153, 130, 154,
    133, 173, 157, 158, 159, 160, 161,
    190, 246, 243,
};

const int EYE_RIGHT_INNER_LMARKS[] =  {
    249, 263, 362, 382, 381, 380, 374,
    373, 388, 387, 386, 385, 384, 390,
    398, 466, 463
};

const int EYE_BROW_LEFT_UPPER[] = {
    70, 63, 105, 66, 107
};

const int EYE_BROW_RIGHT_UPPER[] = {
    300, 293, 334, 296, 336
};

// lower eyebrow landmarks are not
// being used, added for future purposes
const int EYE_BROW_LEFT_LOWER[] = {
    46, 53, 52, 65, 55
};

const int EYE_BROW_RIGHT_LOWER[] = {
    300, 283, 282, 295, 285
};

class EyeAnalyzer : GenericAnalyzer
{
public:
    EyeAnalyzer(int width, int height);

    EyeAnalyzer(mediapipe::NormalizedLandmarkList list, int width, int height);

    // Landmarks list setter
    void SetLandmarks(mediapipe::NormalizedLandmarkList landmarks);

    // Return left and right eyes area
    double GetEyeInnerArea();

    // Return the sum of distances eyebrow landmarks
    // and anchor points normalized
    double GetEyebrow();

private:
 
    std::vector<cv::Point> eye_right_contour_;
    std::vector<cv::Point> eye_left_contour_;
    double eye_area_, eyebrow_anchor_dist_sum_;

    // Updates all feature values each time
    // a new landmark list is set
    void Update();

    // Computes the eye area for landmarks which are
    // closer to the eye
    // Relative to feature F3 in Fernando's paper
    void UpdateEyeInnerArea();

    // Computes the eyebrow landmarks distance sum
    // Relative to feature F4 in Fernando's paper
    void UpdateEyebrow();

    // Creates two vectors of cv::Points that
    // describes eyes contours for further area
    // calculation.
    void GenerateEyeContours_();

    // Compute the right and left eye area
    // using contourArea() from OpenCV
    // and adds this two values
    double ComputeEyesContoursArea_();
};
#endif