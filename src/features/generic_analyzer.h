#ifndef _READMINDS_FEATURES_FACE_H_
#define _READMINDS_FEATURES_FACE_H_

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"

const int ANCHOR_LANDMARKS[] = {
    1, 4, 5, 195, 197, 6,
};

// F4 feature anchors, landmarks
// 1 and 4 was not needed
const int F4_ANCHORS[] = {
    5, 195, 197, 6,
};

// Face object providing generic data used by
// other features, such as A and K.
class GenericAnalyzer {
    public:
        GenericAnalyzer() = default;

        GenericAnalyzer(int img_width, int img_height);
        
        GenericAnalyzer(mediapipe::NormalizedLandmarkList landmarks, 
                     int img_width, int img_height);

        virtual ~GenericAnalyzer() = default;

        // landmarks_ setter
        void SetLandmarks(mediapipe::NormalizedLandmarkList landmarks);

        // Sets all needed attributes
        void Initialize(int img_width, int img_height);
        
        void Initialize(mediapipe::NormalizedLandmarkList landmarks, 
                        int img_width, int img_height);

    protected:
        double norm_factor_;

        mediapipe::NormalizedLandmarkList landmarks_;
        int img_width_;
        int img_height_;

        // Implements the euclidean distance between
        // two cv::Points
        double EuclideanDistance(cv::Point a, cv::Point b);
        
        double EuclideanDistance(double x1, double y1, double x2, double y2);

        // Updates the normalize factor with landmars value
        void SetNormFactor();

};

#endif