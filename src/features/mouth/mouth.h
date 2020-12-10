#ifndef _READMINDS_MOUTH_H_
#define _READMINDS_MOUTH_H_

#include "mediapipe/framework/formats/landmark.pb.h"

// Class that wraps mouth-related features.
//

const int MOUTH_UPPER_LIP[] = {
    61, 185, 40, 39, 37, 0, 267, 269,
    270, 409, 291,
};

const int MOUTH_LOWER_LIP[] = {
    146, 91, 181, 84, 17, 314, 405,
    321, 375,
};

class Mouth {

    public:
        Mouth(int img_width, int img_height);

        Mouth(mediapipe::NormalizedLandmarkList landmarks, 
              int img_width, int img_height);

        // landmarks_ setter
        void SetLandmarks(mediapipe::NormalizedLandmarkList landmarks);

        // Get the area of the mouth. Area is calculated using OpenCV's contourArea().
        // The mouth shape is defined as a combination of all outer points of the
        // upper and the lower lip.
        double Area();

    private:
        int img_width_;
        int img_height_;
        mediapipe::NormalizedLandmarkList landmarks_;

        double m_area_;

        // Calculates the area of the mouth. 
        void UpdateMouthArea();

};

#endif