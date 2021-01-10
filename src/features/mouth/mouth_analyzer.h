#ifndef _READMINDS_FEATURES_MOUTH_MOUTH_ANALYZER_H_
#define _READMINDS_FEATURES_MOUTH_MOUTH_ANALYZER_H_

#include "mediapipe/framework/formats/landmark.pb.h"
#include "src/features/generic_analyzer.h"

// Landmarks related to upper lip region.
const int MOUTH_UPPER_LIP[] = {
    61, 185, 40, 39, 37, 0, 267, 269,
    270, 409, 291,
};

// Landmarks related to lower lip region.
const int MOUTH_LOWER_LIP[] = {
    146, 91, 181, 84, 17, 314, 405,
    321, 375,
};

// Landmarks for mouth corners.
const int MOUTH_CORNERS[] = {
    57, 287,
};

// Class that wraps mouth-related features.
//
class MouthAnalyzer : public GenericAnalyzer {

    public:
        MouthAnalyzer() = default;

        MouthAnalyzer(int img_width, int img_height);

        MouthAnalyzer(mediapipe::NormalizedLandmarkList landmarks, 
              int img_width, int img_height);

        // Get the area of the mouth. Area is calculated using OpenCV's contourArea().
        // The mouth shape is defined as a combination of all outer points of the
        // upper and the lower lip.
        double GetMouthArea();

        // Gets the sum of the distances between mouth contour landmarks and 
        // Anchor landmarks. The sum is normalized by the K factor.
        double GetMouthOuter();
        
        // Gets the sum of distances between mouth corner landmarks and
        // Anchor landmarks. The sum is normalized by the K factor.
        double GetMouthCorner();

    private:
        double mouth_area_;
        double mouth_outer_;
        double mouth_corner_;

        // Calls all calculation functions.
        void Update();

        // Calculates the area of the mouth region.
        void CalculateMouthArea();

        // Calculates the distances between mouth outer landmarks and 
        // anchor landmarks.
        void CalculateMouthOuter();
        
        // Calculates the distances between mouth corner landmarks 
        // and anchor landmarks.
        void CalculateMouthCorner();

};

#endif