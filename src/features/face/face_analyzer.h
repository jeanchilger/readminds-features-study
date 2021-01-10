#ifndef _READMINDS_FEATURES_FACE_FACE_ANALYZER_H_
#define _READMINDS_FEATURES_FACE_FACE_ANALYZER_H_

#include "mediapipe/framework/formats/landmark.pb.h"
#include "src/features/generic_analyzer.h"

// Class that calculates general face-related features.
//
class FaceAnalyzer : public GenericAnalyzer {

    public:
        FaceAnalyzer() = default;

        FaceAnalyzer(int img_width, int img_height);

        FaceAnalyzer(mediapipe::NormalizedLandmarkList landmarks, 
                int img_width, int img_height);

        // Gets the area of face. This is calculated as 
        // the area of the convex hull that contains all landmarks.
        double GetFaceArea();

        // 
        double GetFaceMotion();

    protected:
        double face_area_;
        double face_motion_;
        int num_frames_motion_ = 50;

        // Calls all other update functions.
        void Update();

        // Calculates the area of the convex hull that closes
        // the whole face.
        void CalculateFaceArea();

        //
        void CalculateFaceMotion();


};

#endif