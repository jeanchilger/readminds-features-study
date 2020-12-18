#ifndef _READMINDS_FACE_H_
#define _READMINDS_FACE_H_

#include "mediapipe/framework/formats/landmark.pb.h"
#include "src/features/generic_analyzer.h"

// Class that calculates general face-related features.
//
class FaceAnalyzer : public GenericAnalyzer {

    public:
        FaceAnalyzer(int img_width, int img_height);

        FaceAnalyzer(mediapipe::NormalizedLandmarkList landmarks, 
                int img_width, int img_height);

        // landmarks_ setter
        void SetLandmarks(mediapipe::NormalizedLandmarkList landmarks);

        // Gets the area of the polygon that contains all landmarks.
        double GetFaceArea();

    protected:
        double f_face_area_;

        // Calls all other update functions.
        void Update();

        // Calculates the area of the convex hull that closes
        // the whole face.
        void UpdateFaceArea();


};

#endif