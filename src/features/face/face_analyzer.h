#ifndef _READMINDS_FEATURES_FACE_FACE_ANALYZER_H_
#define _READMINDS_FEATURES_FACE_FACE_ANALYZER_H_

#include <deque>

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
        double face_area_ = -1;
        double face_motion_ = -1;

        int num_frames_motion_ = 50;
        std::deque<mediapipe::NormalizedLandmarkList> last_anchors_;

        // Calculates the area of the convex hull that closes
        // the whole face.
        //
        // Relative to feature F5 in Fernando's paper.
        void CalculateFaceArea();

        // Calculates the Euclidean distance between the last 
        // `num_frames_motion` anchor landmarks. The distance is computed
        // between the first anchor list and all others.
        //
        // Represents the amount of movement the head perfored in a short
        // period of time.
        //
        // Relative to feature F6 in Fernando's paper.
        void CalculateFaceMotion();

        // Calls all other update functions.
        void Update() override;

        // Updates the list containing the last `num_frames_motion_` anchor
        // lists. If less than this number of frames have ocurred, it simply
        // adds a new anchor list, otherwise it inserts the new and removes
        // the oldest.
        void UpdateLastAnchors();


};

#endif