#include "src/calculators/landmarks_to_features_calculator.h"

#include <vector>

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

#include "src/features/eye/eye_analyzer.h"
#include "src/features/mouth/mouth_analyzer.h"
#include "src/features/face/face_analyzer.h"

namespace mediapipe {

REGISTER_CALCULATOR(LandmarksToFeaturesCalculator);

::mediapipe::Status LandmarksToFeaturesCalculator::GetContract(
        CalculatorContract* cc) {
    
    cc->InputSidePackets().Tag("FRAME_WIDTH").Set<int>();
    cc->InputSidePackets().Tag("FRAME_HEIGHT").Set<int>();

    cc->Inputs().Tag("LANDMARKS").Set<::std::vector<NormalizedLandmarkList>>();
    cc->Outputs().Tag("VECTOR").Set<::std::vector<double>>();

    return ::mediapipe::OkStatus();
}

::mediapipe::Status LandmarksToFeaturesCalculator::Open(
        CalculatorContext* cc) {
 
    frame_width_ = cc->InputSidePackets().Tag("FRAME_WIDTH").Get<int>();
    frame_height_ = cc->InputSidePackets().Tag("FRAME_HEIGHT").Get<int>();

    ::std::cout << "CALCULATOR SAYS:"
            << "Width: " << frame_width_
            << "Height: " << frame_height_
            << ::std::endl;

    return ::mediapipe::OkStatus();
}

::mediapipe::Status LandmarksToFeaturesCalculator::Process(
        CalculatorContext* cc) {

    // auto& input_landmarks = 

    return ::mediapipe::OkStatus();
}

::mediapipe::Status LandmarksToFeaturesCalculator::Close(
        CalculatorContext* cc) {

    return ::mediapipe::OkStatus();
}

}
