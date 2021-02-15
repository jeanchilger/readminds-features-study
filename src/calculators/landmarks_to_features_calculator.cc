// Copyright 2021 The authors

#include "src/calculators/landmarks_to_features_calculator.h"

#include <vector>

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

REGISTER_CALCULATOR(LandmarksToFeaturesCalculator);

::mediapipe::Status LandmarksToFeaturesCalculator::GetContract(
        CalculatorContract* cc) {
    cc->InputSidePackets().Tag("FRAME_WIDTH").Set<int>();
    cc->InputSidePackets().Tag("FRAME_HEIGHT").Set<int>();
    cc->InputSidePackets().Tag("FPS").Set<int>();

    cc->Inputs().Tag("LANDMARKS").Set<::std::vector<NormalizedLandmarkList>>();
    cc->Outputs().Tag("VECTOR").Set<::std::vector<double>>();

    return ::mediapipe::OkStatus();
}

::mediapipe::Status LandmarksToFeaturesCalculator::Open(
        CalculatorContext* cc) {
    frame_width_ = cc->InputSidePackets().Tag("FRAME_WIDTH").Get<int>();
    frame_height_ = cc->InputSidePackets().Tag("FRAME_HEIGHT").Get<int>();
    frame_rate_ = cc->InputSidePackets().Tag("FPS").Get<int>();

    return ::mediapipe::OkStatus();
}

::mediapipe::Status LandmarksToFeaturesCalculator::Process(
        CalculatorContext* cc) {
    auto& input_landmarks_vector =
            cc->Inputs().Tag("LANDMARKS")
                    .Get<::std::vector<NormalizedLandmarkList>>();

    NormalizedLandmarkList input_landmarks = input_landmarks_vector[0];

    mouth_descriptor_.Initialize(input_landmarks, frame_width_, frame_height_);
    face_descriptor_.Initialize(input_landmarks, frame_width_, frame_height_);
    eye_descriptor_.Initialize(input_landmarks, frame_width_, frame_height_);

    double f1 = mouth_descriptor_.GetMouthOuter();
    double f2 = mouth_descriptor_.GetMouthCorner();
    double f3 = eye_descriptor_.GetEyeInnerArea();
    double f4 = eye_descriptor_.GetEyebrow();
    double f5 = face_descriptor_.GetFaceArea();
    double f6 = face_descriptor_.GetFaceMotion();
    double f7 = face_descriptor_.GetFaceCOM();

    ::std::vector<double> features{ f1, f2, f3, f4, f5, f6, f7 };

    cc->Outputs()
        .Tag("VECTOR")
        .AddPacket(MakePacket<::std::vector<double>>(features)
                        .At(cc->InputTimestamp()));

    return ::mediapipe::OkStatus();
}

::mediapipe::Status LandmarksToFeaturesCalculator::Close(
        CalculatorContext* cc) {
    return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
