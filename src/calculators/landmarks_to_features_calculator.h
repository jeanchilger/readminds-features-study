// Copyright 2021 The authors

#ifndef SRC_CALCULATORS_LANDMARKS_TO_FEATURES_CALCULATOR_H_
#define SRC_CALCULATORS_LANDMARKS_TO_FEATURES_CALCULATOR_H_

#include "mediapipe/framework/calculator_framework.h"
#include "src/features/eye/eye_analyzer.h"
#include "src/features/face/face_analyzer.h"
#include "src/features/mouth/mouth_analyzer.h"

// Document class here
//
namespace mediapipe {

class LandmarksToFeaturesCalculator : public CalculatorBase {
    public:
        static ::mediapipe::Status GetContract(CalculatorContract* cc);
        ::mediapipe::Status Open(CalculatorContext* cc) override;
        ::mediapipe::Status Process(CalculatorContext* cc) override;
        ::mediapipe::Status Close(CalculatorContext* cc) override;

    private:
        int frame_width_;
        int frame_height_;
        int frame_rate_;

        MouthAnalyzer mouth_descriptor_;
        FaceAnalyzer face_descriptor_;
        EyeAnalyzer eye_descriptor_;
};

}  // namespace mediapipe

#endif  // SRC_CALCULATORS_LANDMARKS_TO_FEATURES_CALCULATOR_H_
