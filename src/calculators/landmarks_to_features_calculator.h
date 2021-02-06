#ifndef READMINDS_CALCULATORS_LANDMARKS_TO_FEATURES_CALCULATOR_
#define READMINDS_CALCULATORS_LANDMARKS_TO_FEATURES_CALCULATOR_

#include "mediapipe/framework/calculator_framework.h"

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
};

} // namespace mediapipe

#endif