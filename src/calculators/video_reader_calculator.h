// Copyright 2021 The authors

#ifndef SRC_CALCULATORS_VIDEO_READER_CALCULATOR_H_
#define SRC_CALCULATORS_VIDEO_READER_CALCULATOR_H_

#include <stdlib.h>
#include <memory>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/status.h"

const int DEVICE_ID = 0;

namespace mediapipe {

class VideoReaderCalculator : public CalculatorBase {
    public:
        static ::mediapipe::Status GetContract(CalculatorContract* cc);
        ::mediapipe::Status Open(CalculatorContext* cc) override;
        ::mediapipe::Status Process(CalculatorContext* cc) override;
        ::mediapipe::Status Close(CalculatorContext* cc) override;

    private:
        std::unique_ptr<cv::VideoCapture> cap_;
        int frame_count_;
        ImageFormat::Format format_;
        int height_;
        int width_;
        int frame_rate_;
        Timestamp prev_timestamp_ = Timestamp::Unset();
        int readed_frames_;
};

}  // namespace mediapipe

#endif  // SRC_CALCULATORS_VIDEO_READER_CALCULATOR_H_
