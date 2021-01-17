#include "src/calculators/video_reader_calculator.h"

#include <stdlib.h>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
// #include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/status_util.h"

namespace mediapipe {

// Returns a ImageFormat based on an input.
// This input is intended to be the number of channels 
// in source image.
ImageFormat::Format GetImageFormat(int num_channels) {
    ImageFormat::Format format;

    switch (num_channels) {
        case 1: {
            format = ImageFormat::GRAY8;
            break;
        }

        case 3: {
            format = ImageFormat::SRGB;
            break;
        }

        case 4: {
            format = ImageFormat::SRGBA;
            break;
        }

        default: {
            format = ImageFormat::UNKNOWN;
            break;
        }
    }

    return format;
}

REGISTER_CALCULATOR(VideoReaderCalculator);

::mediapipe::Status VideoReaderCalculator::GetContract(
        CalculatorContract* cc) {
    cc->InputSidePackets().Tag("VIDEO_STREAM").Set<std::string>();
    cc->Outputs().Tag("IMAGE").Set<ImageFrame>();

    return mediapipe::OkStatus();
}

::mediapipe::Status VideoReaderCalculator::Open(CalculatorContext* cc) {
    const std::string file_path =
        cc->InputSidePackets().Tag("VIDEO_STREAM").Get<std::string>();
    
    if (file_path.empty()) {
        cap_ = absl::make_unique<cv::VideoCapture>(DEVICE_ID, cv::CAP_ANY);
        cap_->open(0, cv::CAP_ANY); 
    }

    else {
        cap_ = absl::make_unique<cv::VideoCapture>(file_path);
    }

    if (!cap_->isOpened()) {
        return InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
        << "Fail to open video file at " << file_path;
    }

    frame_count_ = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_COUNT)) -1;

    cv::Mat frame;
    cap_->read(frame);

    if (frame.empty()) {
        return InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
            << "Fail to read any frames from the video file at "
            << file_path;
    }

    format_ = GetImageFormat(frame.channels());
    width_ = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_HEIGHT));
    
    if (format_ == ImageFormat::UNKNOWN) {
        return InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
            << "Unsupported video format of the video file at "
            << file_path;
    }

    cap_->set(cv::CAP_PROP_POS_AVI_RATIO, 0);

    return ::mediapipe::OkStatus();
}

::mediapipe::Status VideoReaderCalculator::Process(CalculatorContext* cc) {
    auto image_frame = absl::make_unique<ImageFrame>(format_, 
                                                     width_, 
                                                     height_, 
                                                     1);

    Timestamp timestamp(cap_->get(cv::CAP_PROP_POS_MSEC) * 1000);

    if (format_ == ImageFormat::GRAY8) {
        cv::Mat frame = formats::MatView(image_frame.get());
        cap_->read(frame);

        if (frame.empty()) {
            return tool::StatusStop();
        }

    } else {
        cv::Mat tmp_frame;
        cap_->read(tmp_frame);

        if (tmp_frame.empty()) {
            return tool::StatusStop();
        }

        if (format_ == ImageFormat::SRGB) {
            cv::cvtColor(tmp_frame, formats::MatView(image_frame.get()),
                        cv::COLOR_BGR2RGB);

        } else if (format_ == ImageFormat::SRGBA) {
            cv::cvtColor(tmp_frame, formats::MatView(image_frame.get()),
                        cv::COLOR_BGRA2RGBA);
        }
    }

    if (prev_timestamp_ < timestamp) {
        cc->Outputs().Tag("IMAGE").Add(image_frame.release(), timestamp);
        prev_timestamp_ = timestamp;
        readed_frames_++;
    }

    return ::mediapipe::OkStatus();
}

::mediapipe::Status VideoReaderCalculator::Close(CalculatorContext* cc) {
    if (cap_ && cap_->isOpened()) {
        cap_->release();
    }

    if (readed_frames_ < frame_count_) {
        LOG(WARNING) << "Not all the frames are decoded (total frames: "
            << frame_count_ << " vs decoded frames: " << readed_frames_
            << ").";
        
        return ::mediapipe::OkStatus();
    }

    return ::mediapipe::OkStatus();
}

} // namespace
