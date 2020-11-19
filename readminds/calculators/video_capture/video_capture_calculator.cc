#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/status_util.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"

namespace mediapipe {

    namespace {
        ImageFormat::Format GetImageFormat(int num_channels) {
            ImageFormat::Format format;
            switch (num_channels) {
                case 1:
                format = ImageFormat::GRAY8;
                break;
                case 3:
                format = ImageFormat::SRGB;
                break;
                case 4:
                format = ImageFormat::SRGBA;
                break;
                default:
                format = ImageFormat::UNKNOWN;
                break;
            }
            return format;
        }
    }

    class VideoCaptureCalculator : public CalculatorBase {

        public:
            static mediapipe::Status GetContract(CalculatorContract* cc) {
                cc->InputSidePackets().Tag("FILE_PATH").Set<std::string>();
                cc->Outputs().Tag("OUTPUT_FRAME").Set<ImageFrame>();
            }

            Status Open(CalculatorContext* cc) override {
                const std::string& file_path = cc->Inputs().Tag("FILE_PATH").Get<std::string>();
                cap_ = absl::make_unique<cv::VideoCapture>(file_path);

                if (!cap_->isOpened()) {
                    return InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                    << "Fail to open video file at " << file_path;
                }

                width_ = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_WIDTH));
                height_ = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_HEIGHT));
                double fps = static_cast<double>(cap_->get(cv::CAP_PROP_FPS));
                frame_count_ = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_COUNT));

                cv::Mat frame;
                cap_->read(frame);
                if(frame.empty()) {
                    return InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                    << "Fail to read any frames from the video file at "
                    << file_path;
                }

                format_ = GetImageFormat(frame.channels());
                if (format_ == ImageFormat::UNKNOWN) {
                    return InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                            << "Unsupported video format of the video file at "
                            << file_path;
                }

                if (fps <= 0 || frame_count_ <= 0 || width_ <= 0 || height_ <= 0) {
                    return InvalidArgumentErrorBuilder(MEDIAPIPE_LOC)
                        << "Fail to make video header due to the incorrect metadata from "
                            "the video file at "
                        << file_path;
                }
                auto header = absl::make_unique<VideoHeader>();
                header->format = format_;
                header->width = width_;
                header->height = height_;
                header->frame_rate = fps;
                header->duration = frame_count_ / fps;

                // rewind frame reading to first one
                cap_->set(cv::CAP_PROP_POS_AVI_RATIO, 0);

                return OkStatus();
            }

            Status Process(CalculatorContext* cc) override {
                auto image_frame = absl::make_unique<ImageFrame>(format_, width_, height_, 1);

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
                    // image_frame.release() frees memory
                    cc->Outputs().Tag("OUTPUT_FRAME").Add(image_frame.release(), timestamp);
                    prev_timestamp_ = timestamp;
                    readed_frames_++;
                }

                return OkStatus();
            }
            Status Close(CalculatorContext* cc) override {
                
                if(cap_ && cap_->isOpened()) {
                    cap_->release();
                }

                if(readed_frames_ < frame_count_) {
                    LOG(WARNING) << "Not all the frames are decoded (total frames: "
                                 << frame_count_ << " vs decoded frames: " << readed_frames_
                                 << ").";
                }
                return OkStatus();
            }

        private:
            std::unique_ptr<cv::VideoCapture> cap_;
            int width_;
            int height_;
            int frame_count_;
            int readed_frames_;
            ImageFormat::Format format_;
            Timestamp prev_timestamp_ = Timestamp::Unset();
    };

    REGISTER_CALCULATOR(VideoCaptureCalculator);
}
