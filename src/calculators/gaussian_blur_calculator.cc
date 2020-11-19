#include <cstdio>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status.h"

//

namespace mediapipe {

const int KERNEL_SIZE = 15;
const double SIGMA_X = 1.0;
const double SIGMA_Y = 0.0;

class GaussianBlurCalculator : public CalculatorBase {
    public:
        static mediapipe::Status GetContract(CalculatorContract* cc) {
            if (cc->Inputs().HasTag("IMAGE")) {
                cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
            }

            if (cc->Outputs().HasTag("IMAGE")) {
                cc->Outputs().Tag("IMAGE").Set<ImageFrame>();
            }

            return mediapipe::OkStatus();
        }

        mediapipe::Status Open(CalculatorContext* cc) {
            ksize_ = KERNEL_SIZE;
            sigma_x_ = SIGMA_X;
            sigma_y_ = SIGMA_Y;

            return mediapipe::OkStatus();
        }

        mediapipe::Status Process(CalculatorContext* cc) {
            // Retrieves the ImageFrame in stream.
            const ImageFrame& input_image = cc->Inputs()
                    .Tag("IMAGE")
                    .Get<ImageFrame>();

            // Gets the OpenCV Mat of that ImageFrame
            cv::Mat input_mat = formats::MatView(&input_image);

            // Creates a unique_ptr for the input frame
            std::unique_ptr<ImageFrame> output_image(new ImageFrame(
                input_image.Format(),
                input_mat.cols,
                input_mat.rows
            ));

            // Gets the poiter for output_image and creates a cv::Mat of it
            cv::Mat output_mat = formats::MatView(output_image.get());

            // All the operations may be performed on output_mat. The final
            // result is then hold at output_image, since they are bound by a 
            // pointer (output_image is this "bounding" pointer actually).
            // output_mat
            cv::GaussianBlur(
                input_mat,
                output_mat,
                cv::Size(ksize_, ksize_),
                sigma_x_,
                sigma_y_
            );

            cc->Outputs().Tag("IMAGE").Add(
                output_image.release(),
                cc->InputTimestamp()
            );

            return mediapipe::OkStatus();
        }

    private:
        // GaussianBlurCalculatorOptions options_;
        int ksize_;
        double sigma_x_;
        double sigma_y_;
};

REGISTER_CALCULATOR(GaussianBlurCalculator);

} // namespace mediapipe
