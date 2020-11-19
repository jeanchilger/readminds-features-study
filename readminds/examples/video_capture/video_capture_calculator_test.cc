#include "mediapipe/framework/calculator_runner.h"
#include "mediapipe/framework/deps/status_matchers.h"
#include "mediapipe/framework/port/gmock.h"
#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {

    namespace {
        
        TEST(VideoCaptureCalculatorTest, ReadVideoFrames) {

            std::string video_file = "video-1.mp4";

            CalculatorRunner runner(R"pb(
                node {
                    calculator: "VideoCaptureCalculator"
                    input_stream: "FILE_PATH:input_file"
                    output_stream: "OUTPUT_FRAME:output_frame"
                }
            )pb");

            MP_ASSERT_OK(runner.Run());
            
            runner.MutableInputs()->Tag("FILE_PATH").packets.push_back(
                MakePacket<std::string>(video_file).At(Timestamp(1))
            );

            MP_ASSERT_OK(runner.Run());

            const std::vector<Packet>& outputs = runner.Outputs().Tag("OUTPUT_FRAME").packets;

            std::cout << outputs.size() << '\n' ;

        }

    }

}