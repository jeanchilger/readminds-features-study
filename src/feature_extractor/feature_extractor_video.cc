// Copyright 2021 The authors

#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

#include "src/calculators/array_to_csv_row_calculator.pb.h"
#include "src/features/face/face_analyzer.h"
#include "src/features/mouth/mouth_analyzer.h"
#include "src/features/eye/eye_analyzer.h"

// Extracts features from video stream.
//
// Build:
//     bazel build --define MEDIAPIPE_DISABLE_GPU=1 \
    //src/feature_extractor:feature_extractor_video --nocheck_visibility \
    && bazel-bin/src/feature_extractor/feature_extractor_video \
    --frame_width=640 --frame_height=480 --frame_rate=30
// RUN:
//     bazel-bin/src/feature_extractor/feature_extractor_video

// Validates both the frame_width and frame_height
// flags.
static bool ValidateFrameSizeFlag(const char* flagname, int32 value) {
    if (value > 0 && value < 20000) {
        return true;
    }

    printf("Invalid value for --%s: %d\n", flagname, static_cast<int>(value));

    return false;
}

static bool ValidateFrameRateFlag(const char* flagname, int32 value) {
    if (value > 0 && value < 500) {
        return true;
    }

    printf("Invalid value for --%s: %d\n", flagname, static_cast<int>(value));

    return false;
}

DEFINE_int32(
        frame_width, -1,
        "Frame width for the input video/camera.");
DEFINE_validator(frame_width, &ValidateFrameSizeFlag);

DEFINE_int32(
        frame_height, -1,
        "Frame height for the input video/camera.");
DEFINE_validator(frame_height, &ValidateFrameSizeFlag);

DEFINE_int32(
        frame_rate, -1,
        "Frame rate for the input video/camera. Expressed in frames per second.");
DEFINE_validator(frame_rate, &ValidateFrameRateFlag);

DEFINE_string(
        feature_file_path, "data/dataset/test.csv",
        "Path to store the resulting file.");

DEFINE_string(
    video_source, "",
    "Path to input video, empty if needed to read from a camera.");

namespace mediapipe {

mediapipe::Status RunVideoReader() {
    // TODO(@jeanchilger):
    //      - Removes a window of N seconds from beggining
    CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "in"
        input_stream: "video_width"
        input_stream: "video_height"
        input_stream: "video_fps"
        output_stream: "out_image"
        output_stream: "out_vector"
        node {
            calculator: "VideoReaderCalculator"
            input_side_packet: "VIDEO_STREAM:in"
            output_stream: "IMAGE:out_image"
        }
        node: {
            calculator: "FaceLandmarkFrontCpu"
            input_stream: "IMAGE:out_image"
            output_stream: "LANDMARKS:out_landmarks"
        }
        node: {
            calculator: "LandmarksToFeaturesCalculator"
            input_side_packet: "FRAME_WIDTH:video_width"
            input_side_packet: "FRAME_HEIGHT:video_height"
            input_side_packet: "FPS:video_fps"
            input_stream: "LANDMARKS:out_landmarks"
            output_stream: "VECTOR:out_vector"
        }
        node {
            calculator: "DoubleVectorToCsvRowCalculator"
            input_stream: "out_vector"
            node_options: {
                [type.googleapis.com/mediapipe.ArrayToCsvRowCalculatorOptions] {
                    file_path: ")" + FLAGS_feature_file_path + R"("
                    header: [
                        "mouth_outer",
                        "mouth_corner",
                        "eye_area",
                        "eyebrow_activity",
                        "face_area",
                        "face_motion",
                        "facial_com"
                    ]
                }
            }
        }
    )");

    ::std::map<::std::string, Packet> input_side_packets;
    input_side_packets["in"] = MakePacket<::std::string>(FLAGS_video_source);
    input_side_packets["video_width"] = MakePacket<int>(FLAGS_frame_width);
    input_side_packets["video_height"] = MakePacket<int>(FLAGS_frame_height);
    input_side_packets["video_fps"] = MakePacket<int>(FLAGS_frame_rate);

    CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config, input_side_packets));

    // Adds output pollers to image and landmark output streams.
    ASSIGN_OR_RETURN(OutputStreamPoller poller_image,
                   graph.AddOutputStreamPoller("out_image"));

    ASSIGN_OR_RETURN(OutputStreamPoller poller_vector,
                   graph.AddOutputStreamPoller("out_vector"));

    MP_RETURN_IF_ERROR(graph.StartRun({}));

    mediapipe::Packet image_packet;
    mediapipe::Packet vector_packet;

    while (poller_vector.Next(&vector_packet)) {
        // Get landmarks from output
        auto& output_vector =
                vector_packet.Get<::std::vector<double> >();

        if (!output_vector.empty()) {
            for (double x : output_vector) {
                ::std::cout << " " << x;
            }

            ::std::cout << "\n";
        }

        // Get Image from output
        poller_image.Next(&image_packet);

        auto& output_frame = image_packet.Get<ImageFrame>();
        cv::Mat output_frame_mat = formats::MatView(&output_frame);

        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

        cv::imshow("Output Image", output_frame_mat);

        char c = static_cast<char>(cv::waitKey(1));
        if (c == 27) {
            break;
        }
    }

    cv::destroyAllWindows();

    return ::mediapipe::OkStatus();
}

}  // namespace mediapipe


int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    mediapipe::Status output_status = mediapipe::RunVideoReader();

    std::cout << output_status.message() << std::endl;

    return EXIT_SUCCESS;
}
