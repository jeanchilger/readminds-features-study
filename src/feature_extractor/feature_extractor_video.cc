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

#include "src/features/face/face_analyzer.h"
#include "src/features/mouth/mouth_analyzer.h"
#include "src/features/eye/eye_analyzer.h"

// Extracts features from video stream.
//
// Build: 
//     bazel build --define MEDIAPIPE_DISABLE_GPU=1 //src/feature_extractor:feature_extractor_video --nocheck_visibility
// RUN:
//     bazel-bin/src/feature_extractor/feature_extractor_video

namespace mediapipe {

mediapipe::Status RunVideoReader() {

    CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "in"
        output_stream: "out_landmarks"
        output_stream: "out_image"
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
    )");

    std::map<std::string, Packet> input_side_packets;
    input_side_packets["in"] = MakePacket<std::string>();

    CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config, input_side_packets));

    // Adds output pollers to image and landmark output streams.
    ASSIGN_OR_RETURN(OutputStreamPoller poller_image,
                   graph.AddOutputStreamPoller("out_image"));

    ASSIGN_OR_RETURN(OutputStreamPoller poller_landmarks,
                   graph.AddOutputStreamPoller("out_landmarks"));

    MP_RETURN_IF_ERROR(graph.StartRun({}));

    mediapipe::Packet image_packet;
    mediapipe::Packet landmarks_packet;

    // Instantiate analyzers
    MouthAnalyzer mouth_descriptor;
    FaceAnalyzer face_descriptor;
    EyeAnalyzer eye_descriptor;

    while (poller_landmarks.Next(&landmarks_packet)) {

        // Get landmarks from output
        auto& output_landmark_vector = 
                landmarks_packet.Get<std::vector<NormalizedLandmarkList> >();

        NormalizedLandmarkList face_landmarks = output_landmark_vector[0];
        
        // Get Image from output
        poller_image.Next(&image_packet);

        auto& output_frame = image_packet.Get<ImageFrame>();
        cv::Mat output_frame_mat = formats::MatView(&output_frame);

        int width = output_frame.Width();
        int height = output_frame.Height();

        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

        cv::imshow("Output Image", output_frame_mat);

        mouth_descriptor.Initialize(face_landmarks, width, height);
        face_descriptor.Initialize(face_landmarks, width, height);
        eye_descriptor.Initialize(face_landmarks, width, height);

        char c = (char) cv::waitKey(1);
        if (c == 27) {
            break;
        
        } else if (c == 13) {

            // ===========================
            // F1
            // ===========================
            double f1 = mouth_descriptor.GetMouthOuter();
            std::cout << "F1: " << f1 << std::endl;

            // ===========================
            // F2
            // ===========================
            double f2 = mouth_descriptor.GetMouthCorner();
            std::cout << "F2: " << f2 << std::endl;

            // ===========================
            // F3
            // ===========================
            double f3 = eye_descriptor.GetEyeInnerArea();
            std::cout << "F3: " << f3 << std::endl;

            // ===========================
            // F4
            // ===========================
            double f4 = eye_descriptor.GetEyebrow();
            std::cout << "F4: " << f4 << std::endl;
            
            //============================
            // F5
            // ===========================
            double f5 = face_descriptor.GetFaceArea();
            std::cout << "F5: " << f5 << std::endl;

            //============================
            // F6
            // ===========================
            double f6 = face_descriptor.GetFaceMotion();
            if (f6 > 0) {
                std::cout << "F6: " << f6 << std::endl;
            }

            //============================
            // F7
            // ===========================
            double f7 = face_descriptor.GetFaceCOM();
            if (f7 > 0) {
                std::cout << "F7: " << f7 << std::endl;
            }

            std::cout << "======================================\n\n";
        }   
    }

    cv::destroyAllWindows();

    return ::mediapipe::OkStatus();
}

} // namespace


int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    mediapipe::Status output_status = mediapipe::RunVideoReader();

    std::cout << output_status.message() << std::endl;

    return EXIT_SUCCESS;
}