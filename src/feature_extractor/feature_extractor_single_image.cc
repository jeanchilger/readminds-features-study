// Gets an image and outputs the face landmarks.
//
// Build:
//      bazel build --define MEDIAPIPE_DISABLE_GPU=1 --nocheck_visibility //src/feature_extractor:feature_extractor_single_image
//
// Run:
//      bazel-bin/src/feature_extractor/feature_extractor_single_image --input_image_path=path/to/image.jpg

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

#include "src/features/mouth/mouth.h"

DEFINE_string(input_image_path, "",
              "Path to the image.");

DEFINE_bool(split_landmarks, false,
            "Whether or not to split landmarks on more than one image.");

DEFINE_bool(show_image, false,
            "Whether ot not to show the (result) image on the screen.");

DEFINE_bool(save_image, true,
            "Whether or not to save the final image.");

//
mediapipe::Status RunGraph() {

    mediapipe::CalculatorGraphConfig config = 
            mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"(
                input_stream: "IMAGE:input_image"
                output_stream: "LANDMARKS:multi_face_landmarks" 
                node: {
                    calculator: "FaceLandmarkFrontCpu"
                    input_stream: "IMAGE:input_image"
                    output_stream: "LANDMARKS:multi_face_landmarks"
                }
            )");

    // creates the graph with those configs
    mediapipe::CalculatorGraph graph;

    MP_RETURN_IF_ERROR(graph.Initialize(config));

    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                    graph.AddOutputStreamPoller("multi_face_landmarks"));
    
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    // read input image 
    cv::Mat raw_image = cv::imread(FLAGS_input_image_path);

    // wrap cv::Mat into a ImageFrame
    auto input_frame = std::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB,
        raw_image.cols,
        raw_image.rows
    );
    
    
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());

    cv::cvtColor(raw_image, raw_image, cv::COLOR_BGR2RGB);

    int width = input_frame_mat.size().width;
    int height = input_frame_mat.size().height;

    raw_image.copyTo(input_frame_mat);

    // send input ImageFrame to graph
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        "input_image",
        mediapipe::Adopt(input_frame.release())
            .At(mediapipe::Timestamp(0))
    ));

    cv::cvtColor(raw_image, raw_image, cv::COLOR_RGB2BGR);

    // gets graph output
    mediapipe::Packet output_packet;
    
    if (!poller.Next(&output_packet)) return mediapipe::OkStatus();

    // get landmark points
    auto& output_landmark_vector = 
            output_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

    mediapipe::NormalizedLandmarkList face_landmarks = output_landmark_vector[0];
    
    // ===========================
    // F1
    // ===========================
    Mouth mouth_descriptor(face_landmarks, width, height);

    double f1 = mouth_descriptor.GetMouthOuter();
    std::cout << f1 << std::endl;

    
    MP_RETURN_IF_ERROR(graph.CloseInputStream("input_image"));

    return graph.WaitUntilDone();
}


int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    mediapipe::Status output_status = RunGraph();

    std::cout << output_status.message() << std::endl;

    return EXIT_SUCCESS;
}