// Gets an image and outputs the face landmarks.
//
// Build:
//      bazel build --define MEDIAPIPE_DISABLE_GPU --nocheck_visibility //src/face_landmark:face_landmark
//
// Run:
//      bazel-bin/src/face_landmark/face_landmark --input_image_path=path/to/image.jpg

#include <cstdlib>
#include <iostream>
#include <vector>

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

DEFINE_string(input_image_path, "",
              "Path to the image.");

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
    raw_image.copyTo(input_frame_mat);

    // send input ImageFrame to graph
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        "input_image",
        mediapipe::Adopt(input_frame.release())
            .At(mediapipe::Timestamp(0))
    ));

    // gets graph output
    mediapipe::Packet output_packet;

    
    if (!poller.Next(&output_packet)) return mediapipe::OkStatus();

    // get landmark points
    auto& output_landmark_vector = 
            output_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();

    mediapipe::NormalizedLandmarkList face_landmarks = output_landmark_vector[0];

    for (int i=0; i < face_landmarks.landmark_size(); i++) {
        const mediapipe::NormalizedLandmark landmark = face_landmarks.landmark(i);

        std::cout << "x: " << landmark.x()
                    << " y: " << landmark.y()
                    << " z: " << landmark.z()
                    << std::endl;
    }

    // cv::imshow("Window Name", raw_image);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
    
    MP_RETURN_IF_ERROR(graph.CloseInputStream("input_image"));
    return graph.WaitUntilDone();
}


int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::cout << FLAGS_input_image_path << std::endl;

    mediapipe::Status output_status = RunGraph();

    std::cout << output_status.message() << std::endl;


    return EXIT_SUCCESS;
}