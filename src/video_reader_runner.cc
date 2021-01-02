#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"

const std::string VIDEO_PATH = "data/dummy/video_1.mp4";

/*
    Build: 
        bazel build --define MEDIAPIPE_DISABLE_GPU=1 //src:video_reader_runner --check_visibility=false
    RUN:
        bazel-bin/src/video_reader_runner

    Inputs:
        VIDEO_STREAM: empty if intents to record from webcam,
        video file path otherwise

    Ouput:
        ImageFrame
*/

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

    while (poller_landmarks.Next(&landmarks_packet)) {
        // Get landmarks from output
        auto& output_landmark_vector = 
                landmarks_packet.Get<std::vector<NormalizedLandmarkList> >();

        NormalizedLandmarkList face_landmarks = output_landmark_vector[0];

        // Get Image from output
        poller_image.Next(&image_packet);

        auto& output_frame = image_packet.Get<ImageFrame>();
        cv::Mat output_frame_mat = formats::MatView(&output_frame);

        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

        // Setup landmark on image
        int width = output_frame_mat.size().width;
        int height = output_frame_mat.size().height;
        int x, y;

        for (int i=0; i < 468; i++) {
            const NormalizedLandmark landmark = face_landmarks.landmark(i);

            x = (int) floor(landmark.x() * width);
            y = (int) floor(landmark.y() * height);

            cv::circle(
                output_frame_mat,
                cv::Point(x, y),
                2,
                cv::Scalar(255, 0, 0),
                -1,
                8
            );
        }

        // Show landmarks
        cv::imshow("Output Image", output_frame_mat);

        char c = (char) cv::waitKey(1);
        if (c == 27) {
            break;
        }

    }

    cv::destroyAllWindows();

    return ::mediapipe::OkStatus();
}

} // namespace

int main(int argc, char** argv) {
    ::mediapipe::Status status = ::mediapipe::RunVideoReader();

    std::cout << status.message() << '\n';

    return 0;
}