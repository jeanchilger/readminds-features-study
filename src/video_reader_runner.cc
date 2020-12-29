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

// TODO: In order to show the landmarks using our
// `video_reader_calculator` we must use something like
// `FaceRendererCpu`

mediapipe::Status RunVideoReader() {

    CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "in"
        output_stream: "out"
        node {
            calculator: "VideoReaderCalculator"
            input_side_packet: "VIDEO_STREAM:in"
            output_stream: "IMAGE:out_frame"
        }
        node: {
            calculator: "FaceLandmarkFrontCpu"
            input_stream: "IMAGE:out_frame"
            output_stream: "LANDMARKS:out"
        }
    )");

    std::map<std::string, Packet> input_side_packets;
    input_side_packets["in"] = MakePacket<std::string>();

    CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config, input_side_packets));

    ASSIGN_OR_RETURN(
            ::mediapipe::StatusOrPoller status_or_poller, 
            graph.AddOutputStreamPoller("out"));

    OutputStreamPoller poller = std::move(status_or_poller.ValueOrDie());

    MP_RETURN_IF_ERROR(graph.StartRun({}));

    mediapipe::Packet packet;

    while (poller.Next(&packet)) {
        // Get frames from output
        // auto& output_frame = packet.Get<ImageFrame>();
        // cv::Mat output_frame_mat = formats::MatView(&output_frame);

        // cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

        // Get landmarks from output
        auto& output_landmark_vector = 
                packet.Get<std::vector<NormalizedLandmarkList> >();

        NormalizedLandmarkList face_landmarks = output_landmark_vector[0];

        // Setup landmark on image
        // int width = output_frame_mat.size().width;
        // int height = output_frame_mat.size().height;
        // int x, y;

        // for (int i=0; i < 468; i++) {
        //     const NormalizedLandmark landmark = face_landmarks.landmark(i);

        //     x = (int) floor(landmark.x() * width);
        //     y = (int) floor(landmark.y() * height);

        //     cv::circle(
        //         output_frame_mat,
        //         cv::Point(x, y),
        //         10,
        //         cv::Scalar(255, 0, 0),
        //         -1,
        //         8
        //     );

        //     cv::putText(
        //         output_frame_mat,
        //         std::to_string(i),
        //         cv::Point(x, y),
        //         cv::FONT_HERSHEY_DUPLEX,
        //         0.6,
        //         cv::Scalar(0, 0, 255));
        // }

        // // Show landmarks
        // cv::imshow("Output Image", output_frame_mat);
        // cv::waitKey(0);
        // cv::destroyAllWindows();

        // char c = (char) cv::waitKey(1);
        // if (c == 27) {
        //     break;
        // }

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