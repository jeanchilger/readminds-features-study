#include <iostream>
#include <string>

#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

const std::string file_path = "data/dummy/kodak01.png";

mediapipe::Status TestGraph() {
    CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "in"
        output_stream: "out"
        node {
            calculator: "GaussianBlurCalculator"
            input_stream: "IMAGE:in"
            output_stream: "IMAGE:out"
        }
    )");

    CalculatorGraph graph;

    MP_RETURN_IF_ERROR(graph.Initialize(config));

    // Adds a poller to the output stream
    ASSIGN_OR_RETURN(OutputStreamPoller poller,
                   graph.AddOutputStreamPoller("out"));

    MP_RETURN_IF_ERROR(graph.StartRun({}));

    // Reads the input image, and converts to RGB
    cv::Mat raw_input_image = cv::imread(file_path);
    cv::cvtColor(raw_input_image, raw_input_image, cv::COLOR_BGR2RGB);

    // Equivalent to:
    // auto input_image = ::std::make_unique<ImageFrame>(args)
    ::std::unique_ptr<ImageFrame> input_frame(new ImageFrame(
        mediapipe::ImageFormat::SRGB,
        raw_input_image.cols,
        raw_input_image.rows
    ));

    // Copy raw image to ImageFrame pointer
    cv::Mat input_frame_mat = formats::MatView(input_frame.get());
    raw_input_image.copyTo(input_frame_mat);

    // Adds the ImageFrame to the input stream.
    // Adopt() creates a Packet that assumes the ownership
    // of the given pointer.
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        "in",
        mediapipe::Adopt(input_frame.release()) 
            .At(mediapipe::Timestamp(0))
    ));

    MP_RETURN_IF_ERROR(graph.CloseInputStream("in"));

    // Gets the output Packet and displays its content
    mediapipe::Packet output_packet;
    poller.Next(&output_packet);

    if (!output_packet.IsEmpty()) {
        auto& output_frame = output_packet.Get<ImageFrame>();
        cv::Mat output_frame_mat = formats::MatView(&output_frame);

        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

        cv::imshow("adw", output_frame_mat);
        cv::waitKey(0);
        // cv::destroyAllWindows();
    }

    return graph.WaitUntilDone();
}

} // namespace mediapipe

int main(int argc, char** argv) {
    ::mediapipe::Status status = mediapipe::TestGraph();

    std::cout << status.message() << std::endl;

    return 0;
}