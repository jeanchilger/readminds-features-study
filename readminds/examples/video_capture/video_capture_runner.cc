#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/formats/image_frame.h"

namespace readminds {

  ::mediapipe::Status RunVideoCapture() {

    std::string video_path = "video-1.mp4";

    ::mediapipe::CalculatorGraphConfig config = ::mediapipe::ParseTextProtoOrDie<::mediapipe::CalculatorGraphConfig>(R"(
      input_stream: "FILE_PATH"
      output_stream: "out"
      node {
        calculator: "VideoCaptureCalculator"
        input_stream: "FILE_PATH:input_file"
        output_stream: "out"
      }

    )");

    ::mediapipe::CalculatorGraph graph;
    graph.Initialize(config);
    
    std::cout << "Got Here" << '\n';
    
    ASSIGN_OR_RETURN(::mediapipe::OutputStreamPoller poller, graph.AddOutputStreamPoller("out"));
    
    std::cout << "Miaau 1" << '\n';

    graph.StartRun({});
    graph.AddPacketToInputStream("input_file", ::mediapipe::MakePacket<std::string>(video_path).At(::mediapipe::Timestamp(0)));

    graph.CloseInputStream("input_file");

    mediapipe::Packet packet;
    // auto frame;

    while (poller.Next(&packet)) {
      packet.Get<::mediapipe::ImageFrame>();
      std::cout << "miaau" << '\n';

      //cv::imshow("Frame-samakun", frame);
    }
    return graph.WaitUntilDone();
  }

}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  CHECK(::readminds::RunVideoCapture().ok());

  google::LogMessage::Fail();

  return 0;
}