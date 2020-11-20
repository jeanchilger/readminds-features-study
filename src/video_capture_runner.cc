#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"

const std::string video_path = "data/dummy/leaves_in_the_wind.mp4";/*"data/dummy/background-image.png";*/

namespace mediapipe {

  mediapipe::Status RunVideoCapture() {

    CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(R"(
        input_stream: "in"
        output_stream: "out"
        node {
            calculator: "VideoCaptureCalculator"
            input_side_packet: "FILE_PATH:in"
            output_stream: "OUTPUT_FRAME:out"
        }
    )");

    std::map<std::string, Packet> input_side_packets;
    input_side_packets["in"] = MakePacket<std::string>(video_path);

    CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config, input_side_packets));
    
    std::cout << "Got Here" << '\n';
    
    ASSIGN_OR_RETURN(::mediapipe::StatusOrPoller status_or_poller, graph.AddOutputStreamPoller("out"));

    // (status_or_poller.ok());
    OutputStreamPoller poller = std::move(status_or_poller.ValueOrDie());
    
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    //

    // mediapipe::Packet packet;
    // auto frame;

    // std::cout << "Miaau 1" << '\n';

    // poller.Next(&packet);

    //while () {
      // auto& output_frame = packet.Get<ImageFrame>();
      // cv::Mat output_frame_mat = formats::MatView(&output_frame);
      
      // cv::imshow("Miaau", output_frame_mat);
      //cv::waitKey(0);
      // std::cout << "Miaau" << '\n';
      //cv::imshow("Frame-samakun", frame);
    //}


    return graph.WaitUntilDone();
  }

}

int main(int argc, char** argv) {
  //google::InitGoogleLogging(argv[0]);
  ::mediapipe::Status status = ::mediapipe::RunVideoCapture();
  
  std::cout << status.message() << '\n';

  // google::LogMessage::Fail();

  return 0;
}