#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

#include "src/calculators/array_to_csv_row_calculator.pb.h"



// Build: 
//     bazel build --define MEDIAPIPE_DISABLE_GPU=1 //src:array_to_csv_row_runner --check_visibility=false
// Run:
//     bazel-bin/src/array_to_csv_row_runner


namespace mediapipe {

mediapipe::Status RunVideoReader() {

    CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(
            R"(
        input_stream: "in"
        node {
            calculator: "DoubleVectorToCsvRowCalculator"
            input_stream: "in"
            node_options: {
                [type.googleapis.com/mediapipe.ArrayToCsvRowCalculatorOptions] {
                    file_path: "example_file.csv"
                    header: ["as", "sa", "faa"]
                }
            }
        }
    )");

    CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    MP_RETURN_IF_ERROR(graph.StartRun({}));

    std::vector<double> input;
    for (int i=0; i < 10; i++) {
        for (int j=0; j < 3; j++) {
            input.push_back(i * j);
        }

        graph.AddPacketToInputStream(
                "in", MakePacket<std::vector<double>>(input).At(Timestamp(i)));

        input.clear();

        // std::cout << "\n@@@@@@@@@@@@@@@@@@@@@@@@@: " << i << std::endl << std::endl;
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream("in"));

    return graph.WaitUntilDone();
}

} // namespace

int main(int argc, char** argv) {
    ::mediapipe::Status status = ::mediapipe::RunVideoReader();

    std::cout << status.message() << '\n';

    return 0;
}