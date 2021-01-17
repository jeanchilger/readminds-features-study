#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/port/logging.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"


/*
    Build: 
        bazel build --define MEDIAPIPE_DISABLE_GPU=1 //src:array_to_csvrow_runner --check_visibility=false
    RUN:
        bazel-bin/src/array_to_csvrow_runner
*/

namespace mediapipe {

mediapipe::Status RunVideoReader() {

    CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(
            R"(
        input_stream: "in"
        node {
            calculator: "ArrayFloatToCsvRowCalculator"
            input_stream: "in"
        }
    )");

    CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    MP_RETURN_IF_ERROR(graph.StartRun({}));

    return ::mediapipe::OkStatus();
}

} // namespace

int main(int argc, char** argv) {
    ::mediapipe::Status status = ::mediapipe::RunVideoReader();

    std::cout << status.message() << '\n';

    return 0;
}