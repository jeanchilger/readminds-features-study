#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"

namespace readminds {

// Simple Calculator implementation, providing an descritive example of
// MediaPipe's Calculator framework. Inputs and Outputs must match in number.
// Inputs a sequence of numbers and produces as output the same sequence,
// added by one.
class DummyCalculator : public CalculatorBase {
    public:
        // GetContract() is used to define expected types for inputs and
        // outputs to the calculator.
        static mediapipe::Status GetContract(CalculatorContract* cc) {
            // Defines that all Inputs and Outputs must be of
            // int type. Here Inputs and Outputs are accessed by
            // index.
            for (CollectionItemId id = cc->Inputs().BeginId();
                    id < cc->Inputs().EndId(); id++) {
                cc->Inputs().Get(id)->Set(<int>);
                cc->Outputs().Get(id)->Set(<int>);
            }

            return mediapipe::OkStatus();
        }

        // The Open() function reads and interprets the node's
        // configuration operations and sets the calculator to
        // a run state.
        static mediapipe::Status Open(CalculatorContext* cc) final {
            return mediapipe::OkStatus();
        }

        // If a calculator has inputs, the frameworks calls Process()
        // over and over, whenever there is at leats one input stream
        // with available a packet.
        static mediapipe::Status Process(CalculatorContext* cc) final {
            return mediapipe::OkStatus();
        }

        // When all calls to Process() ends, or when all input streams
        // are closed, the method Close() is called. If Open() was
        // successfully executed, Close() will be called, even if the run
        // was terminated by an error.
        static mediapipe::Status Close(CalculatorContext* cc) final {
            return mediapipe::OkStatus();
        }
};

} // readminds namespace
