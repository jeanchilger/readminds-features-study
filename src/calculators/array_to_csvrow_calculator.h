#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

// Class documentation
//

namespace mediapipe {

// Writes the contents of a given container to a row of a csv file.
// Every element will be placed in a single column. At each timestamp,
// an input (array, vector...) from stream is written.
template <class C>
class ArrayToCsvRowCalculator : public CalculatorBase {

    public:
        static ::mediapipe::Status GetContract(CalculatorContract* cc) {
            RET_CHECK(cc->Inputs().NumEntries() != 0);
            
            // Container and its type will be validated later
            for (int i=0; i< cc->Inputs().NumEntries(); i++) {
                cc->Inputs().Index(i).SetAny();
            }

            return ::mediapipe::OkStatus();
        }

        ::mediapipe::Status Open(CalculatorContext* cc) override {
            header_ = cc->Options<::mediapipe::ArrayToCsvRowCalculatorOptions>()
                    .header();

            file_name_ = 
                    cc->Options<::mediapipe::ArrayToCsvRowCalculatorOptions>()
                            .file_name();

            return ::mediapipe::OkStatus();
        }

        ::mediapipe::Status Process(CalculatorContext* cc) override {

            return ::mediapipe::OkStatus();
        }

        ::mediapipe::Status Close(CalculatorContext* cc) override {

            return ::mediapipe::OkStatus();
        }

    private:
        const char** header_;
        const char* file_name_;

};

} // namespace mediapipe