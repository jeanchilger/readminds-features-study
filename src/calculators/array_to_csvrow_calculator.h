#ifndef READMINDS_CALCULATORS_ARRAY_TO_CSV_ROW_CALCULATOR_
#define READMINDS_CALCULATORS_ARRAY_TO_CSV_ROW_CALCULATOR_

#include <string>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

#include "src/calculators/array_to_csvrow_calculator.pb.h"

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
            ::mediapipe::ArrayToCsvRowCalculatorOptions options = 
                    cc->Options<::mediapipe::ArrayToCsvRowCalculatorOptions>();
            
            for (int i=0; i < options.header_size(); i++) {
                header_.push_back(options.header(i));
            }

            file_name_ = options.file_name();

            return ::mediapipe::OkStatus();
        }

        ::mediapipe::Status Process(CalculatorContext* cc) override {
            std::cout << cc->Inputs().NumEntries() << std::endl;
            for (int i=0; i < cc->Inputs().NumEntries(); i++) {
                std::vector<float> input = cc->Inputs().Index(i).Get<C>();

                for (int j=0; j < input.size(); j++) {
                    std::cout << input.at(j) << ", ";
                }

                std::cout << std::endl;
            }

            return ::mediapipe::OkStatus();
        }

        ::mediapipe::Status Close(CalculatorContext* cc) override {

            return ::mediapipe::OkStatus();
        }

    private:
        std::vector<std::string> header_;
        std::string file_name_;

};

} // namespace mediapipe

#endif