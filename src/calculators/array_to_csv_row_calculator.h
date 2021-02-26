// Copyright 2021 The authors

#ifndef SRC_CALCULATORS_ARRAY_TO_CSV_ROW_CALCULATOR_H_
#define SRC_CALCULATORS_ARRAY_TO_CSV_ROW_CALCULATOR_H_

#include <fstream>
#include <string>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

#include "src/calculators/array_to_csv_row_calculator.pb.h"

namespace mediapipe {

// Writes the contents of a given container to a row of a csv file.
// Every element will be placed in a single column. At each timestamp,
// an input (array, vector...) from stream is written.
//
// To use this calculator for a specific container, register it in .cc
// file as in the example:
// typedef ArrayToCsvRowCalculator<std::vector<float>>
//        ArrayFloatVectorToCsvRowCalculator;
// REGISTER_CALCULATOR(ArrayFloatVectorToCsvRowCalculator);
template <typename C>
class ArrayToCsvRowCalculator : public CalculatorBase {
    public:
        static ::mediapipe::Status GetContract(CalculatorContract* cc) {
            RET_CHECK(cc->Inputs().NumEntries() != 0);

            // TODO(@jeanchilger): Container validation?
            for (int i=0; i< cc->Inputs().NumEntries(); i++) {
                cc->Inputs().Index(i).SetAny();
            }

            return ::mediapipe::OkStatus();
        }

        ::mediapipe::Status Open(CalculatorContext* cc) override {
            ::mediapipe::ArrayToCsvRowCalculatorOptions options =
                    cc->Options<::mediapipe::ArrayToCsvRowCalculatorOptions>();
            file_path_ = options.file_path();

            // Before writting, the file is created if it don't exists.
            // If it already exists, its content is erased.
            ::std::fstream output_csv_file(
                    file_path_,
                    ::std::ios_base::out |
                    ::std::ios_base::trunc);

            // Writes header
            for (int i=0; i < options.header_size(); i++) {
                output_csv_file << options.header(i) << ",";
            }

            if (options.header_size() != 0) {
                output_csv_file << "\n";
            }

            output_csv_file.close();

            return ::mediapipe::OkStatus();
        }

        ::mediapipe::Status Process(CalculatorContext* cc) override {
            // Open file in append mode
            ::std::fstream output_csv_file(file_path_, ::std::ios_base::app);

            for (int i=0; i < cc->Inputs().NumEntries(); i++) {
                auto begin = ::std::begin(cc->Inputs().Index(i).Get<C>());
                auto end = ::std::end(cc->Inputs().Index(i).Get<C>());

                if (begin == end) {
                    break;
                }

                for (; begin != end; ++begin) {
                    output_csv_file << *begin << ",";
                }

                output_csv_file << "\n";
            }

            output_csv_file.close();

            return ::mediapipe::OkStatus();
        }

        ::mediapipe::Status Close(CalculatorContext* cc) override {
            return ::mediapipe::OkStatus();
        }

    private:
        std::vector<std::string> header_;
        std::string file_path_;
};

}  // namespace mediapipe

#endif  // SRC_CALCULATORS_ARRAY_TO_CSV_ROW_CALCULATOR_H_
