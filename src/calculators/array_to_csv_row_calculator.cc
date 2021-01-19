#include "array_to_csv_row_calculator.h"

#include <string>
#include <vector>

namespace mediapipe {

typedef ArrayToCsvRowCalculator<::std::vector<int>> 
        IntVectorToCsvRowCalculator;
REGISTER_CALCULATOR(IntVectorToCsvRowCalculator);

// typedef ArrayToCsvRowCalculator<int*> 
//         ArrayIntArrayToCsvRowCalculator;
// REGISTER_CALCULATOR(ArrayIntArrayToCsvRowCalculator);

typedef ArrayToCsvRowCalculator<::std::vector<double>> 
        DoubleVectorToCsvRowCalculator;
REGISTER_CALCULATOR(DoubleVectorToCsvRowCalculator);

typedef ArrayToCsvRowCalculator<::std::vector<::std::string>> 
        StringVectorToCsvRowCalculator;
REGISTER_CALCULATOR(StringVectorToCsvRowCalculator);

} // namespace mediapipe