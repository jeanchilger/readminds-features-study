#include "array_to_csv_row_calculator.h"

#include <string>
#include <vector>

namespace mediapipe {

typedef ArrayToCsvRowCalculator<::std::vector<int>> 
        ArrayIntVectorToCsvRowCalculator;
REGISTER_CALCULATOR(ArrayIntVectorToCsvRowCalculator);

// typedef ArrayToCsvRowCalculator<int*> 
//         ArrayIntArrayToCsvRowCalculator;
// REGISTER_CALCULATOR(ArrayIntArrayToCsvRowCalculator);

typedef ArrayToCsvRowCalculator<::std::vector<float>> 
        ArrayFloatVectorToCsvRowCalculator;
REGISTER_CALCULATOR(ArrayFloatVectorToCsvRowCalculator);

typedef ArrayToCsvRowCalculator<::std::vector<::std::string>> 
        ArrayStringVectorToCsvRowCalculator;
REGISTER_CALCULATOR(ArrayStringVectorToCsvRowCalculator);

} // namespace mediapipe