#include "array_to_csv_row_calculator.h"

#include <vector>

namespace mediapipe {

typedef ArrayToCsvRowCalculator<std::vector<float>> ArrayFloatToCsvRowCalculator;
REGISTER_CALCULATOR(ArrayFloatToCsvRowCalculator);

} // namespace mediapipe