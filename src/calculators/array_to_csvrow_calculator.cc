#include "array_to_csvrow_calculator.h"

namespace mediapipe {

typedef ArrayToCsvRowCalculator<float> ArrayFloatToCsvRowCalculator;
REGISTER_CALCULATOR(ArrayFloatToCsvRowCalculator);

} // namespace mediapipe