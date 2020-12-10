#include "src/features/mouth/mouth.h"

#include "mediapipe/framework/port/opencv_imgproc_inc.h"


Mouth::Mouth(int img_width, int img_height) {
    img_width_ = img_width;
    img_height_ = img_height;
}

void Mouth::SetLandmarks(mediapipe::NormalizedLandmarkList landmarks) {
    landmarks_ = landmarks;
}