// Gets an image and outputs the face landmarks.
//
// Build:
//      bazel build --define MEDIAPIPE_DISABLE_GPU --nocheck_visibility //src/face_landmark:face_landmark
//
// Run:
//      bazel-bin/src/face_landmark/face_landmark --input_image_path=path/to/image.jpg

#include <cstdlib>
#include <iostream>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

DEFINE_string(input_image_path, "",
              "Path to the image.");

// std::string FACE_LANDMARK_FILE_PATH = "";


//
// ::mediapipe::Status RunGraph() {
    // std::string face_landmark_graph_contents;

    // MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
    //     FACE_LANDMARK_FILE_PATH, &calculator_graph_config_contents));
// }



int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::cout << FLAGS_input_image_path << std::endl;

    return EXIT_SUCCESS;
}