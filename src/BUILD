cc_library(
    name = "dummy_calculator",
    srcs = ["dummy_calculator.cc"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
    ]
)

cc_binary(
    name = "run_gaussian",
    srcs = ["run_gaussian_blur_calculator.cc"],
    deps = [
        "//src/calculators:gaussian_blur_calculator",
        "//mediapipe/framework:calculator_graph",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_imgcodecs",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
    ] 
)