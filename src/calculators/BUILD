load("//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

cc_library(
    name = "video_reader_calculator",
    srcs = ["video_reader_calculator.cc"],
    hdrs = ["video_reader_calculator.h"],
    visibility = [
        "//visibility:public"
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_format_cc_proto",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/formats:video_stream_header",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:status_util",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "array_to_csvrow_calculator_proto",
    srcs = ["array_to_csvrow_calculator.proto"],
    deps = [
        "//mediapipe/framework:calculator_options_proto",
        "//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "array_to_csvrow_calculator",
    srcs = ["array_to_csvrow_calculator.cc"],
    hdrs = ["array_to_csvrow_calculator.h"],
    visibility = [
        "//visibility:public"
    ],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:ret_check",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/tool:status_util",
    ],
    alwayslink = 1,
)
