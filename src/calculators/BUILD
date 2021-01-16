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
