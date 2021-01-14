cc_library(
    name = "mouth_analyzer",
    srcs = ["mouth_analyzer.cc"],
    hdrs = ["mouth_analyzer.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//src/features:generic_analyzer",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_core",
    ],
    alwayslink = 1,
)