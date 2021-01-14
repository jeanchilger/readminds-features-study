cc_library(
    name = "eye_analyzer",
    srcs = ["eye_analyzer.cc"],
    hdrs = ["eye_analyzer.h"],
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