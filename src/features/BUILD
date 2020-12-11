cc_library(
    name = "face_analyzer",
    srcs = ["face_analyzer.cc"],
    hdrs = ["face_analyzer.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_core",
    ],
    alwayslink = 1,
)