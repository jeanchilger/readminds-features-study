cc_binary(
    name = "video_reader_runner",
    srcs = ["video_reader_runner.cc"],
    deps = [
        "//src/calculators:video_reader_calculator",
        "//mediapipe/modules/face_landmark:face_landmark_front_cpu",
        "//mediapipe/framework:calculator_graph",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/formats:image_frame_opencv",
    ]
)

cc_binary(
    name = "array_to_csv_row_runner",
    srcs = ["array_to_csv_row_runner.cc"],
    deps = [
        "//src/calculators:array_to_csv_row_calculator",
        "//mediapipe/framework:calculator_graph",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:commandlineflags",
    ]
)