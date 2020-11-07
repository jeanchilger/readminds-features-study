package(default_visibility = ["//mediapipe:__subpackages__"])

cc_binary(
    name = "hello",
    srcs = ["hello_world.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/framework:calculator_graph",
        "//mediapipe/framework/port:logging",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
    ]
)