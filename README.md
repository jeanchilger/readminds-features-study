# Readminds Feature Study
Readminds [1] is a software with the capability of remote emotion detection, applied on players in eletronic games. Features like heart rate and facial actions are currently used to estimate a user's emotional state. Whereas these features provide satisfactory results for the method, new ones may furnish a reasonable increase on the overall performance. Therefore, this study comes with two primary objectives:
1. Engineer new features aimed at the emotion detection context;
2. Study the viability of the newly built features in the data used in [1].

## Getting Started

All the important (made by us) scripts are within the `src` folder. The other folders are mostly related to MediaPipe configuration.

**Note:** Since all packages within `mediapipe/` are private, thus can't be used outside that folder. To overcome this condition, every Bazel command (e.g. `build`, `run`, ...) should be applied with the `--check_visibility=false` option. This is a **extremely bad** solution, and will be fixed in the future.

### Installing
To get the project running you'll need [Bazel](https://bazel.build/), [OpenCV](https://opencv.org/) and [FFmpeg](https://ffmpeg.org/). Installation instructions for all these may be found on the [MediaPipe installation guide](https://google.github.io/mediapipe/getting_started/install.html), please follow it.

### Running the Hello World
The hello world example, within `src` is the same as [MediaPipe's Hello World! on Desktop](https://google.github.io/mediapipe/getting_started/hello_world_desktop.html). To run it execute the follow commands:

```
# This is needed to see the messages on terminal.
# Otherwise they will be placed in a log file.
export GLOG_logtostderr=1
```

```
bazel run --define MEDIAPIPE_DISABLE_GPU=1 \
    //src/hello_world:hello --check_visibility=false
```

## References
[1] _Fernando Bevilacqua_. **Game-calibrated and user-tailored remote detection of emotions:  A non-intrusive,  multifactorial camera-based approach for detecting stress and boredom of players in games**. PhD thesis, University of Skövde.
