from pyVHR.signals.video import Video as pyVHRVideo


class VideoWrapper:

    """Class that wraps pyVHR Video and provides
    a basic documentation regarding parameters
    and methods.
    VideoWrapper can be used for video assessment
    (ROI methods and thresholds evaluation).
    WARNING: large video files make face detection
    slow.
    """

    def __init__(self, video_path, detector, extractor,
                 type_roi, skin_thresh_adapt=None,
                 skin_thresh_fix=None, rect_regions=None,
                 rect_coords=None, end_time=None):
        """Creates a VideoWrapper given a video path, face
        detector, video extractor, ROI method and ROI related
        parameters.

        Args:
            video_path (str): path to the video about to be processed.
            detector (str): face detector method
                (dlib, mtcnn, mtcnn_kalmann).
            extractor (str): library used for video reading
                (opencv, skvideo).
            type_roi (str): method used for ROI extraction
                (rect, skin_adapt, skin_fix).
            skin_thresh_adapt (float, optional): Threshold used
                when `type_roi` is set to skin_adapt. Defaults to None.
            skin_thresh_fix (float, optional): Threshold used
                when `type_roi` is set to skin_fix. Defaults to None.
            rect_regions (list, optional): List of retangular regions
                to be extracted from the face, e.g ['forehead', 'lcheek',
                'rcheek', 'nose']. Defaults to None.
            rect_coords (list, optional): List of retangular ROIs, e.g
                [[X0, Y0, W0, H0], [Xn, Yn, Wn, Hn]]. Defaults to None.
            end_time (int, optional): Final instant of the Video in seconds.
                Defaults to None.

        Returns:
            VideoWrapper: video wrapper object, able to be
                passed to a estimator method.
        """
        self._video = pyVHRVideo(video_path)
        self._detector = detector
        self._extractor = extractor
        self._end_time = end_time
        self._type_roi = type_roi
        self._skin_thresh_adapt = skin_thresh_adapt
        self._skin_thresh_fix = skin_thresh_fix
        self._rect_regions = rect_regions
        self._rect_coords = rect_coords

    @property
    def video(self):
        return self._video

    @property
    def detector(self):
        return self._detector

    @property
    def extractor(self):
        return self._extractor

    @property
    def type_roi(self):
        return self._type_roi

    @property
    def skin_thresh_adapt(self):
        return self._skin_thresh_adapt

    @property
    def end_time(self):
        return self._end_time

    def extract_faces(self, verbose=False):
        """Extracts faces from a video.

        Args:
            verbose (bool, optional): Shows video metadata.
                Defaults to False.
        """
        self.video.getCroppedFaces(
                detector=self._detector,
                extractor=self._extractor)
        if verbose:
            self._video.printVideoInfo()

    def show_faces(self):
        """Sets ROI method and needed parameters
           then display resulting video.
        """
        self._video.setMask(typeROI=self.type_roi,
                            skinThresh_adapt=self._skin_thresh_adapt,
                            skinThresh_fix=self._skin_thresh_fix,
                            rectRegions=self._rect_regions,
                            rectCoords=self._rect_coords)
        self._video.showVideo()
