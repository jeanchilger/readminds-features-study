class BaseEstimator:
    """Entity that holds heart rate estimators' common properties
    and methods.
    """
    def __init__(self, video_wrapper):
        """Creates a BaseEstimator given a
        VideoWrapper instance.

        Args:
            video_wrapper (VideoWrapper): VideoWrapper instance.
        """
        self._bpm_estimation = None
        self._video_wrapper = video_wrapper
        self._roi_mask = self._video_wrapper.type_roi
        self._skin_adapt = self._video_wrapper.skin_thresh_adapt
        self._skin_fix = self._video_wrapper.skin_thresh_fix
        self._end_time = self._video_wrapper.end_time
        self._rect_coords = self._video_wrapper.rect_coords
        self._rect_regions = self._video_wrapper.rect_regions

    @property
    def bpm_estimation(self):
        return self._bpm_estimation

    def save(self, file_name):
        output = open(file_name, "w")
        output.write("second,hr\n")
        hr_estimation = self._bpm_estimation[0][0]
        seconds = self._bpm_estimation[1]

        for i, hr in enumerate(hr_estimation):
            instant = seconds[i]
            output.write(f"{instant},{hr}\n")
        output.close()
