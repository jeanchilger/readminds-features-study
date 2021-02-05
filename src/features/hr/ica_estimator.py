from base_estimator import BaseEstimator
from pyVHR.methods.ica import ICA


class ICAEstimator(BaseEstimator):
    """Defines a wrapper for heart rate estimator based in Idependent
    Component Analysis (ICA) of remote photoplethysmography (rPPG) data
    from pyVHR implementation.
    """
    def __init__(self, video_wrapper, method="jade", verbose=1):
        """
        Args:
            video_wrapper (VideoWrapper): pyVHR Video Wrapper.
            method (str, optional): ICA method, e.g fast_ica or jade.
                `fast_ica` don't work very well. Defaults to "jade".
            verbose (int, optional): Whether to display or not
                running info (1), graphs (2), etc. Defaults to 1.
        """
        super().__init__(video_wrapper)
        self._ica = ICA(
            ICAmethod=method,
            video=self._video_wrapper.video,
            verb=verbose)

    def run_offline(self):
        estimation = self._ica.runOffline(
            ROImask=self._roi_mask,
            skinAdapt=self._skin_adapt,
            skinFix=self._skin_fix,
            rectCoords=self._rect_coords,
            rectRegions=self._rect_regions)
        self._bpm_estimation = estimation
