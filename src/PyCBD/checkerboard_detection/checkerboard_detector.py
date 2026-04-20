from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt
from libCBDetect import Checkerboard


class CheckerboardDetector:
    """Checkerboard detector class."""

    def __init__(self) -> None:
        self.detector: Checkerboard = Checkerboard()
        self.detector.norm = True
        self.detector.score_thr = 0.01
        self.detector.strict_grow = False
        self.detector.show_grow_processing = False
        self.detector.overlay = True
        self.detector.show_debug_image = False

    def detect_checkerboard(self, image: npt.NDArray[np.float64]) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """Detect the checkerboard in the image.

        Sometimes failed detections are caused by the image being too big or too small (generally > 2000 pixels or < 400
        pixels), in this case rescaling the image might solve the problem.

        :param image: A numpy array representing the image. If it is a color image it should be in BGR format.
        :returns: Board_uv: coordinates (u, v) of detected corners assigned to a checkerboard; board_xy: local
           coordinates (x, y) of the detected corners assigned to checkerboard; corners_uv: coordinates (u, v) of all
           detected corners, including those that have not been assigned to a checkerboard.
        """
        self.detector.cols = 0
        self.detector.rows = 0
        image = self._prepare_image(image)
        image, scale_factor = self._rescale_image(image)
        result = self._detect_corners(image)
        if result:
            board_uv, corners_uv = self._extract_corners()
            if scale_factor != 1.0:
                board_uv = board_uv / scale_factor
                corners_uv = corners_uv / scale_factor
            board_xy = self._calculate_local_coordinates(board_uv)
            return board_uv, board_xy, corners_uv
        else:
            return np.array([]), np.array([]), np.array([])

    @staticmethod
    def _rescale_image(image: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], float]:
        """Rescale the image if any dimension falls outside the optimal detection range [400, 2000] pixels.

        If the largest dimension exceeds 2000px, the image is downsampled so the largest dimension equals 2000px.
        If the largest dimension is below 400px, the image is upsampled so the largest dimension equals 400px.
        Aspect ratio is preserved in both cases. A scale factor is returned so that detected corner coordinates
        can be mapped back to the original image resolution.

        :param image: A 2D normalised float64 grayscale image array.
        :returns: A tuple of (rescaled_image, scale_factor), where scale_factor is the ratio of the rescaled
            dimension to the original dimension (scale_factor > 1 means upsampled, < 1 means downsampled,
            == 1 means unchanged). Divide detected corner coordinates by scale_factor to recover original
            image coordinates.
        """
        height, width = image.shape
        max_dim = max(height, width)

        if max_dim > 2000:
            scale_factor = 2000.0 / max_dim
            interpolation = cv2.INTER_AREA  # Preferred for downsampling: reduces aliasing
        elif max_dim < 400:
            scale_factor = 400.0 / max_dim
            interpolation = cv2.INTER_CUBIC  # Preferred for upsampling: smoother result
        else:
            return image, 1.0

        new_width = round(width * scale_factor)
        new_height = round(height * scale_factor)
        rescaled_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        return rescaled_image, scale_factor

    def _detect_corners(self, image: npt.NDArray[np.float64]) -> bool:
        """Detect corners in image."""
        height, width = image.shape
        image = image.reshape(-1)
        self.detector.array_norm_to_image(image, height, width)
        self.detector.find_corners()
        self.detector.find_board_from_corners()
        if self.detector.rows == 0 or self.detector.cols == 0:
            return False
        else:
            return True

    def _extract_corners(self) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Extract detected corners from C++ object."""
        corners_u = np.zeros(self.detector.number_of_corners)
        corners_v = np.zeros(self.detector.number_of_corners)
        self.detector.get_corners(corners_u, corners_v)
        corners_uv = np.vstack((corners_u, corners_v)).T
        board_u = np.zeros(self.detector.rows * self.detector.cols)
        board_v = np.zeros(self.detector.rows * self.detector.cols)
        self.detector.get_board_corners(board_u, board_v)
        board_uv = np.vstack((board_u, board_v)).T
        return board_uv, corners_uv

    def _calculate_local_coordinates(self, board_uv: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Gives local corner coordinates in (x, y)."""
        board_xy = np.zeros((board_uv.shape[0], 2))  # initialise
        cnt = 0
        for i in range(self.detector.rows):
            for j in range(self.detector.cols):
                new_local_coordinate = (i, j)
                board_xy[cnt, :] = new_local_coordinate
                cnt += 1
        return board_xy

    @staticmethod
    def _prepare_image(image: npt.NDArray) -> npt.NDArray[np.float64]:
        """Check whether the image is compatible, convert the image to grayscale if necessary, and normalize the image.

        :param image: A numpy array representing the image. If it is a color image it should be in BGR format.
        :raises TypeError: If `image` is not a string or numpy array.
        :raises ValueError: If the image array has incompatible dimensions or an incompatible amount of channels.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("`image` should be a numpy array.")
        image = image.copy()  # Just to make sure the original image does not get modified

        # Convert to grayscale if necessary
        n_dimensions = len(image.shape)
        if n_dimensions == 3:
            n_channels = image.shape[2]
            if n_channels == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif not n_channels == 1:  # Usually the array won't have 3 dimensions if there is only 1 channel
                raise ValueError("The image should have 1 or 3 channels.")
        elif image.ndim != 2:
            raise ValueError("Image arrays should be 2D or 3D.")

        # Convert to float and normalize
        image = image.astype(np.float64)
        image = (image - np.amin(image)) / np.ptp(image)
        return image