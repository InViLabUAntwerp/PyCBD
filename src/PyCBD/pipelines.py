"""This module contains the checkerboard detection pipelines."""

from .checkerboard_detection import CheckerboardDetector
from .checkerboard_enhancement import CheckerboardEnhancer
import numpy as np
import numpy.typing as npt
import cv2
from typing import Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import logging


class CBDPipeline:
    """Checkerboard detection pipeline that combines the checkerboard detector and checkerboard enhancer.

    :ivar checkerboard_detector: Detector that will be used to detect the checkerboard. The default detector
       is :py:class:`~PyCBD.checkerboard_detection.checkerboard_detector.GeigerDetector`, but other detectors can be
       used as long as they have a :py:meth:`~PyCBD.checkerboard_detection.detectors.CheckerboardDetector.detect_checkerboard`
       method that takes the same inputs and delivers the same outputs as our detector.
    :ivar checkerboard_enhancer: :py:class:`~PyCBD.checkerboard_enhancement.checkerboard_enhancer.CheckerboardEnhancer`
       for performing checkerboard enhancement. You can change the attributes of the enhancer to fit your needs.
    :ivar expand: Whether to try to expand the detected checkerboard or not by using checkerboard enhancement when some
       corners remain that have not been assigned to the board. This can also be used to bridge large occlusions.
       When using the expansion it is recommended that you provide a checkerboard size when performing the detection, so
       the enhancer does not needlessly try to expand when the entire checkerboard has already been detected.
    :ivar predict: Whether to refine the corner positions with Gaussian processes or not. This also fills in occlusions
       and can expand the checkerboard beyond the image borders.
    :ivar out_of_image: Whether to include predicted points beyond the limits of the image during the refinement
       step.
    :ivar plot_pipeline_steps: Whether to plot intermediate results of the pipeline.
    :ivar max_iterations: Sets the max_iterations attribute of the checkerboard_enhancer.
    :ivar max_expansion_factor: Sets the max_expansion_factor attribute of the checkerboard_enhancer.
    :ivar max_dist_factor: Sets the max_dist_factor attribute of the checkerboard_enhancer.
    :ivar must_plot_gp_stuff: Sets the must_plot_gp_stuff attribute of the checkerboard_enhancer.
    :ivar must_plot_iterations: Sets the must_plot_iterations attribute of the checkerboard_enhancer.
    :ivar dewarped_res_factor: Sets the dewarped_res_factor attribute of the checkerboard_enhancer.
    """

    def __init__(self, detector: any = CheckerboardDetector(), expand: bool = False, predict: bool = False,
                 out_of_image: bool = False) -> None:
        """Class constructor.

        :param detector: An instance of the detector that should be used for checkerboard detection. Our detector is
           used by default but other detectors can be used as long as they have a `detect_checkerboard` method that
           takes the same inputs and returns the same outputs as our detector.
        :param expand: Whether to try to expand detected checkerboard or not.
        :param predict: Whether to use our refinement step and fill in occlusions or not.
        :param out_of_image: Whether to include predicted points beyond the limits of the image during the refinement
           step.
        """
        self._logger = logging.getLogger(__name__)
        self.checkerboard_detector: Any = detector
        self.checkerboard_enhancer: CheckerboardEnhancer = CheckerboardEnhancer()
        self.plot_pipeline_steps = False
        self.expand: bool = expand
        self.predict: bool = predict
        self.out_of_image: bool = out_of_image

    @property
    def max_iterations(self) -> int:
        return self.checkerboard_enhancer.max_iterations

    @max_iterations.setter
    def max_iterations(self, new_max_iterations: int):
        self.checkerboard_enhancer.max_iterations = new_max_iterations

    @property
    def max_expansion_factor(self) -> int:
        return self.checkerboard_enhancer.max_expansion_factor

    @max_expansion_factor.setter
    def max_expansion_factor(self, new_max_expansion_factor: int):
        self.checkerboard_enhancer.max_expansion_factor = new_max_expansion_factor

    @property
    def max_dist_factor(self) -> float:
        return self.checkerboard_enhancer.max_dist_factor

    @max_dist_factor.setter
    def max_dist_factor(self, new_max_dist_factor: float):
        self.checkerboard_enhancer.max_dist_factor = new_max_dist_factor

    @property
    def must_plot_gp_stuff(self) -> bool:
        return self.checkerboard_enhancer.must_plot_GP_stuff

    @must_plot_gp_stuff.setter
    def must_plot_gp_stuff(self, state: bool):
        self.checkerboard_enhancer.must_plot_GP_stuff = state

    @property
    def must_plot_iterations(self) -> bool:
        return self.checkerboard_enhancer.must_plot_iterations

    @must_plot_iterations.setter
    def must_plot_iterations(self, state: bool):
        self.checkerboard_enhancer.must_plot_iterations = state

    @property
    def dewarped_res_factor(self) -> float:
        return self.checkerboard_enhancer.dewarped_res_factor

    @dewarped_res_factor.setter
    def dewarped_res_factor(self, new_dewarped_res_factor: float):
        self.checkerboard_enhancer.dewarped_res_factor = new_dewarped_res_factor

    def detect_checkerboard(self, image: npt.NDArray,
                            size: Optional[Tuple[int, int]] = None) -> Tuple[int, npt.NDArray[np.float64],
                                                                             npt.NDArray[np.float64]]:
        """Perform checkerboard detection on an image.

        :param image: Image to perform detection on. This should be a numpy array that represents a grayscale or color
           image in BGR format.
        :param size: The size of the checkerboard, i.e., the amount of inner corner rows and columns (rows, columns).
           This is an optional argument that allows the detector to verify whether the entire checkerboard has been
           found or not and whether the checkerboard xy positions are absolute or relative. It is recommended to provide
           the checkerboard size when performing the checkerboard expansion so the algorithm doesn't needlessly try to
           expand when all rows and columns have already been found.
        :returns: result: to what degree the checkerboard detection was successful, 0 means the detection failed, 1
           means the xy coordinates are relative, 2 means the xy coordinates are absolute; board_uv: image coordinates
           (u, v) of detected corners assigned to the checkerboard; board_xy: local coordinates (x, y) of the detected
           corners assigned to the checkerboard.
        """
        board_uv, board_xy, corners_uv = self.checkerboard_detector.detect_checkerboard(image)
        result, struct_found, all_found = self._verify_board(board_xy, size)
        if self.plot_pipeline_steps:
            fig, ax = plt.subplots()
            if len(image.shape) == 3:
                ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(image, cmap='gray')
            if corners_uv.size != 0:
                ax.plot(corners_uv[:, 0], corners_uv[:, 1], 'bo', markeredgecolor='k')
            if board_uv.size != 0:
                ax.plot(board_uv[:, 0], board_uv[:, 1], 'r-o', markeredgecolor='k')
                trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=-0.4, y=-0.20, units='inches')
                ax.text(board_uv[0, 0], board_uv[0, 1],
                        '(' + str(int(board_xy[0, 0])) + ', ' + str(int(board_xy[0, 1])) + ')',
                        color="red", transform=trans_offset)
                trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=0.05, y=0.05, units='inches')
                ax.text(board_uv[-1, 0], board_uv[-1, 1],
                        '(' + str(int(board_xy[-1, 0])) + ', ' + str(int(board_xy[-1, 1])) + ')',
                        color="red", transform=trans_offset)
            plt.title("Pipeline: After detection")
            plt.axis('off')
            plt.show()

        if self.expand and not struct_found and result > 0:
            board_mask = (corners_uv == board_uv[:, None]).all(-1).any(0)
            corners_uv = corners_uv[~board_mask, :]
            if corners_uv.size != 0:
                board_uv, board_xy = self.checkerboard_enhancer.fit_and_expand_board(image, board_uv, board_xy,
                                                                                     corners_uv, size)
                result, _, _ = self._verify_board(board_xy, size)
                if self.plot_pipeline_steps:
                    fig, ax = plt.subplots()
                    if len(image.shape) == 3:
                        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    else:
                        ax.imshow(image, cmap='gray')
                    ax.plot(corners_uv[:, 0], corners_uv[:, 1], 'bo', markeredgecolor='k')
                    ax.plot(board_uv[:, 0], board_uv[:, 1], 'r-o', markeredgecolor='k')
                    trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=-0.4, y=-0.20, units='inches')
                    ax.text(board_uv[0, 0], board_uv[0, 1],
                            '(' + str(int(board_xy[0, 0])) + ', ' + str(int(board_xy[0, 1])) + ')',
                            color="red", transform=trans_offset)
                    trans_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=0.05, y=0.05, units='inches')
                    ax.text(board_uv[-1, 0], board_uv[-1, 1],
                            '(' + str(int(board_xy[-1, 0])) + ', ' + str(int(board_xy[-1, 1])) + ')',
                            color="red", transform=trans_offset)
                    plt.title("Pipeline: After expansion")
                    plt.axis('off')
                    plt.show()
            else:
                self._logger.info("No unassigned corners left, skipping expansion.")
        elif self.expand and struct_found:
            self._logger.info("Entire checkerboard structure found, skipping expansion.")
        elif result == 0:
            self._logger.info("Checkerboard detection failed, skipping all other steps.")

        if self.predict and result > 0:
            board_uv, board_xy = self.checkerboard_enhancer.fit_and_predict_board(image, board_uv, board_xy,
                                                                                  self.out_of_image)

        return result, board_uv, board_xy

    def dewarp_image(self, image: npt.NDArray, board_uv: npt.NDArray, board_xy: npt.NDArray,
                     use_stored: bool = True) -> npt.NDArray:
        """Remove lens and perspective distortion from the image.

        This method can either be performed separately, in which case new Gaussian processes need to be fitted to the
        checkerboard, or after performing checkerboard detection with :py:data:`predict` set to True, where the stored
        Gaussian processes can be used instead.

        :param image: Original image that needs to be dewarped.
        :param board_uv: Image corner coordinates (u, v).
        :param board_xy: Local corner coordinates (x, y).
        :param use_stored: Whether to reuse the stored Gaussian processes or fit new ones.
        :returns: The dewarped image.
        """
        return self.checkerboard_enhancer.dewarp_image(image, board_uv, board_xy, use_stored)

    @staticmethod
    def _verify_board(board_xy, size):
        """Verify to what degree the checkerboard has been detected."""
        result = 0
        all_found = False
        struct_found = False
        if board_xy.size != 0:
            result = 1  # found a board, coordinates are relative
            if size is not None:
                x_max, y_max = board_xy.max(axis=0)
                if size[0] != size[1]:
                    if x_max + 1 == size[1] and y_max + 1 == size[0]:
                        struct_found = True
                        result = 2  # found all rows and columns, coordinates are absolute
                else:
                    if x_max + 1 == size[1] and y_max + 1 == size[0]:
                        struct_found = True
                if board_xy.shape[0] == size[0] * size[1]:
                    all_found = True
        return result, struct_found, all_found
