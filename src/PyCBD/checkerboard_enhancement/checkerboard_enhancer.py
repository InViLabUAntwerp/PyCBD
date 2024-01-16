"""This module contains the checkerboard enhancers."""

import GPy
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from typing import Tuple, Any, Optional
import logging


class CheckerboardEnhancer:
    """Class for checkerboard enhancement with Gaussian processes

    :var max_iterations: Determines the maximum allowed number of times you want to expand the local grid of
       points (in xy-space on the checkerboard itself) and try to find matching detected corners (in the image in
       uv-space).
    :var max_expansion_factor: The maximum amount of times the local grid of points (x, y) can expand in either
       direction without finding matching detected corners (u, v).
    :var max_dist_factor: The GP models predict uv-coordinates for a new corner. The algorithm will try to match
       those predictions to actual detected corners. This distance determines how far a detected corner can deviate from
       a predicted one.
    :var must_plot_GP_stuff: Set this to true if you want to visualize the GP predictions. You only need this in case
       something is obviously wrong.
    :var must_plot_iterations: Set this to true if you want to visualize each step of the algorithm. You only need this
       in case something is obviously wrong.
    :var max_nr_of_iters: Training the GPs means finding an optimal set of hyperparameters. This could in some case take
       a lot of iterations. Lower this number if you want to limit this. In general, you should not have to change this.
    :var optimizer: The optimizer used to train the GPs. In general, you should not have to change this.
    :var num_restarts: The optimizer training the GPS could get stuck in a local optimum, although for well-behaved
       checkerboards, this is rare. In general, you should not have to change this. If the results (log marginal
       likelihood values) do not differ much, then you could get away with setting this to zero (no restarts).
    :var lengthscale_bounded: In case the GPs end up with weird predictions, especially when working with only a few
       detected corners, you could try to work with bounds (limits) for the lengthscale.
    :var min_lengthscale: If lengthscale_bounded = True, this values puts a lower limit on the allowed lengthscale.
       Increase this number not to end up with unreasonably low lengthscales.
    :var max_lengthscale: If lengthscale_bounded = True, this values puts an upper limit on the allowed lengthscale.
    :var likelihood_variance_bounded: In case the GPs end up with weird predictions, especially when working with only a
       few detected corners, you could try to work with bounds (limits) for the noise allowed on the training points.
       In rare cases, the data could also be explained with low lengthscales and high noise levels. This is equivalent
       of explaining the data away with noise.
    :var min_likelihood_variance: If likelihood_variance_bounded = True, this values puts a lower limit on the allowed
       noise. Increase this number, so you don't end up with unreasonably low noise values.
    :var max_likelihood_variance: If likelihood_variance_bounded = True, this values puts an upper limit on the allowed
       noise. Increase this number, so you don't end up with unreasonably high noise values.
    :var m_xy_to_u: Stored model for predicting u values.
    :var m_xy_to_v: Stored model for predicting v values.
    :var scaler: Stored scaler
    :var dewarped_res_factor: How many pixels per square width in the dewarped image.
    """

    def __init__(self) -> None:
        """Class constructor."""
        self._logger = logging.getLogger(__name__)
        self.max_iterations: int = 12
        self.max_expansion_factor: int = 2
        self.max_dist_factor: float = 0.25
        self.must_plot_GP_stuff: bool = False
        self.must_plot_iterations: bool = False
        self.max_nr_of_iters: int = 25000
        self.optimizer: str = 'lbfgs'
        self.num_restarts: int = 3
        self.lengthscale_bounded: bool = True
        self.min_lengthscale: int = 0
        self.max_lengthscale: int = 25
        self.likelihood_variance_bounded: bool = False
        self.min_likelihood_variance = 0
        self.max_likelihood_variance = 1
        self.m_xy_to_u: Any = None
        self.m_xy_to_v: Any = None
        self.scaler: Any = None
        self.dewarped_res_factor = 50

    def fit_and_expand_board(self, image: npt.NDArray, board_uv: npt.NDArray, board_xy: npt.NDArray,
                             corners_uv: npt.NDArray, board_shape: Optional[Tuple[int, int]] = None) -> Tuple[npt.NDArray, npt.NDArray]:
        """Train a model based on corner inputs and use this model to predict all corners in the checkerboard.

        Two Gaussian processes are used for this. One to map the local xy-coordinates to the u-coordinates of the
        corners in the image. Another one to map the local xy-coordinates to the v-coordinates of the corners in the
        image. We iteratively expand the grid in local xy-space to try and find corners in the image that are not
        allocated to a point in the checkerboard grid yet.

        :param image: Image containing the checkerboard (used for plotting intermediate results).
        :param board_uv: Image corner coordinates (u, v)
        :param board_xy: Local corner coordinates (x, y)
        :param corners_uv: Coordinates (u, v) of all detected corners, including those that have not been assigned to a
           checkerboard.
        :param board_shape: The shape (rows, cols) of the checkerboard inner corners. This can be used to prevent the
           method from trying to expand the checkerboard further in the x/y direction when all columns/rows have already
           been found.
        :returns: board_uv, the corner image coordinates, and board_xy, their local coordinates.
        """
        current_iteration = 1
        expansion_factor = 1  # expand the current board by this many squares in north, east, south and west direction
        if board_shape is not None:
            n_rows = np.unique(board_xy[:, 1]).size
            n_cols = np.unique(board_xy[:, 0]).size
            expand_vertical = n_rows < board_shape[0]
            expand_horizontal = n_cols < board_shape[1]
        else:
            expand_horizontal = True
            expand_vertical = True

        while current_iteration <= self.max_iterations:
            self._logger.info("Starting iteration: " + str(current_iteration))
            self._logger.info("Nr of training points: " + str(board_xy.shape[0]))

            m_xy_to_u, m_xy_to_v, scaler = self._train_checkerboard(board_uv, board_xy)

            # Expand board_xy by expansion_factor
            new_board_xy = self._expand_board_xy(board_xy, expansion_factor, expand_horizontal, expand_vertical)

            # Use map to find more corners
            new_board_xy_scaled = scaler.transform(new_board_xy)
            mean_u_new, cov_u_new = m_xy_to_u.predict_noiseless(new_board_xy_scaled, full_cov=False)
            mean_v_new, cov_v_new = m_xy_to_v.predict_noiseless(new_board_xy_scaled, full_cov=False)

            # Remove all predicted points that are not inside the image
            limits_mask = np.squeeze((mean_u_new < 0) | (mean_u_new > image.shape[1]) | (mean_v_new < 0) | (mean_v_new > image.shape[0]))
            if limits_mask.any():
                mean_u_new = mean_u_new[~limits_mask]
                cov_u_new = cov_u_new[~limits_mask]
                mean_v_new = mean_v_new[~limits_mask]
                cov_v_new = cov_v_new[~limits_mask]
                new_board_xy = new_board_xy[~limits_mask]
                new_board_xy_scaled = new_board_xy_scaled[~limits_mask]

            if new_board_xy.shape[0] == 0:
                self._logger.info("All new points are outside of the image, stop iterating")
                break

            new_board_uv = np.concatenate((mean_u_new, mean_v_new), axis=1)

            if self.must_plot_GP_stuff:
                expand_grid_factor = 2  # this determines the grid size for the plots, purely cosmetics
                min_local_x = np.min(board_xy[:, 0]) - expand_grid_factor
                max_local_x = np.max(board_xy[:, 0]) + expand_grid_factor
                min_local_y = np.min(board_xy[:, 1]) - expand_grid_factor
                max_local_y = np.max(board_xy[:, 1]) + expand_grid_factor
                [Xi, Yj] = np.meshgrid(np.linspace(min_local_x, max_local_x, 50),
                                       np.linspace(min_local_y, max_local_y, 50))
                xy_test = np.vstack((Xi.ravel(), Yj.ravel())).T
                xy_test_scaled = scaler.transform(xy_test)
                mean_u, cov_u = m_xy_to_u.predict_noiseless(xy_test_scaled, full_cov=False)
                mean_v, cov_v = m_xy_to_v.predict_noiseless(xy_test_scaled, full_cov=False)

                nr_of_levels = 20
                levels_u = np.linspace(np.min(mean_u[:, 0]), np.max(mean_u[:, 0]), num=nr_of_levels)
                levels_v = np.linspace(np.min(mean_v[:, 0]), np.max(mean_v[:, 0]), num=nr_of_levels)

                plt.figure(figsize=(14, 6))
                plt.subplot(121)
                plt.contour(Xi, Yj, mean_u.reshape(Xi.shape), levels_u)
                plt.plot(board_xy[:, 0], board_xy[:, 1], 'ro'), plt.axis("square")
                plt.plot(new_board_xy[:, 0], new_board_xy[:, 1], 'go'), plt.axis("square")
                plt.xlabel("local x"), plt.ylabel("local y")
                plt.title("Mean of GP fit for U"), plt.colorbar()
                plt.subplot(122)
                plt.pcolor(Xi, Yj, cov_u.reshape(Xi.shape))
                plt.plot(board_xy[:, 0], board_xy[:, 1], 'ro'), plt.axis("square")
                plt.plot(new_board_xy[:, 0], new_board_xy[:, 1], 'go'), plt.axis("square")
                plt.xlabel("local x"), plt.ylabel("local y")
                plt.title("Variance of GP fit for U"), plt.colorbar()
                plt.show()

                plt.figure(figsize=(14, 6))
                plt.subplot(121)
                plt.contour(Xi, Yj, mean_v.reshape(Xi.shape), levels_v)
                plt.plot(board_xy[:, 0], board_xy[:, 1], 'ro'), plt.axis("square")
                plt.plot(new_board_xy[:, 0], new_board_xy[:, 1], 'go'), plt.axis("square")
                plt.xlabel("local x"), plt.ylabel("local y")
                plt.title("Mean of GP fit for V"), plt.colorbar()
                plt.subplot(122)
                plt.pcolor(Xi, Yj, cov_v.reshape(Xi.shape))
                plt.plot(board_xy[:, 0], board_xy[:, 1], 'ro'), plt.axis("square")
                plt.plot(new_board_xy[:, 0], new_board_xy[:, 1], 'go'), plt.axis("square")
                plt.xlabel("local x"), plt.ylabel("local y")
                plt.title("Variance of GP fit for V"), plt.colorbar()
                plt.show()

            # Find a match between predicted and detected corners
            max_dist = self.max_dist_factor * np.linalg.norm(board_uv[0, :] - board_uv[1, :])
            distances = cdist(new_board_uv, corners_uv, 'euclidean')
            minimum_indices = np.argmin(distances, axis=0)
            selection_mask = distances[minimum_indices, range(minimum_indices.size)] <= max_dist
            nr_of_points_found = np.sum(selection_mask)
            board_uv = np.concatenate((board_uv, corners_uv[selection_mask, :]), axis=0)
            board_xy = np.concatenate((board_xy, new_board_xy[minimum_indices[selection_mask], :]), axis=0)
            corners_uv = corners_uv[~selection_mask, :]

            if self.must_plot_iterations:
                # Plot final result of this iteration
                if len(image.shape) == 3:
                    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    plt.imshow(image, cmap='gray')
                plt.plot(corners_uv[:, 0], corners_uv[:, 1], 'bo', markeredgecolor='k')
                plt.plot(mean_u_new, mean_v_new, 'go', markeredgecolor='k')
                plt.plot(board_uv[:, 0], board_uv[:, 1], 'ro', markeredgecolor='k')
                title = "Enhancer: After iteration " + str(current_iteration)
                plt.title(title)
                plt.axis('off')
                plt.show()

            # Stop if all points are accounted for
            if corners_uv.size == 0:
                break

            if nr_of_points_found == 0:
                if expansion_factor >= self.max_expansion_factor:
                    self._logger.info("No new points added and exceeded max_expansion_factor")
                    break
                self._logger.info("No new points added, increasing expansion factor")
                expansion_factor = expansion_factor + 1
            else:
                expansion_factor = 1

            # Stop if the board shape has been found
            if board_shape is not None:
                n_rows = np.unique(board_xy[:, 1]).size
                n_cols = np.unique(board_xy[:, 0]).size
                expand_vertical = n_rows + expansion_factor <= board_shape[0]
                expand_horizontal = n_cols + expansion_factor <= board_shape[1]
                if not expand_vertical and not expand_horizontal:
                    self._logger.info("Found board with specified dimensions.")
                    break

            current_iteration += 1

        board_uv, board_xy = self._reset_origin_and_order(board_uv, board_xy)

        return board_uv, board_xy

    def fit_and_predict_board(self, image: npt.NDArray, board_uv: npt.NDArray, board_xy: npt.NDArray,
                              out_of_image: bool = False) -> Tuple[npt.NDArray, npt.NDArray]:
        """Train a model based on corner inputs and use this model to predict all corners in the checkerboard.

        This function is used after the final iteration in function fit_and_expand_board. It is assumed that all
        detected corner are allocated to point in the checkerboard grid. This means no loose detected corners and also
        false positives (detected corners that are not part of the checkerboard grid) are removed. Now we train the GPs
        one final time on all corners. We fill in the positions in xy-space where no corner was detected, e.g.
        occlusions. Once this model is trained, we can use it to predict points (pixel uv-coordinates) in the image for
        all corners. This fills up any occlusions and smooths out the board, removing jitter from the detected corners.

        :param image: The image containing the checkerboard (used for determining image limits).
        :param board_uv: Image corner coordinates (u, v).
        :param board_xy: Local corner coordinates (x, y).
        :param out_of_image: Whether to include predicted points beyond the limits of the image.
        :returns: board_uv_predicted_grid, the predicted corner image coordinates, and board_xy_predicted_grid, their
           local coordinates.
        """
        self.m_xy_to_u, self.m_xy_to_v, self.scaler = self._train_checkerboard(board_uv, board_xy)

        min_local_x = np.min(board_xy[:, 0])
        max_local_x = np.max(board_xy[:, 0])
        min_local_y = np.min(board_xy[:, 1])
        max_local_y = np.max(board_xy[:, 1])
        cols = int(max_local_x - min_local_x + 1)
        rows = int(max_local_y - min_local_y + 1)
        [Xi, Yj] = np.meshgrid(np.linspace(min_local_x, max_local_x, cols), np.linspace(min_local_y, max_local_y, rows))
        board_xy_predicted_grid = np.vstack((Xi.ravel(), Yj.ravel())).T
        board_xy_predicted_grid_scaled = self.scaler.transform(board_xy_predicted_grid)

        # Use GP to predict for entire grid
        mean_u, cov_u = self.m_xy_to_u.predict_noiseless(board_xy_predicted_grid_scaled, full_cov=False)
        mean_v, cov_v = self.m_xy_to_v.predict_noiseless(board_xy_predicted_grid_scaled, full_cov=False)
        board_uv_predicted_grid = np.concatenate((mean_u, mean_v), axis=1)
        board_uv_predicted_grid_uncertainty = np.concatenate((cov_u, cov_v), axis=1)  # use this as a quality label

        if self.must_plot_GP_stuff:
            expand_grid_factor = 2  # this determines the grid size for the plots, purely cosmetics
            min_local_x = np.min(board_xy_predicted_grid[:, 0]) - expand_grid_factor
            max_local_x = np.max(board_xy_predicted_grid[:, 0]) + expand_grid_factor
            min_local_y = np.min(board_xy_predicted_grid[:, 1]) - expand_grid_factor
            max_local_y = np.max(board_xy_predicted_grid[:, 1]) + expand_grid_factor
            [Xi, Yj] = np.meshgrid(np.linspace(min_local_x, max_local_x, 50), np.linspace(min_local_y, max_local_y, 50))
            xy_test = np.vstack((Xi.ravel(), Yj.ravel())).T
            xy_test_scaled = self.scaler.transform(xy_test)
            mean_u, cov_u = self.m_xy_to_u.predict_noiseless(xy_test_scaled, full_cov=False)
            mean_v, cov_v = self.m_xy_to_v.predict_noiseless(xy_test_scaled, full_cov=False)

            nr_of_levels = 20
            levels_u = np.linspace(np.min(mean_u[:, 0]), np.max(mean_u[:, 0]), num=nr_of_levels)
            levels_v = np.linspace(np.min(mean_v[:, 0]), np.max(mean_v[:, 0]), num=nr_of_levels)

            plt.figure(figsize=(14, 6))
            plt.subplot(121)
            plt.contour(Xi, Yj, mean_u.reshape(Xi.shape), levels_u)
            plt.plot(board_xy_predicted_grid[:, 0], board_xy_predicted_grid[:, 1], 'ro'), plt.axis("square")
            plt.xlabel("local x"), plt.ylabel("local y")
            plt.title("Mean of GP fit for U"), plt.colorbar()
            plt.subplot(122)
            plt.pcolor(Xi, Yj, cov_u.reshape(Xi.shape))
            plt.plot(board_xy_predicted_grid[:, 0], board_xy_predicted_grid[:, 1], 'ro'), plt.axis("square")
            plt.xlabel("local x"), plt.ylabel("local y")
            plt.title("Variance of GP fit for U"), plt.colorbar()
            plt.show()

            plt.figure(figsize=(14, 6))
            plt.subplot(121)
            plt.contour(Xi, Yj, mean_v.reshape(Xi.shape), levels_v)
            plt.plot(board_xy_predicted_grid[:, 0], board_xy_predicted_grid[:, 1], 'ro'), plt.axis("square")
            plt.xlabel("local x"), plt.ylabel("local y")
            plt.title("Mean of GP fit for V"), plt.colorbar()
            plt.subplot(122)
            plt.pcolor(Xi, Yj, cov_v.reshape(Xi.shape))
            plt.plot(board_xy_predicted_grid[:, 0], board_xy_predicted_grid[:, 1], 'ro'), plt.axis("square")
            plt.xlabel("local x"), plt.ylabel("local y")
            plt.title("Variance of GP fit for V"), plt.colorbar()
            plt.show()

        # Remove point that fall beyond the image limits if desired.
        if not out_of_image:
            limits_mask = np.squeeze((board_uv_predicted_grid[:, 0] < 0) |
                                     (board_uv_predicted_grid[:, 0] > image.shape[1]) |
                                     (board_uv_predicted_grid[:, 1] < 0) |
                                     (board_uv_predicted_grid[:, 1] > image.shape[0]))
            if limits_mask.any():
                board_uv_predicted_grid = board_uv_predicted_grid[~limits_mask, :]
                board_xy_predicted_grid = board_xy_predicted_grid[~limits_mask, :]

        board_uv_predicted_grid, board_xy_predicted_grid = self._reset_origin_and_order(board_uv_predicted_grid,
                                                                                        board_xy_predicted_grid)

        return board_uv_predicted_grid, board_xy_predicted_grid

    def dewarp_image(self, image: npt.NDArray, board_uv: npt.NDArray, board_xy: npt.NDArray,
                     use_stored: bool = True) -> npt.NDArray:
        """ Remove lens and perspective distortion from the image.

        This method can either be performed separately, in which case new Gaussian processes need to be fitted to the
        checkerboard, or after using :py:meth:`.fit_and_predict_board`, where the stored Gaussian
        processes can be used instead.

        :param image: Original image that needs to be dewarped.
        :param board_uv: Image corner coordinates (u, v).
        :param board_xy: Local corner coordinates (x, y).
        :param use_stored: Whether to reuse the stored Gaussian processes or fit new ones.
        :returns: The dewarped image.
        """
        if not use_stored or (self.m_xy_to_u is None or self.m_xy_to_v is None):
            self.m_xy_to_u, self.m_xy_to_v, self.scaler = self._train_checkerboard(board_uv, board_xy)

        min_local_x = np.min(board_xy[:, 0])
        max_local_x = np.max(board_xy[:, 0])
        min_local_y = np.min(board_xy[:, 1])
        max_local_y = np.max(board_xy[:, 1])
        nr_of_cols = int(max_local_x - min_local_x) + 1
        nr_of_rows = int(max_local_y - min_local_y) + 1
        res_u = self.dewarped_res_factor * nr_of_cols
        res_v = self.dewarped_res_factor * nr_of_rows
        dewarped_image = np.zeros([res_v, res_u, 3], dtype=np.uint8)

        min_local_x = np.min(board_xy[:, 0])
        max_local_x = np.max(board_xy[:, 0])
        min_local_y = np.min(board_xy[:, 1])
        max_local_y = np.max(board_xy[:, 1])

        xs = np.linspace(min_local_x, max_local_x, res_u)
        ys = np.linspace(min_local_y, max_local_y, res_v)

        [xi, yj] = np.meshgrid(xs, ys)
        xy_test = np.vstack((xi.ravel(), yj.ravel())).T
        xy_test_scaled = self.scaler.transform(xy_test)
        mean_u, _ = self.m_xy_to_u.predict_noiseless(xy_test_scaled, full_cov=False)
        mean_v, _ = self.m_xy_to_v.predict_noiseless(xy_test_scaled, full_cov=False)

        UVForEntireGridAndAllInBetween = np.concatenate((mean_u, mean_v), axis=1)
        cnt = 0
        for j in range(res_v):
            for i in range(res_u):
                new_UV = UVForEntireGridAndAllInBetween[cnt, :]
                u = int(new_UV[0])
                v = int(new_UV[1])
                if u < image.shape[1] and v < image.shape[0] and u > 0 and v > 0:
                    new_pixel_value = image[v, u]
                else:
                    new_pixel_value = (0, 0, 0)
                dewarped_image[res_v - 1 - j, i] = new_pixel_value
                cnt += 1

        return dewarped_image

    def _train_gp(self, board_xy_scaled: npt.NDArray, training_image_axis: npt.NDArray) -> Any:
        """Train model for a single image coordinate axis."""
        k_xy_to_image_axis = GPy.kern.RBF(2)

        m_xy_to_image_axis = GPy.models.GPRegression(board_xy_scaled, training_image_axis, k_xy_to_image_axis,
                                                     normalizer=True)
        m_xy_to_image_axis.kern.lengthscale = 10
        if self.likelihood_variance_bounded:
            m_xy_to_image_axis.likelihood.variance.constrain_bounded(self.min_likelihood_variance,
                                                                     self.max_likelihood_variance)
            m_xy_to_image_axis.likelihood.variance.fix()
        if self.lengthscale_bounded:
            m_xy_to_image_axis.rbf.lengthscale.constrain_bounded(self.min_lengthscale, self.max_lengthscale)
        m_xy_to_image_axis.optimize_restarts(messages=False, num_restarts=self.num_restarts, verbose=False,
                                             max_iters=self.max_nr_of_iters, optimizer=self.optimizer)
        return m_xy_to_image_axis

    def _train_checkerboard(self, board_uv, board_xy):
        """Train models for entire checkerboard."""
        scaler = StandardScaler()
        scaler.fit(board_xy)
        board_xy_scaled = scaler.transform(board_xy)

        # Find map between board_xy and board_uv, i.e. training GPs
        training_us = board_uv[:, 0]
        training_us = training_us[:, None]
        m_xy_to_u = self._train_gp(board_xy_scaled, training_us)

        training_vs = board_uv[:, 1]
        training_vs = training_vs[:, None]
        m_xy_to_v = self._train_gp(board_xy_scaled, training_vs)
        return m_xy_to_u, m_xy_to_v, scaler

    @staticmethod
    def _expand_board_xy(board_xy: npt.NDArray, expansion_factor: int, horizontal: bool, vertical: bool) -> npt.NDArray:
        """Expand local coordinate array with additional rows and columns of local coordinates."""
        # new min max values
        if horizontal:
            min_x = np.min(board_xy[:, 0]) - expansion_factor
            max_x = np.max(board_xy[:, 0]) + expansion_factor
        else:
            min_x = np.min(board_xy[:, 0])
            max_x = np.max(board_xy[:, 0])
        if vertical:
            min_y = np.min(board_xy[:, 1]) - expansion_factor
            max_y = np.max(board_xy[:, 1]) + expansion_factor
        else:
            min_y = np.min(board_xy[:, 1])
            max_y = np.max(board_xy[:, 1])

        x_positions = np.arange(min_x + 1, max_x, 1)
        y_positions = np.arange(min_y + 1, max_y, 1)
        north = np.transpose(np.array([x_positions, [max_y] * len(x_positions)]))
        south = np.transpose(np.array([x_positions, [min_y] * len(x_positions)]))
        east = np.transpose(np.array([[max_x] * len(y_positions), y_positions]))
        west = np.transpose(np.array([[min_x] * len(y_positions), y_positions]))
        corners = np.transpose(np.array([[min_x, min_x, max_x, max_x], [min_y, max_y, min_y, max_y]]))

        if horizontal and vertical:
            new_board_xy = np.concatenate((north, south, east, west, corners))
        elif horizontal:
            new_board_xy = np.concatenate((east, west, corners))
        else:
            new_board_xy = np.concatenate((north, south, corners))

        return new_board_xy

    @staticmethod
    def _reset_origin_and_order(board_uv: npt.NDArray, board_xy: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        """Reset the origin of the local coordinate axes and order the corners according to their local position."""
        x_min = np.min(board_xy[:, 0])
        y_min = np.min(board_xy[:, 1])
        board_xy[:, 0] = board_xy[:, 0] - x_min
        board_xy[:, 1] = board_xy[:, 1] - y_min

        ordered_indexes = np.lexsort((board_xy[:, 1], board_xy[:, 0]))
        board_uv = board_uv[ordered_indexes]
        board_xy = board_xy[ordered_indexes]

        return board_uv, board_xy
