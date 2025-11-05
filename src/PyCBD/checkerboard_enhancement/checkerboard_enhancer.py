"""This module contains the checkerboard enhancers."""

from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import cv2
import matplotlib.pyplot as plt
import numpy.typing as npt
from typing import Tuple, Any, Optional
import logging
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.priors import UniformPrior
import numpy as np
import warnings


class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, lower_scale=None, higher_scale=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        # Create base kernel with optional constraints
        if lower_scale is not None and higher_scale is not None:
            # Make sure lower_scale is at least 0.01 to avoid numerical issues
            lower_scale = max(0.01, lower_scale)
            base_kernel = RBFKernel(
                lengthscale_constraint=Interval(lower_scale, higher_scale)
            )
        else:
            base_kernel = RBFKernel(
                lengthscale_constraint=Interval(0.01, 10.0)
            )

        self.covar_module = ScaleKernel(
            base_kernel,
            outputscale_constraint=Interval(0.01, 100000.0)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def predict_noiseless_gpytorch(model, likelihood, X_test, full_cov=False):
    """
    Mimic GPy's predict_noiseless(..., full_cov=False).
    Returns (mean, variance) both of shape (N,) or (N, 1).
    """
    model.eval()
    likelihood.eval()

    X_test = torch.from_numpy(X_test).to(torch.float32)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(X_test))

    with torch.no_grad():
        latent_pred = model(X_test)

    mean_norm = latent_pred.mean
    if full_cov:
        var_norm = latent_pred.covariance_matrix
    else:
        var_norm = latent_pred.variance

    # Denormalize
    mean = mean_norm * model._y_std + model._y_mean
    var = var_norm * (model._y_std ** 2)  # variance scales with square

    # Return shapes consistent with GPy: column vectors (N, 1)
    mean = mean.unsqueeze(-1)  # (N, 1)
    var = var.unsqueeze(-1) if not full_cov else var  # (N, 1) or (N, N)

    return mean.numpy(), var.numpy()

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
        self.max_nr_of_iters: int = 50
        self.optimizer: str = 'lbfgs'
        self.num_restarts: int = 3
        self.lengthscale_bounded: bool = True
        self.min_lengthscale: float = 1e-8
        self.max_lengthscale: float = 25.0
        self.likelihood_variance_bounded: bool = False
        self.min_likelihood_variance = 1e-8
        self.max_likelihood_variance = 1.0
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

            self.m_xy_to_u, self.m_xy_to_v, self.likelihood_u, self.likelihood_v, scaler = self._train_checkerboard(board_uv, board_xy)

            # Expand board_xy by expansion_factor
            new_board_xy = self._expand_board_xy(board_xy, expansion_factor, expand_horizontal, expand_vertical)

            # Use map to find more corners
            new_board_xy_scaled = scaler.transform(new_board_xy)
            mean_u_new, cov_u_new = predict_noiseless_gpytorch(
                self.m_xy_to_u, self.likelihood_u, new_board_xy_scaled, full_cov=False
            )

            mean_v_new, cov_v_new = predict_noiseless_gpytorch(
                self.m_xy_to_v, self.likelihood_v, new_board_xy_scaled, full_cov=False
            )

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
                mean_u, cov_u = predict_noiseless_gpytorch(
                    self.m_xy_to_u, self.likelihood_u, xy_test_scaled, full_cov=False
                )
                mean_v, cov_v = predict_noiseless_gpytorch(
                    self.m_xy_to_v, self.likelihood_v, xy_test_scaled, full_cov=False
                )
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
        self.m_xy_to_u, self.m_xy_to_v, self.likelihood_u, self.likelihood_v, self.scaler = self._train_checkerboard(board_uv, board_xy)

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
        mean_u, cov_u = predict_noiseless_gpytorch(
            self.m_xy_to_u, self.likelihood_u, board_xy_predicted_grid_scaled, full_cov=False
        )
        mean_v, cov_v = predict_noiseless_gpytorch(
            self.m_xy_to_v, self.likelihood_v, board_xy_predicted_grid_scaled, full_cov=False
        )
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
            mean_u, cov_u = predict_noiseless_gpytorch(
                self.m_xy_to_u, self.likelihood_u, xy_test_scaled, full_cov=False
            )
            mean_v, cov_v = predict_noiseless_gpytorch(
                self.m_xy_to_v, self.likelihood_v, xy_test_scaled, full_cov=False
            )

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
            self.m_xy_to_u, self.m_xy_to_v, self.likelihood_u, self.likelihood_v, self.scaler = self._train_checkerboard(board_uv, board_xy)

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

        mean_u, _ = predict_noiseless_gpytorch(
            self.m_xy_to_u, self.likelihood_u, xy_test_scaled, full_cov=False
        )
        mean_v, _ = predict_noiseless_gpytorch(
            self.m_xy_to_v, self.likelihood_v, xy_test_scaled, full_cov=False
        )

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

    def _train_gp(self, board_xy_scaled: np.ndarray, training_image_axis: np.ndarray) -> Any:
        device = torch.device("cpu")
        X = torch.from_numpy(board_xy_scaled).to(torch.float32).to(device)
        y = torch.from_numpy(training_image_axis.squeeze()).to(torch.float32).to(device)

        # Normalize
        y_mean = y.mean()
        y_std = y.std() + 1e-8
        y_normalized = (y - y_mean) / y_std

        best_loss = float('inf')
        best_model = None
        best_likelihood = None

        for restart in range(self.num_restarts):
            try:
                # Setup likelihood with constraints
                if self.likelihood_variance_bounded:
                    likelihood = GaussianLikelihood(
                        noise_constraint=Interval(self.min_likelihood_variance, self.max_likelihood_variance)
                    )
                else:
                    likelihood = GaussianLikelihood(
                        noise_constraint=Interval(1e-4, 1.0)  # Add reasonable default bounds
                    )

                # Setup model - now works with or without constraints
                if self.lengthscale_bounded:
                    model = GPModel(X, y_normalized, likelihood, self.min_lengthscale, self.max_lengthscale)
                else:
                    model = GPModel(X, y_normalized, likelihood)

                # Initialize parameters to stable values
                with torch.no_grad():
                    model.covar_module.base_kernel.lengthscale = 1.0  # Start at 1.0, not 10.0!
                    model.covar_module.outputscale = 1.0
                    likelihood.noise = 0.1

                model = model.to(device)
                likelihood = likelihood.to(device)

                # Fix likelihood variance if needed (matching GPy's .fix())
                if self.likelihood_variance_bounded:
                    likelihood.noise_covar.raw_noise.requires_grad = False

                model.train()
                likelihood.train()

                mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

                # Use configurable optimizer (matching GPy version)
                if self.optimizer == 'lbfgs':
                    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1,
                                                  max_iter=20, line_search_fn='strong_wolfe')

                    # LBFGS requires a closure function
                    def closure():
                        optimizer.zero_grad()
                        output = model(X)
                        loss = -mll(output, y_normalized)
                        loss.backward()
                        return loss

                    # LBFGS optimization loop with higher jitter
                    with gpytorch.settings.cholesky_jitter(1e-3):
                        for i in range(self.max_nr_of_iters):
                            loss = optimizer.step(closure)

                            # Check for invalid loss
                            if not torch.isfinite(loss):
                                raise ValueError(f"Non-finite loss at iteration {i}: {loss.item()}")

                else:  # Adam or other optimizers
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

                    # Standard optimization loop with higher jitter
                    with gpytorch.settings.cholesky_jitter(1e-3):
                        for i in range(self.max_nr_of_iters):
                            optimizer.zero_grad()
                            output = model(X)
                            loss = -mll(output, y_normalized)

                            # Check for invalid loss
                            if not torch.isfinite(loss):
                                raise ValueError(f"Non-finite loss at iteration {i}: {loss.item()}")

                            loss.backward()
                            optimizer.step()

                # Successfully completed training
                current_loss = loss.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_model = model
                    best_likelihood = likelihood

            except Exception as e:
                warnings.warn(f"Restart {restart + 1}/{self.num_restarts} failed: {e}")
                if restart == self.num_restarts - 1 and best_model is None:
                    # Last restart and no success yet - print more detail
                    print(f"Data shapes: X={X.shape}, y={y_normalized.shape}")
                    print(
                        f"Data ranges: X=[{X.min():.3f}, {X.max():.3f}], y=[{y_normalized.min():.3f}, {y_normalized.max():.3f}]")
                    print(
                        f"Constraint settings: likelihood_bounded={self.likelihood_variance_bounded}, lengthscale_bounded={self.lengthscale_bounded}")
                    if self.lengthscale_bounded:
                        print(f"Lengthscale bounds: [{self.min_lengthscale}, {self.max_lengthscale}]")
                continue

        if best_model is None:
            raise RuntimeError(f"All {self.num_restarts} GP optimization restarts failed. Check data and constraints.")

        best_model._y_mean = y_mean
        best_model._y_std = y_std

        return best_model, best_likelihood

    def _train_checkerboard(self, board_uv, board_xy):
        scaler = StandardScaler()
        scaler.fit(board_xy)
        board_xy_scaled = scaler.transform(board_xy)

        training_us = board_uv[:, 0][:, None]
        m_xy_to_u, likelihood_u = self._train_gp(board_xy_scaled, training_us)  # ← unpack

        training_vs = board_uv[:, 1][:, None]
        m_xy_to_v, likelihood_v = self._train_gp(board_xy_scaled, training_vs)  # ← unpack

        return m_xy_to_u, m_xy_to_v, likelihood_u, likelihood_v, scaler

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