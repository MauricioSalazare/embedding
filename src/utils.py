import numpy as np
import warnings


class StandardizationPipeline:
    def __init__(self, order="row-column"):
        """
        Initialize the standardization pipeline with a given order.

        Parameters:
        - order: str, the standardization order. Options:
            - "row-column": Row-wise standardization, then column-wise
            - "column-row": Column-wise standardization, then row-wise
            - "row": Only row-wise standardization
            - "column": Only column-wise standardization
        """
        self.order = order
        self.row_means, self.row_stds = None, None
        self.col_means, self.col_stds = None, None

    def _standardize(self, X, axis):
        """Applies standardization along the specified axis (0 = column, 1 = row)."""
        means = np.nanmean(X, axis=axis, keepdims=True)
        stds = np.nanstd(X, axis=axis, keepdims=True)
        stds = _handle_zeros_in_scale(stds)  # Use your function to avoid zero-division issues
        return (X - means) / stds, means, stds

    def _inverse_standardize(self, X_transformed, means, stds):
        """Reverts the standardization process using stored means and stds."""
        return (X_transformed * stds) + means

    def fit_transform(self, X):
        """Apply standardization using the selected order."""
        X = np.asarray(X, dtype=np.float64)

        if self.order == "row-column":
            X, self.row_means, self.row_stds = self._standardize(X, axis=1)
            X, self.col_means, self.col_stds = self._standardize(X, axis=0)

        elif self.order == "column-row":
            X, self.col_means, self.col_stds = self._standardize(X, axis=0)
            X, self.row_means, self.row_stds = self._standardize(X, axis=1)

        elif self.order == "row":
            X, self.row_means, self.row_stds = self._standardize(X, axis=1)

        elif self.order == "column":
            X, self.col_means, self.col_stds = self._standardize(X, axis=0)

        else:
            raise ValueError(
                "Invalid standardization order. "
                "Choose from: 'row-column', 'column-row', 'row', 'column'."
            )

        return X
    def inverse_transform(self, X_transformed):
        """Reverts the applied standardization transformations in the correct order."""
        X_transformed = np.asarray(X_transformed, dtype=np.float64)

        if self.order == "row-column":
            X_transformed = self._inverse_standardize(X_transformed, self.col_means, self.col_stds)
            X_transformed = self._inverse_standardize(X_transformed, self.row_means, self.row_stds)

        elif self.order == "column-row":
            X_transformed = self._inverse_standardize(X_transformed, self.row_means, self.row_stds)
            X_transformed = self._inverse_standardize(X_transformed, self.col_means, self.col_stds)

        elif self.order == "row":
            X_transformed = self._inverse_standardize(X_transformed, self.row_means, self.row_stds)

        elif self.order == "column":
            X_transformed = self._inverse_standardize(X_transformed, self.col_means, self.col_stds)

        else:
            raise ValueError(
                "Invalid standardization order. "
                "Choose from: 'row-column', 'column-row', 'row', 'column'."
            )

        return X_transformed

def _handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    """Set scales of near constant features to 1.

    The goal is to avoid division by very small or zero values.

    Near constant features are detected automatically by identifying
    scales close to machine precision unless they are precomputed by
    the caller and passed with the `constant_mask` kwarg.

    Typically, for standard scaling, the scales are the standard
    deviation while near constant features are better detected on the
    computed variances which are closer to machine precision by
    construction.
    """
    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == 0.0:
            scale = 1.0
        return scale
    # scale is an array
    else:
        if constant_mask is None:
            # Detect near constant values to avoid dividing by a very small
            # value that could lead to surprising results and numerical
            # stability issues.
            constant_mask = scale < 10 * np.finfo(scale.dtype).eps

        if copy:
            # New array to avoid side-effects
            scale = np.asarray(scale, copy=True)
        scale[constant_mask] = 1.0
        return scale


def scale(X, *, axis=0, with_mean=True, with_std=True, copy=True, ddof=0):
    """
    Standardize a dataset along any axis.

    Center to the mean and component wise scale to unit variance.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to center and scale.

    axis : {0, 1}, default=0
        Axis used to compute the means and standard deviations along. If 0,
        independently standardize each feature, otherwise (if 1) standardize
        each sample.

    with_mean : bool, default=True
        If True, center the data before scaling.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : bool, default=True
        If False, try to avoid a copy and scale in place.
        This is not guaranteed to always work in place; e.g. if the data is
        a numpy array with an int dtype, a copy will be returned even with
        copy=False.

    ddof : int, default=0
        Means Delta Degrees of Freedom. The divisor used in calculations is N - ddof, where N represents the number
        of non-NaN elements. By default ddof is zero.

    Returns
    -------
    X_tr : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The transformed data.



    """

    X = np.asarray(X, copy=copy).astype(float)

    if with_mean:
        mean_ = np.nanmean(X, axis)
    if with_std:
        scale_ = np.nanstd(X, axis, ddof=ddof)
    # Xr is a view on the original array that enables easy use of
    # broadcasting on the axis in which we are interested in
    # Any change in X is a change in Xr
    Xr = np.rollaxis(X, axis)

    if with_mean:
        Xr -= mean_
        mean_1 = np.nanmean(Xr, axis=0)
        # Verify that mean_1 is 'close to zero'. If X contains very
        # large values, mean_1 can also be very large, due to a lack of
        # precision of mean_. In this case, a pre-scaling of the
        # concerned feature is efficient, for instance by its mean or
        # maximum.
        if not np.allclose(mean_1, 0):
            warnings.warn(
                "Numerical issues were encountered "
                "when centering the data "
                "and might not be solved. Dataset may "
                "contain too large values. You may need "
                "to prescale your features."
            )
            Xr -= mean_1
    if with_std:
        scale_ = _handle_zeros_in_scale(scale_, copy=False)
        Xr /= scale_
        if with_mean:
            mean_2 = np.nanmean(Xr, axis=0)
            # If mean_2 is not 'close to zero', it comes from the fact that
            # scale_ is very small so that mean_2 = mean_1/scale_ > 0, even
            # if mean_1 was close to zero. The problem is thus essentially
            # due to the lack of precision of mean_. A solution is then to
            # subtract the mean again:
            if not np.allclose(mean_2, 0):
                warnings.warn(
                    "Numerical issues were encountered "
                    "when scaling the data "
                    "and might not be solved. The standard "
                    "deviation of the data is probably "
                    "very close to 0. "
                )
                Xr -= mean_2
    return X
