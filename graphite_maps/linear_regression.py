import numpy as np
import scipy.sparse as sp
from scipy.integrate import quad
from scipy.sparse import spmatrix
from scipy.stats import chi2
from sklearn.linear_model import LassoCV, SGDRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from welford import Welford


def linear_l1_regression(U, Y, verbose_level: int = 0):
    """Performs LASSO regression for each response in Y against predictors in
    U, constructing a sparse matrix of regression coefficients.

    The function scales features in U using standard scaling before applying
    LASSO, then re-scales the coefficients to the original scale of U. This
    extracts the effect of each feature in U on each response in Y, ignoring
    intercepts and constant terms.

    Parameters
    ----------
    U : np.ndarray
        2D array of predictors with shape (n, p).
    Y : np.ndarray
        2D array of responses with shape (n, m).

    Returns
    -------
    H_sparse : scipy.sparse.csc_matrix
        Sparse matrix (m, p) with re-scaled LASSO regression coefficients for
        each response in Y.

    Raises
    ------
    AssertionError
        If the number of samples in U and Y do not match, or if the shape of
        H_sparse is not (m, p).
    """
    n, p = U.shape  # p: number of features
    n_y, m = Y.shape  # m: number of y responses

    # Assert that the first dimension of U and Y are the same
    assert n == n_y, "Number of samples in U and Y must be the same"

    if verbose_level > 0:
        print(f"Learning sparse linear map of shape {(m, p)}")

    scaler_u = StandardScaler()
    U_scaled = scaler_u.fit_transform(U)

    scaler_y = StandardScaler()
    Y_scaled = scaler_y.fit_transform(Y)

    # Loop over features
    i_H, j_H, values_H = [], [], []
    for j in tqdm(range(m), desc="Learning sparse linear map for each response"):
        y_j = Y_scaled[:, j]

        # Learn individual regularization and fit
        eps = 1e-3
        max_iter = 10000
        model_cv = LassoCV(cv=10, fit_intercept=False, max_iter=max_iter, eps=eps)
        model_cv.fit(U_scaled, y_j)

        # Extract coefficients
        for non_zero_ind in model_cv.coef_.nonzero()[0]:
            i_H.append(j)
            j_H.append(non_zero_ind)
            values_H.append(
                scaler_y.scale_[j]
                * model_cv.coef_[non_zero_ind]
                / scaler_u.scale_[non_zero_ind]
            )

    H_sparse = sp.csc_matrix(
        (np.array(values_H), (np.array(i_H), np.array(j_H))), shape=(m, p)
    )

    # Assert shape of H_sparse
    assert H_sparse.shape == (m, p), "Shape of H_sparse must be (m, p)"

    if verbose_level > 0:
        print(
            f"Total elements: {m * p}\n"
            f"Non-zero elements: {H_sparse.nnz}\n"
            f"Fraction of non-zeros: {H_sparse.nnz / (m * p)}"
        )

    return H_sparse


class IncrementalSparseRegressor:
    """Incremental sparse regression using SGD with ElasticNet penalty.

    This class provides an sklearn-style partial_fit interface for learning
    sparse linear regression coefficients when data is too large to fit in
    memory.

    Parameters
    ----------
    p : int
        Number of features (columns in U).
    m : int
        Number of response variables (columns in Y).
    alpha : float, default=0.0001
        Regularization strength. Higher values encourage sparser solutions.
    l1_ratio : float, default=0.9
        ElasticNet mixing parameter. l1_ratio=1 is pure L1 (LASSO),
        l1_ratio=0 is pure L2 (Ridge). Values between give elastic net.
    sparsity_threshold : float, default=1e-4
        Coefficients with absolute value below this threshold are set to zero
        when finalize() is called.
    verbose_level : int, default=0
        Verbosity level for progress reporting.

    Examples
    --------
    Two-pass workflow:

    >>> regressor = IncrementalSparseRegressor(p=500, m=100)
    >>> # Pass 1: Compute global statistics
    >>> for U_batch, Y_batch in data_loader:
    ...     regressor.partial_fit_scaler(U_batch, Y_batch)
    >>> # Pass 2: Train the model
    >>> for U_batch, Y_batch in data_loader:
    ...     regressor.partial_fit(U_batch, Y_batch)
    >>> H = regressor.finalize()

    Notes
    -----
    **Two-Pass Workflow with Welford's Algorithm**

    This class uses a two-pass approach to solve the "moving target" problem
    in online regularized regression.

    In sparse regression (LASSO/ElasticNet), features must be scaled/standardized
    (z-score) so that the regularization penalty treats all features equally.
    However, in an online setting, the sample mean and standard deviation are not
    known a priori.

    The two-pass approach:
    1.  **Pass 1 (Statistics)**: Call partial_fit_scaler() on all data to compute
        global mean and variance using Welford's online algorithm.
    2.  **Pass 2 (Training)**: Call partial_fit() on all data. On the first call,
        the scaler is frozen using the statistics from Pass 1, then training
        proceeds with a fixed coordinate system.
    """

    def __init__(
        self,
        p: int,
        m: int,
        alpha: float = 0.0001,
        l1_ratio: float = 0.9,
        random_state: int | None = None,
        sparsity_threshold: float = 1e-4,
        verbose_level: int = 0,
    ):
        self.p = p
        self.m = m
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.random_state = random_state
        self.sparsity_threshold = sparsity_threshold
        self.verbose_level = verbose_level

        # Running statistics for standardization using Welford's algorithm
        self._welford_u = Welford()
        self._welford_y = Welford()
        self._mean_u: np.ndarray | None = None
        self._mean_y: np.ndarray | None = None
        self._std_u: np.ndarray | None = None
        self._std_y: np.ndarray | None = None

        self._scaler_frozen = False

        # SGD models (one per response variable)
        self._models: list[SGDRegressor] | None = None
        self._is_finalized = False

    def _update_scaler_stats(self, U_batch: np.ndarray, Y_batch: np.ndarray) -> None:
        """Update running statistics using Welford's algorithm."""
        self._welford_u.add_all(U_batch, backup_flg=False)
        self._welford_y.add_all(Y_batch, backup_flg=False)

    def _validate_batch_shapes(self, U_batch: np.ndarray, Y_batch: np.ndarray) -> None:
        """Validate that batch shapes match expected dimensions."""
        batch_n, batch_p = U_batch.shape
        batch_n_y, batch_m = Y_batch.shape

        assert batch_p == self.p, f"Expected {self.p} features, got {batch_p}"
        assert batch_m == self.m, f"Expected {self.m} responses, got {batch_m}"
        assert batch_n == batch_n_y, "U_batch and Y_batch must have same number of rows"

    def partial_fit_scaler(self, U_batch: np.ndarray, Y_batch: np.ndarray) -> None:
        """Update scaling statistics without training the model.

        This is Pass 1 of the two-pass workflow: iterate over all data to
        compute global mean and variance using Welford's online algorithm.

        Parameters
        ----------
        U_batch : np.ndarray
            2D array of predictors with shape (batch_size, p).
        Y_batch : np.ndarray
            2D array of responses with shape (batch_size, m).
        """
        if self._scaler_frozen:
            raise RuntimeError("Cannot update scaler after training has started")

        self._validate_batch_shapes(U_batch, Y_batch)
        self._update_scaler_stats(U_batch, Y_batch)

    def partial_fit(
        self, U_batch: np.ndarray, Y_batch: np.ndarray
    ) -> "IncrementalSparseRegressor":
        """Incrementally fit the regressor with a batch of data.

        This is Pass 2 of the two-pass workflow. On the first call, the scaler
        is frozen using statistics from partial_fit_scaler() calls (Pass 1),
        then training proceeds.

        Parameters
        ----------
        U_batch : np.ndarray
            2D array of predictors with shape (batch_size, p).
        Y_batch : np.ndarray
            2D array of responses with shape (batch_size, m).

        Returns
        -------
        self : IncrementalSparseRegressor
            Returns self for method chaining.

        Raises
        ------
        RuntimeError
            If partial_fit_scaler() was not called before partial_fit().
        """
        if self._is_finalized:
            raise RuntimeError("Cannot call partial_fit after finalize()")

        self._validate_batch_shapes(U_batch, Y_batch)

        if not self._scaler_frozen:
            if self._welford_u.count <= 0:
                raise RuntimeError(
                    "Must call partial_fit_scaler() before partial_fit(). "
                    "Use a two-pass workflow: Pass 1 computes statistics, "
                    "Pass 2 trains the model."
                )
            self._freeze_scaler_and_initialize_models()

        self._train_on_batch(U_batch, Y_batch)
        return self

    def _freeze_scaler_and_initialize_models(self) -> None:
        if self._scaler_frozen:
            return
        if self._welford_u.count <= 0:
            raise RuntimeError("Cannot freeze scaler before observing any samples")

        # Extract statistics from Welford accumulators
        self._mean_u = self._welford_u.mean.copy()
        self._mean_y = self._welford_y.mean.copy()

        # Use population variance (var_p) since we want to standardize with the
        # actual observed variance, not an unbiased estimate
        std_u = np.sqrt(self._welford_u.var_p)
        std_y = np.sqrt(self._welford_y.var_p)

        # Avoid division by zero for constant features
        std_u[std_u < 1e-10] = 1.0
        std_y[std_y < 1e-10] = 1.0

        self._std_u = std_u
        self._std_y = std_y

        if self._models is None:
            self._models = [
                SGDRegressor(
                    loss="squared_error",
                    penalty="elasticnet",
                    alpha=self.alpha,
                    l1_ratio=self.l1_ratio,
                    fit_intercept=False,
                    max_iter=1,
                    tol=None,
                    warm_start=True,
                    random_state=self.random_state,
                )
                for _ in range(self.m)
            ]

        self._scaler_frozen = True

    def _train_on_batch(self, U_batch: np.ndarray, Y_batch: np.ndarray) -> None:
        if self._models is None:
            raise RuntimeError("Internal error: models not initialized")
        if self._std_u is None or self._std_y is None:
            raise RuntimeError("Internal error: scaler not initialized")
        if self._mean_u is None or self._mean_y is None:
            raise RuntimeError("Internal error: mean not initialized")

        U_scaled = (U_batch - self._mean_u) / self._std_u
        Y_scaled = (Y_batch - self._mean_y) / self._std_y

        for j in range(self.m):
            self._models[j].partial_fit(U_scaled, Y_scaled[:, j])

    def finalize(self, verbose_level: int | None = None) -> spmatrix:
        """Build the final sparse coefficient matrix.

        Parameters
        ----------
        verbose_level : int, optional
            Override instance verbose_level for this call.

        Returns
        -------
        H_sparse : scipy.sparse.csc_matrix
            Sparse matrix (m, p) with regression coefficients.
        """
        if self._is_finalized:
            raise RuntimeError("finalize() has already been called")
        if self._models is None:
            raise RuntimeError(
                "Must call partial_fit() at least once before finalize()"
            )

        if not self._scaler_frozen:
            raise RuntimeError(
                "Must call partial_fit() at least once before finalize()"
            )

        verbose = verbose_level if verbose_level is not None else self.verbose_level

        # These are guaranteed to be set after partial_fit() is called
        assert self._std_u is not None and self._std_y is not None

        # Extract coefficients and build sparse matrix
        i_H, j_H, values_H = [], [], []
        for j in range(self.m):
            coefficients = self._models[j].coef_

            # Re-scale coefficients to original scale
            scaled_coefs = self._std_y[j] * coefficients / self._std_u

            # Apply sparsity threshold and extract non-zeros
            for i, coef in enumerate(scaled_coefs):
                if np.abs(coef) > self.sparsity_threshold:
                    i_H.append(j)
                    j_H.append(i)
                    values_H.append(coef)

        H_sparse = sp.csc_matrix(
            (np.array(values_H), (np.array(i_H), np.array(j_H))),
            shape=(self.m, self.p),
        )

        if verbose > 0:
            print(
                f"Finalized sparse H matrix:\n"
                f"  Total elements: {self.m * self.p}\n"
                f"  Non-zero elements: {H_sparse.nnz}\n"
                f"  Fraction of non-zeros: {H_sparse.nnz / (self.m * self.p):.6f}"
            )

        self._is_finalized = True
        self._models = None  # Free memory

        return H_sparse


def expected_max_chisq(p):
    """Expected maximum of p central chi-square(1) random variables."""

    def dmaxchisq(x):
        return 1.0 - np.exp(p * chi2.logcdf(x, df=1))

    expectation, _ = quad(dmaxchisq, 0, np.inf)
    return expectation


def mse(residuals):
    return 0.5 * np.mean(residuals**2)


def calculate_psi_M(x, y, beta_estimate):
    """The psi/score function for mse: 0.5*residual**2."""
    residuals = y - beta_estimate * x
    psi = -residuals * x
    M = -np.mean(x**2)
    return psi, M


def calculate_influence(x, y, beta_estimate):
    """The influence of (x, y) on beta_estimate as an mse M-estimator."""
    psi, M = calculate_psi_M(x, y, beta_estimate)
    return psi / M


def boost_linear_regression(
    X, y, learning_rate=0.5, tol=1e-6, max_iter=10000, effective_dimension=None
):
    """Boost coefficients of linearly regressing y on standardized X.

    The coefficient selection utilizes information theoretic weighting. The
    stopping criterion utilizes information theoretic loss-reduction.
    """
    n_samples, n_features = X.shape
    coefficients = np.zeros(n_features)
    residuals = y.copy()
    residuals_loo = y.copy()

    # A stricter criterion is the loo-adjustment: mse(residuals_loo)-mse
    # (residuals). This converges to TIC. Under certain conditions this is AIC.
    # At worst, we are maximizing squares. See Lunde 2020 Appendix A. This
    #  needs to be adjusted for.
    # The mse_factor adjusts for this.
    # if effective_dimension is None:
    #    effective_dimension = n_features
    # mse_factor = expected_max_chisq(np.ceil(effective_dimension))

    for _ in range(max_iter):
        coef_changes = np.dot(X.T, residuals) / n_samples

        # Could be adjusted for IC -- some features already included
        # The IC would build in additional motivation for sparsity
        feature_evaluation = np.abs(coef_changes)

        # Select feature based on loss criterion
        best_feature = np.argmax(feature_evaluation)
        beta_estimate = coef_changes[best_feature]

        # adjust to loo estimates for coef_change
        influence = calculate_influence(X[:, best_feature], residuals, beta_estimate)
        beta_estimate_loo = beta_estimate - influence / n_samples

        # residuals_full = residuals - beta_estimate * X[:, best_feature]
        residuals_full_loo = (
            residuals_loo - learning_rate * beta_estimate_loo * X[:, best_feature]
        )

        if mse(residuals_loo) < mse(residuals_full_loo):
            break

        # Check if adding the full weight of the feature would decrease loss
        # if mse(residuals) < mse(residuals_full) + mse_factor * (
        #    mse(residuals_full_loo) - mse(residuals_full)
        # ):
        #    break

        coef_change = beta_estimate * learning_rate

        # Check for convergence
        if np.abs(coef_change) < tol:
            break
        else:
            # Update
            residuals -= coef_change * X[:, best_feature]
            coefficients[best_feature] += coef_change

            # loo update
            residuals_loo = residuals_full_loo

    # ensure cutoff values -- very small if data standardized
    # prefer sparsity
    cutoff = 2.0 * learning_rate / np.sqrt(n_samples)  # 2.0: 95% ci-ish
    coefficients[np.abs(coefficients) < cutoff] = 0

    return coefficients


def linear_boost_ic_regression(
    U, Y, learning_rate=0.5, effective_dimension=None, verbose_level: int = 0
):
    """Performs boosted linear regression for each response in Y against
    predictors in U, constructing a sparse matrix of regression coefficients.
    The complexity is tuned with an information theoretic approach.

    The function scales features in U using standard scaling before learning
    the coefficients, then re-scales the coefficients to the original scale of
    U. This extracts the effect of each feature in U on each response in Y,
    ignoring intercepts and constant terms.

    Parameters
    ----------
    U : np.ndarray
        2D array of predictors with shape (n, p).
    Y : np.ndarray
        2D array of responses with shape (n, m).

    Returns
    -------
    H_sparse : scipy.sparse.csc_matrix
        Sparse matrix (m, p) with re-scaled LASSO regression coefficients for
        each response in Y.

    Raises
    ------
    AssertionError
        If the number of samples in U and Y do not match, or if the shape of
        H_sparse is not (m, p).
    """
    n, p = U.shape  # p: number of features
    n_y, m = Y.shape  # m: number of y responses

    # Assert that the first dimension of U and Y are the same
    assert n == n_y, "Number of samples in U and Y must be the same"

    if verbose_level > 0:
        print(f"Learning sparse linear map of shape {(m, p)}")

    scaler_u = StandardScaler()
    U_scaled = scaler_u.fit_transform(U)

    scaler_y = StandardScaler()
    Y_scaled = scaler_y.fit_transform(Y)

    # Loop over features
    i_H, j_H, values_H = [], [], []
    for j in tqdm(range(m), desc="Learning sparse linear map for each response"):
        y_j = Y_scaled[:, j]

        # Learn individual fit
        coefficients_j = boost_linear_regression(
            U_scaled,
            y_j,
            learning_rate=learning_rate,
            effective_dimension=effective_dimension,
        )

        # Extract coefficients
        for non_zero_ind in coefficients_j.nonzero()[0]:
            i_H.append(j)
            j_H.append(non_zero_ind)
            values_H.append(
                scaler_y.scale_[j]
                * coefficients_j[non_zero_ind]
                / scaler_u.scale_[non_zero_ind]
            )

    H_sparse = sp.csc_matrix(
        (np.array(values_H), (np.array(i_H), np.array(j_H))), shape=(m, p)
    )

    # Assert shape of H_sparse
    assert H_sparse.shape == (m, p), "Shape of H_sparse must be (m, p)"

    if verbose_level > 0:
        print(
            f"Total elements: {m * p}\n"
            f"Non-zero elements: {H_sparse.nnz}\n"
            f"Fraction of non-zeros: {H_sparse.nnz / (m * p)}"
        )

    return H_sparse


def response_residual(
    U: np.ndarray, Y: np.ndarray, H: spmatrix, verbose_level: int = 0
) -> np.ndarray:
    """Residual from regression H for Y on U"""
    n_u, p = U.shape
    n_y, m = Y.shape
    assert n_u == n_y, "Number of ensembles must be the same"
    assert (m, p) == H.shape, "Coefficients in H must match U and Y dimensions"

    if verbose_level > 0:
        print("Calculating response residuals")

    return Y - U @ H.T


def residual_variance(
    U: np.ndarray, Y: np.ndarray, H: spmatrix, verbose_level: int = 0
) -> np.ndarray:
    """Variance in Y not explained by U through H"""

    n_u, p = U.shape
    n_y, m = Y.shape
    assert n_u == n_y, "Number of ensembles must be the same"
    assert (m, p) == H.shape, "Coefficients in H must match U and Y dimensions"

    R = response_residual(U, Y, H)
    unexplained_variance = np.var(R, axis=0)

    assert unexplained_variance.shape == (m,), (
        "Number of variance components must match number of observations"
    )

    if verbose_level > 0:
        print("Calculating unexplained variance")

    return unexplained_variance
