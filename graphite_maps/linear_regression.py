import numpy as np
import scipy.sparse as sp
from scipy.integrate import quad
from scipy.sparse import spmatrix
from scipy.stats import chi2
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


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


def calculate_nsis(n, p):
    """Calculate the number of predictors to recruit by (I)SIS.

    Follows the SIS R package convention for determining the screening size.

    Parameters
    ----------
    n : int
        Number of samples.
    p : int
        Number of features.

    Returns
    -------
    int
        Number of predictors to screen.
    """
    # Standard recommendation from Fan & Lv (2008)
    nsis = int(n / np.log(n))

    # Ensure nsis is at least 1 and at most p
    return max(1, min(nsis, p))


def marginal_screen(X, y, d, candidate_idx=None):
    """Screen features by absolute marginal correlation with response.

    Follows the SIS R package approach for marginal screening.

    Parameters
    ----------
    X : np.ndarray
        2D array of standardized predictors with shape (n, p).
    y : np.ndarray
        1D array of standardized response with shape (n,).
    d : int
        Number of top features to retain.
    candidate_idx : np.ndarray, optional
        Indices of candidate features to screen. If None, all features are
        considered.

    Returns
    -------
    np.ndarray
        Indices of top-d features ranked by absolute marginal correlation.
    np.ndarray
        All indices ordered by decreasing absolute marginal correlation.
    """
    n_samples = len(y)

    if candidate_idx is None:
        candidate_idx = np.arange(X.shape[1])
        X_candidates = X
    else:
        X_candidates = X[:, candidate_idx]

    # Marginal correlations (for standardized data, this equals correlation)
    correlations = np.abs(X_candidates.T @ y) / n_samples

    # Sort by decreasing correlation
    sorted_local_idx = np.argsort(correlations)[::-1]
    sorted_global_idx = candidate_idx[sorted_local_idx]

    # Return top-d and all sorted indices
    d = min(d, len(candidate_idx))
    return sorted_global_idx[:d], sorted_global_idx


def obtain_conditional_marginal(X, y, selected_idx):
    """Compute residuals after regressing y on selected features.

    Used for conditional screening in ISIS iterations.

    Parameters
    ----------
    X : np.ndarray
        2D array of standardized predictors with shape (n, p).
    y : np.ndarray
        1D array of standardized response with shape (n,).
    selected_idx : np.ndarray
        Indices of already selected features.

    Returns
    -------
    np.ndarray
        Residuals from regressing y on selected features.
    """
    if len(selected_idx) == 0:
        return y

    X_selected = X[:, selected_idx]

    # OLS fit for residuals (using pseudo-inverse for stability)
    coefficients = np.linalg.lstsq(X_selected, y, rcond=None)[0]
    residuals = y - X_selected @ coefficients

    return residuals


def isis_select(
    X,
    y,
    max_iter=10,
    nsis=None,
    learning_rate=0.5,
    effective_dimension=None,
):
    """Iterative Sure Independence Screening (ISIS).

    Performs iterative screening to select a subset of features for
    ultra-high dimensional regression problems where p >> n. This
    implementation follows the SIS R package by Fan, Feng, Samworth, and Wu,
    but uses boosted forward stagewise regression instead of LASSO for
    the regularization step (faster for large problems).

    The algorithm:
    1. Initial SIS: Screen top nsis features by marginal correlation
    2. Regularization: Fit boosted regression on screened features to select
       active set
    3. Conditional screening: Screen remaining features against residuals
    4. Iterate until convergence or max iterations

    Parameters
    ----------
    X : np.ndarray
        2D array of standardized predictors with shape (n, p).
    y : np.ndarray
        1D array of standardized response with shape (n,).
    max_iter : int, optional
        Maximum number of ISIS iterations. Default is 10 (following SIS pkg).
    nsis : int, optional
        Number of predictors to recruit per screening step. If None,
        defaults to n / log(n) following Fan & Lv (2008).
    learning_rate : float, optional
        Learning rate for boosted regression. Default is 0.5.
    effective_dimension : int, optional
        Effective dimension for boosted regression.

    Returns
    -------
    np.ndarray
        Indices of selected features in the original feature space.

    References
    ----------
    Fan, J., & Lv, J. (2008). Sure independence screening for ultrahigh
    dimensional feature space. Journal of the Royal Statistical Society:
    Series B, 70(5), 849-911.

    Saldana, D. F., & Feng, Y. (2018). SIS: An R package for Sure Independence
    Screening in Ultrahigh Dimensional Statistical Models. Journal of
    Statistical Software, 83(2), 1-25.
    """
    n_samples, n_features = X.shape

    # Calculate nsis if not provided (following SIS R package)
    if nsis is None:
        nsis = calculate_nsis(n_samples, n_features)

    # Track models to detect convergence
    models_history = []

    # Initial screening: obtain top nsis features by marginal correlation
    # Finds features with strong marginal (unconditional) correlation with y
    ix0, _ = marginal_screen(X, y, nsis)
    selected_set = set(ix0)

    for _ in range(max_iter):
        # Convert to sorted array for consistent indexing
        ix0_list = sorted(selected_set)

        if len(ix0_list) == 0:
            break

        # Regularization step: fit boosted regression on currently screened
        # features
        X_screened = X[:, ix0_list]

        coefficients = boost_linear_regression(
            X_screened,
            y,
            learning_rate=learning_rate,
            effective_dimension=effective_dimension,
        )

        # Get indices of non-zero coefficients (active set after regularization)
        nonzero_local = np.where(np.abs(coefficients) > 0)[0]

        if len(nonzero_local) == 0:
            # No variables selected, keep original screening
            break

        # Map back to original indices
        ix1 = [ix0_list[i] for i in nonzero_local]

        # Check stopping criteria
        if len(ix1) >= nsis:
            # Maximum number of variables selected
            selected_set = set(ix1)
            break

        # Check if model has been seen before (convergence)
        ix1_tuple = tuple(sorted(ix1))
        if ix1_tuple in models_history:
            selected_set = set(ix1)
            break
        models_history.append(ix1_tuple)

        # Conditional screening on remaining features
        candidate_idx = np.array([i for i in range(n_features) if i not in ix1])

        if len(candidate_idx) == 0:
            selected_set = set(ix1)
            break

        # Number of features to add in this iteration
        pleft = nsis - len(ix1)

        if pleft <= 0:
            selected_set = set(ix1)
            break

        # Compute residuals conditional on selected features
        residuals = obtain_conditional_marginal(X, y, np.array(ix1))

        # Screen remaining features against residuals
        # Finds features with strong conditional correlation with y,
        # features that explain the remaining variance after accounting for
        # already selected features.
        new_idx, _ = marginal_screen(X, residuals, pleft, candidate_idx)

        # Update selected set
        old_set = set(ix1)
        new_set = old_set.union(set(new_idx))

        # Check if set changed
        if new_set == selected_set:
            break

        selected_set = new_set

    return np.array(sorted(selected_set))


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
    U,
    Y,
    learning_rate=0.5,
    effective_dimension=None,
    use_isis=False,
    isis_max_iter=10,
    isis_nsis=None,
    verbose_level: int = 0,
):
    """Performs boosted linear regression for each response in Y against
    predictors in U, constructing a sparse matrix of regression coefficients.
    The complexity is tuned with an information theoretic approach.

    The function scales features in U using standard scaling before learning
    the coefficients, then re-scales the coefficients to the original scale of
    U. This extracts the effect of each feature in U on each response in Y,
    ignoring intercepts and constant terms.

    For ultra-high dimensional problems (p >> n), Iterative Sure Independence
    Screening (ISIS) can be enabled to pre-screen features before boosting,
    dramatically reducing computational cost. The ISIS implementation follows
    the SIS R package by Fan, Feng, Samworth, and Wu.

    Parameters
    ----------
    U : np.ndarray
        2D array of predictors with shape (n, p).
    Y : np.ndarray
        2D array of responses with shape (n, m).
    learning_rate : float, optional
        Learning rate for boosted regression. Default is 0.5.
    effective_dimension : int, optional
        Effective dimension for boosted regression.
    use_isis : bool, optional
        Whether to use Iterative Sure Independence Screening for
        ultra-high dimensional problems. Default is False.
    isis_max_iter : int, optional
        Maximum number of ISIS iterations. Default is 10 (following SIS pkg).
    isis_nsis : int, optional
        Number of predictors to recruit per ISIS screening step. If None,
        defaults to n / log(n) following Fan & Lv (2008).

    Returns
    -------
    H_sparse : scipy.sparse.csc_matrix
        Sparse matrix (m, p) with re-scaled regression coefficients for
        each response in Y.

    Raises
    ------
    AssertionError
        If the number of samples in U and Y do not match, or if the shape of
        H_sparse is not (m, p).

    References
    ----------
    Fan, J., & Lv, J. (2008). Sure independence screening for ultrahigh
    dimensional feature space. Journal of the Royal Statistical Society:
    Series B, 70(5), 849-911.

    Fan, J., & Song, R. (2010). Sure independence screening in generalized
    linear models with NP-dimensionality. The Annals of Statistics, 38(6),
    3567-3604.

    Saldana, D. F., & Feng, Y. (2018). SIS: An R package for Sure Independence
    Screening in Ultrahigh Dimensional Statistical Models. Journal of
    Statistical Software, 83(2), 1-25.
    """
    n, p = U.shape  # p: number of features
    n_y, m = Y.shape  # m: number of y responses

    # Assert that the first dimension of U and Y are the same
    assert n == n_y, "Number of samples in U and Y must be the same"

    if verbose_level > 0:
        print(f"Learning sparse linear map of shape {(m, p)}")

    if verbose_level > 0 and use_isis:
        print(
            f"Using ISIS pre-screening (following SIS R package)\n"
            f"ISIS parameters: max_iter={isis_max_iter}, "
            f"nsis={isis_nsis if isis_nsis else 'n/log(n)'}"
        )

    scaler_u = StandardScaler()
    U_scaled = scaler_u.fit_transform(U)

    scaler_y = StandardScaler()
    Y_scaled = scaler_y.fit_transform(Y)

    # Loop over features
    i_H, j_H, values_H = [], [], []
    for j in tqdm(range(m), desc="Learning sparse linear map for each response"):
        y_j = Y_scaled[:, j]

        if use_isis:
            # ISIS pre-screening following SIS R package
            selected_idx = isis_select(
                U_scaled,
                y_j,
                max_iter=isis_max_iter,
                nsis=isis_nsis,
                learning_rate=learning_rate,
                effective_dimension=effective_dimension,
            )
            U_selected = U_scaled[:, selected_idx]

            # Boost on reduced feature set
            coefficients_reduced = boost_linear_regression(
                U_selected,
                y_j,
                learning_rate=learning_rate,
                effective_dimension=effective_dimension,
            )

            # Map back to original indices
            coefficients_j = np.zeros(p)
            coefficients_j[selected_idx] = coefficients_reduced
        else:
            # Standard boosting on all features
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
