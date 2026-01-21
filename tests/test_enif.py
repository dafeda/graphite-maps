import numpy as np
import pytest
import scipy as sp
from graphite_maps.enif import EnIF
from graphite_maps.precision_estimation import precision_to_graph


# Simulate data
def rar1(T, phi, rng=None):
    """simulate auto-regressive-1.
    The first element is simulated from stationary distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.empty([T])
    x[0] = rng.normal(0, 1 / np.sqrt(1 - phi**2))
    for i in range(1, T):
        x[i] = phi * x[i - 1] + rng.normal(0, 1)
    return x


def nrar1(n, p, phi):
    """Samples n realizations of Ar-1(phi) of size p"""
    rng = np.random.default_rng(42)
    U = np.array([rar1(T=p, phi=phi, rng=rng) for _ in range(n)])
    return U


def create_ar1_precision(p, phi):
    Prec_u = sp.sparse.diags(
        [
            np.repeat(-phi, p - 1),
            np.concatenate(([1.0], np.repeat(1.0 + phi**2, p - 2), [1.0])),
            np.repeat(-phi, p - 1),
        ],
        [-1, 0, 1],
        shape=(p, p),
        format="csc",
    )
    return Prec_u


def create_ar1_graph(p):
    # Graph created through precision matrix
    # Could be created directly
    phi = 0.3
    Prec_u = create_ar1_precision(p, phi)
    Graph_u = precision_to_graph(Prec_u)
    return Graph_u


@pytest.mark.parametrize(
    "n, p, phi", [[100, 1000, 0.5], [200, 100, 0.3], [100, 1000, 0.9]]
)
def test_that_posterior_low_level_api_equals_high_level_api(n, p, phi):
    # Sample prior
    U = nrar1(n, p, phi)

    # Create the graph in a very inelegant way (potential for improvement)
    Graph_u = create_ar1_graph(p)

    # Specify observations and associate uncertainty, and a linear map H
    d = np.array([30.0])
    sd_eps = 1
    H = np.array([0] * p, ndmin=2)
    H[0, np.rint(p / 2).astype(int) - 1] = 1  # middle sencor
    H = sp.sparse.csc_matrix(H)
    Prec_eps = np.array([1 / sd_eps**2], ndmin=2)
    Prec_eps = sp.sparse.csc_matrix(Prec_eps)

    # Run the "forward model"
    Y = U @ H.T

    # EnIF high-level API
    gtmap = EnIF(Graph_u=Graph_u, Prec_eps=Prec_eps, H=H)
    gtmap.fit(U, verbose_level=4)
    U_posterior_highlevel = gtmap.transport(U, Y, d, seed=42, verbose_level=10)

    # EnIF low-level API
    gtmap_lowlevel = EnIF(Graph_u=Graph_u, Prec_eps=Prec_eps, H=H)
    gtmap_lowlevel.fit_precision(U)
    if gtmap_lowlevel.H is None:
        gtmap_lowlevel.fit_H(U, U @ H.T)  # simulations Y = U@H.T
    canonical = gtmap_lowlevel.pushforward_to_canonical(U)
    # Work out residuals and associate unexplained variance
    residual = gtmap_lowlevel.response_residual(U, Y)
    eps = gtmap_lowlevel.generate_observation_noise(n, seed=42)
    residual_noisy = residual + eps
    canonical_posterior = gtmap_lowlevel.update_canonical(canonical, residual_noisy, d)
    U_posterior_lowlevel = gtmap_lowlevel.pullback_from_canonical(
        canonical_posterior, U_prior=U
    )

    assert np.allclose(U_posterior_lowlevel, U_posterior_highlevel, atol=1e-6)


@pytest.mark.parametrize(
    "n, p, phi", [[100, 1000, 0.5], [200, 100, 0.3], [100, 1000, 0.9]]
)
def test_that_enif_equals_kalman_under_exact_precision_and_H(n, p, phi):
    # Sample prior
    U = nrar1(n, p, phi)

    # Create the precision
    Prec_u = create_ar1_precision(p, phi)

    # Specify observations and associate uncertainty, and a linear map H
    d = np.array([30.0])
    sd_eps = 1
    H = np.array([0] * p, ndmin=2)
    H[0, np.rint(p / 2).astype(int) - 1] = 1  # middle sencor
    H = sp.sparse.csc_matrix(H)
    Prec_eps = np.array([1 / sd_eps**2], ndmin=2)
    Prec_eps = sp.sparse.csc_matrix(Prec_eps)

    # Run the "forward model"
    Y = U @ H.T

    # EnIF high-level API with known precision
    gtmap = EnIF(Prec_u=Prec_u, Prec_eps=Prec_eps, H=H)
    gtmap.fit(U, verbose_level=4)
    U_posterior_enif = gtmap.transport(U, Y, d, seed=42, verbose_level=10)

    # Create Kalman update -- use same noise
    eps = gtmap.generate_observation_noise(n, seed=42)
    Sigma_u = np.linalg.inv(Prec_u.toarray())
    Sigma_d = H @ Sigma_u @ H.T + np.linalg.inv(Prec_eps.toarray())
    K = Sigma_u @ H.T @ np.linalg.inv(Sigma_d)
    U_posterior_enkf = np.empty_like(U)
    for i in range(n):
        innovation = d - Y[i, 0] - eps[i, 0]  # scalar
        U_posterior_enkf[i] = U[i] + (K @ innovation).ravel()

    assert np.allclose(U_posterior_enif, U_posterior_enkf, atol=1e-12)


@pytest.mark.parametrize(
    "n, p, phi", [[100, 1000, 0.5], [200, 100, 0.3], [100, 1000, 0.9]]
)
def test_that_pullback_of_pushforward_equals_input(n, p, phi):
    # Sample prior
    U = nrar1(n, p, phi)

    # Create the precision
    Prec_u = create_ar1_precision(p, phi)

    # Specify observations and associate uncertainty, and a linear map H
    sd_eps = 1
    H = np.array([0] * p, ndmin=2)
    H[0, np.rint(p / 2).astype(int) - 1] = 1  # middle sencor
    H = sp.sparse.csc_matrix(H)
    Prec_eps = np.array([1 / sd_eps**2], ndmin=2)
    Prec_eps = sp.sparse.csc_matrix(Prec_eps)

    # Notice: No update in canonical space
    gtmap_pullpush = EnIF(Prec_u=Prec_u, Prec_eps=Prec_eps, H=H)
    canonical = gtmap_pullpush.pushforward_to_canonical(U)
    U_posterior = gtmap_pullpush.pullback_from_canonical(canonical)

    # Due to no update, we should have equality
    assert np.allclose(U, U_posterior, atol=1e-12)


@pytest.mark.parametrize("n, p, m", [(500, 50, 10)])
def test_that_partial_fit_H_learns_signal(n, p, m):
    """Test that partial_fit_H learns the true signal features."""
    rng = np.random.default_rng(42)

    # Create simple H: each response depends on a single feature
    H_true_dense = np.zeros((m, p))
    signal_features = list(range(m))  # First m features are signal
    for j in range(m):
        H_true_dense[j, signal_features[j]] = 1.0 + rng.standard_normal() * 0.1
    H_true = sp.sparse.csc_matrix(H_true_dense)

    # Generate data
    U = rng.standard_normal((n, p))
    noise = rng.standard_normal((n, m)) * 0.01  # Low noise
    Y = U @ H_true.T + noise

    # Create EnIF
    Graph_u = create_ar1_graph(p)
    Prec_eps = sp.sparse.eye(m, format="csc")
    gtmap = EnIF(Graph_u=Graph_u, Prec_eps=Prec_eps)

    # Use partial_fit_H with two-pass workflow
    gtmap.init_partial_fit_H(
        p=p,
        m=m,
        alpha=0.0001,
        l1_ratio=0.9,
        random_state=0,
    )

    batch_size = 100

    # Pass 1: Compute global statistics
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        gtmap.partial_fit_scaler_H(U[start:end], Y[start:end])

    # Pass 2: Train the model (multiple epochs)
    n_epochs = 5
    for _ in range(n_epochs):
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            gtmap.partial_fit_H(U[start:end], Y[start:end])

    gtmap.finalize_fit_H()

    # Check that the learned H identifies the correct signal features
    H_learned = gtmap.H.toarray()
    for j in range(m):
        # The signal feature should be among the strongest coefficients
        top_k = 3
        top_features = np.argsort(np.abs(H_learned[j, :]))[-top_k:]
        assert signal_features[j] in top_features, (
            f"Response {j}: expected feature {signal_features[j]} in top-{top_k}, "
            f"got {top_features}"
        )


def test_partial_fit_H_raises_without_init():
    """Test that partial_fit_H raises error if init not called."""
    Graph_u = create_ar1_graph(10)
    Prec_eps = sp.sparse.eye(5, format="csc")
    gtmap = EnIF(Graph_u=Graph_u, Prec_eps=Prec_eps)

    with pytest.raises(RuntimeError, match="Must call init_partial_fit_H"):
        gtmap.partial_fit_H(np.zeros((10, 10)), np.zeros((10, 5)))


def test_finalize_fit_H_raises_without_partial_fit():
    """Test that finalize_fit_H raises error if partial_fit not called."""
    Graph_u = create_ar1_graph(10)
    Prec_eps = sp.sparse.eye(5, format="csc")
    gtmap = EnIF(Graph_u=Graph_u, Prec_eps=Prec_eps)

    gtmap.init_partial_fit_H(p=10, m=5)

    with pytest.raises(RuntimeError, match="Must call partial_fit"):
        gtmap.finalize_fit_H()


def test_two_pass_workflow_with_distribution_shift():
    """Test the two-pass workflow (fit scaler -> fit model) on shifted data.

    This test simulates a scenario where data comes in blocks with vastly
    different statistics (e.g., porosity then permeability). A standard
    online learning approach might fail or perform poorly if the warmup
    period only sees the first block. The two-pass approach ensures correct
    global scaling.

    The test also performs a full EnIF transport to verify the learned H works
    in a complete posterior update workflow.
    """
    rng = np.random.default_rng(42)
    p = 50
    m = 5
    batch_size = 500

    # Create simple H: response j depends on feature j
    H_true = sp.sparse.lil_matrix((m, p))
    for j in range(m):
        H_true[j, j] = 1.0
    H_true = H_true.tocsc()

    # Generate data with EXTREME distribution shift
    # Block 1 (First 500 samples): Mean +100
    U1 = rng.standard_normal((batch_size, p)) + 100.0
    Y1 = U1 @ H_true.T + rng.standard_normal((batch_size, m)) * 0.1

    # Block 2 (Next 500 samples): Mean -100
    U2 = rng.standard_normal((batch_size, p)) - 100.0
    Y2 = U2 @ H_true.T + rng.standard_normal((batch_size, m)) * 0.1

    # If we treated this as a stream, we have two large batches
    batches = [(U1, Y1), (U2, Y2)]

    # Combined data for full EnIF workflow
    U = np.vstack([U1, U2])
    Y = np.vstack([Y1, Y2])

    # Setup EnIF
    Graph_u = create_ar1_graph(p)
    Prec_eps = sp.sparse.eye(m, format="csc")
    gtmap = EnIF(Graph_u=Graph_u, Prec_eps=Prec_eps)

    # Initialize
    gtmap.init_partial_fit_H(
        p=p,
        m=m,
        alpha=0.01,  # Moderate alpha for sparsity while still learning signal
        l1_ratio=0.95,
        random_state=42,
    )

    # --- PASS 1: Global Statistics Calculation ---
    # Iterate over the full dataset just to update the scaler.
    for U_batch, Y_batch in batches:
        gtmap.partial_fit_scaler_H(U_batch, Y_batch)

    # --- PASS 2: Model Training ---
    # Iterate over the full dataset again to train the model.
    # The scaler is automatically frozen using stats from Pass 1.
    # We run multiple epochs to ensure convergence
    n_epochs = 10
    for _ in range(n_epochs):
        for U_batch, Y_batch in batches:
            gtmap.partial_fit_H(U_batch, Y_batch)

    gtmap.finalize_fit_H()

    # Verify learning: The diagonal elements should be dominant
    H_learned = gtmap.H.toarray()
    for j in range(m):
        # Check that the true feature (j) has the largest coefficient
        predicted_feature = np.argmax(np.abs(H_learned[j, :]))
        assert predicted_feature == j, (
            f"Response {j} incorrectly mapped to feature {predicted_feature} "
            f"instead of {j}. Two-pass scaling might have failed."
        )

    # H matrix should exist and have correct shape
    assert gtmap.H is not None, "H matrix should be created"
    assert gtmap.H.shape == (m, p), f"H should have shape ({m}, {p})"

    # --- Full EnIF Run ---
    # Fit precision matrix
    gtmap.fit_precision(U)

    # Define observation
    d = np.zeros(m)
    for j in range(m):
        d[j] = np.mean(Y[:, j])  # Observe mean response

    # Run full transport to posterior
    U_posterior = gtmap.transport(U, Y, d, seed=42)

    # Verify posterior has same shape as prior
    assert U_posterior.shape == U.shape, "Posterior should have same shape as prior"

    # Verify posterior is different from prior (update happened)
    assert not np.allclose(U_posterior, U, atol=1e-6), (
        "Posterior should differ from prior after conditioning"
    )


@pytest.mark.parametrize("n, p, m", [(500, 50, 5)])
def test_online_learning_comparable_to_lasso(n, p, m):
    """Test that online learning produces results comparable to LASSO.

    Both methods should:
    1. Learn the true signal features (identify which features matter)
    2. Produce predictions that correlate with ground truth
    3. Achieve some level of sparsity

    Note: LASSO uses coordinate descent which can drive coefficients exactly to zero,
    while online SGD with L1 penalty (via ElasticNet) only approximates this behavior.
    However, with appropriate sparsity thresholding, online SGD can achieve similar sparsity.
    """
    rng = np.random.default_rng(42)
    batch_size = 100
    n_true_features = 3  # Number of true non-zero features per response

    # Create sparse ground truth H (only n_true_features non-zero entries per response)
    H_true_dense = np.zeros((m, p))
    for j in range(m):
        nonzero_indices = rng.choice(p, size=n_true_features, replace=False)
        H_true_dense[j, nonzero_indices] = rng.standard_normal(n_true_features)
    H_true = sp.sparse.csc_matrix(H_true_dense)

    # Generate synthetic data: Y = U @ H.T + noise
    U = rng.standard_normal((n, p))
    noise = rng.standard_normal((n, m)) * 0.1
    Y = U @ H_true.T + noise

    # --- LASSO approach ---
    Graph_u = create_ar1_graph(p)
    Prec_eps = sp.sparse.eye(m, format="csc")
    gtmap_lasso = EnIF(Graph_u=Graph_u, Prec_eps=Prec_eps)
    gtmap_lasso.fit_H(U, Y, learning_algorithm="LASSO")
    H_lasso = gtmap_lasso.H

    # --- Influence-boost approach ---
    gtmap_boost = EnIF(Graph_u=Graph_u, Prec_eps=Prec_eps)
    gtmap_boost.fit_H(U, Y, learning_algorithm="influence-boost")
    H_boost = gtmap_boost.H

    # --- Online learning approach ---
    gtmap_online = EnIF(Graph_u=Graph_u, Prec_eps=Prec_eps)
    gtmap_online.init_partial_fit_H(
        p=p,
        m=m,
        alpha=0.01,  # Higher alpha for sparsity
        l1_ratio=0.95,
        random_state=42,
        sparsity_threshold=0.01,
    )

    # Pass 1: Compute global statistics
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        gtmap_online.partial_fit_scaler_H(U[start:end], Y[start:end])

    # Pass 2: Train the model (multiple epochs for convergence)
    n_epochs = 10
    for _ in range(n_epochs):
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            gtmap_online.partial_fit_H(U[start:end], Y[start:end])

    gtmap_online.finalize_fit_H()
    H_online = gtmap_online.H

    # All should have same shape
    assert H_lasso.shape == H_online.shape == H_boost.shape == (m, p)

    # Compare sparsity levels
    lasso_nnz = H_lasso.nnz
    online_nnz = H_online.nnz
    boost_nnz = H_boost.nnz
    true_nnz = m * n_true_features  # Ground truth has exactly this many non-zeros
    total_elements = m * p

    lasso_sparsity = 1.0 - (lasso_nnz / total_elements)
    online_sparsity = 1.0 - (online_nnz / total_elements)
    boost_sparsity = 1.0 - (boost_nnz / total_elements)
    true_sparsity = 1.0 - (true_nnz / total_elements)

    # Print sparsity comparison for visibility
    print("\nSparsity comparison:")
    print(f"  Ground truth: {true_sparsity:.1%} sparse ({true_nnz} non-zeros)")
    print(f"  LASSO:        {lasso_sparsity:.1%} sparse ({lasso_nnz} non-zeros)")
    print(f"  Online SGD:   {online_sparsity:.1%} sparse ({online_nnz} non-zeros)")
    print(f"  Influence-boost: {boost_sparsity:.1%} sparse ({boost_nnz} non-zeros)")
    if lasso_nnz < online_nnz:
        print(f"  LASSO has {online_nnz - lasso_nnz} fewer non-zeros than online")
    else:
        print(f"  Online has {lasso_nnz - online_nnz} fewer non-zeros than LASSO")

    # LASSO should be reasonably sparse
    assert lasso_sparsity > 0.3, f"LASSO H should be sparse, got {lasso_sparsity:.2%}"
    # Online learning may be less sparse due to SGD optimization path
    assert online_sparsity > 0.0, (
        f"Online H should have some sparsity, got {online_sparsity:.2%}"
    )
    # Influence-boost should also exhibit sparsity
    assert boost_sparsity > 0.3, (
        f"Influence-boost H should have some sparsity, got {boost_sparsity:.2%}"
    )

    # Both should identify the true signal features
    # For each response, the top-k features by coefficient magnitude should overlap with truth
    for j in range(m):
        true_nonzero = set(np.where(np.abs(H_true_dense[j, :]) > 0)[0])

        # Get top-k features by absolute coefficient (k = number of true features)
        k = len(true_nonzero)
        lasso_top_k = set(np.argsort(np.abs(H_lasso[j, :].toarray().ravel()))[-k:])
        online_top_k = set(np.argsort(np.abs(H_online[j, :].toarray().ravel()))[-k:])
        boost_top_k = set(np.argsort(np.abs(H_boost[j, :].toarray().ravel()))[-k:])

        # Both methods should identify at least 2 of 3 true features
        lasso_overlap = len(lasso_top_k & true_nonzero)
        online_overlap = len(online_top_k & true_nonzero)
        boost_overlap = len(boost_top_k & true_nonzero)

        assert lasso_overlap >= 2, (
            f"LASSO should identify at least 2/{k} true features for response {j}, "
            f"got {lasso_overlap}"
        )
        assert online_overlap >= 2, (
            f"Online learning should identify at least 2/{k} true features for response {j}, "
            f"got {online_overlap}"
        )
        assert boost_overlap >= 2, (
            f"Influence-boost should identify at least 2/{k} true features for response {j}, "
            f"got {boost_overlap}"
        )

    # Both methods should produce predictions that correlate well with ground truth
    Y_true = U @ H_true.T
    Y_pred_lasso = U @ H_lasso.T
    Y_pred_online = U @ H_online.T
    Y_pred_boost = U @ H_boost.T

    for j in range(m):
        # Correlation with ground truth (how well did we learn the true relationship?)
        corr_lasso_true = np.corrcoef(Y_pred_lasso[:, j], Y_true[:, j])[0, 1]
        corr_online_true = np.corrcoef(Y_pred_online[:, j], Y_true[:, j])[0, 1]
        corr_boost_true = np.corrcoef(Y_pred_boost[:, j], Y_true[:, j])[0, 1]

        assert corr_lasso_true > 0.99, (
            f"LASSO predictions should correlate with truth for response {j}, "
            f"got r={corr_lasso_true:.3f}"
        )
        assert corr_online_true > 0.99, (
            f"Online predictions should correlate with truth for response {j}, "
            f"got r={corr_online_true:.3f}"
        )
        assert corr_boost_true > 0.99, (
            f"Influence-boost predictions should correlate with truth for response {j}, "
            f"got r={corr_boost_true:.3f}"
        )

        # The two methods should also agree with each other
        corr_methods_lasso_online = np.corrcoef(
            Y_pred_lasso[:, j], Y_pred_online[:, j]
        )[0, 1]
        corr_methods_lasso_boost = np.corrcoef(Y_pred_lasso[:, j], Y_pred_boost[:, j])[
            0, 1
        ]
        corr_methods_online_boost = np.corrcoef(
            Y_pred_online[:, j], Y_pred_boost[:, j]
        )[0, 1]

        assert corr_methods_lasso_online > 0.99, (
            f"LASSO and online predictions should correlate for response {j}, "
            f"got r={corr_methods_lasso_online:.3f}"
        )
        assert corr_methods_lasso_boost > 0.99, (
            f"LASSO and influence-boost predictions should correlate for response {j}, "
            f"got r={corr_methods_lasso_boost:.3f}"
        )
        assert corr_methods_online_boost > 0.99, (
            f"Online and influence-boost predictions should correlate for response {j}, "
            f"got r={corr_methods_online_boost:.3f}"
        )
