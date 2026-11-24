"""Tests for angle finder utils."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from guppyalgos.utils import (
    ChebyshevPolynomial,
    CompilerPhasesNumba,
    FourierPolynomial,
    FunctionParity,
    GQSPAngleFinder,
    QSPAngleFinder,
)

coeffs_dict: dict[int, NDArray[np.float64]] = {
    4: np.array([0.01037415, 0.50187459, 0.49577329, 0.13382953, 0.14211109]),
    6: np.array(
        [
            0.52108471,
            0.94551742,
            0.64954046,
            0.86083192,
            0.97084601,
            0.18554169,
            2,
        ]
    ),
    8: np.array(
        [
            0.87340084,
            0.90627728,
            0.75827799,
            0.34690195,
            0.46353471,
            0.61053774,
            0.49628172,
            0.47797401,
            0.43555546,
        ]
    ),
    10: np.array(
        [
            0.75803602,
            0.26227322,
            0.10085859,
            0.91719271,
            0.51746024,
            0.13353968,
            0.11254311,
            0.84164115,
            0.54607474,
            0.26101658,
            0.57582552,
        ]
    ),
    12: np.array(
        [
            0.31385261,
            0.79162319,
            0.49309072,
            0.96515818,
            0.52031194,
            0.9030216,
            0.24980456,
            0.5214873,
            0.50574568,
            0.81307148,
            0.07319422,
            0.836207,
            0.80423564,
        ]
    ),
    5: np.array([0.47612445, 0.95251115, 0.85325763, 0.89850711, 0.0255691, 0.1564229]),
    7: np.array(
        [
            0.47689815,
            0.73472184,
            0.35630948,
            0.56445363,
            0.0454695,
            0.64044131,
            0.68120638,
            0.39563748,
        ]
    ),
    9: np.array(
        [
            0.85796041,
            0.36904,
            0.89899188,
            0.28909783,
            0.38032348,
            0.10937339,
            0.81462461,
            0.93348544,
            0.18310719,
            0.31119857,
        ]
    ),
    11: np.array(
        [
            0.67470404,
            0.07910104,
            0.03272399,
            0.32657562,
            0.38661219,
            0.54896735,
            0.55293117,
            0.33012717,
            0.4256511,
            0.25185686,
            0.20263513,
            0.77752551,
        ]
    ),
    13: np.array(
        [
            0.13959285,
            0.45120371,
            0.12252262,
            0.8162859,
            0.84348993,
            0.11477506,
            0.72098171,
            0.92331548,
            0.38603093,
            0.77711281,
            0.9238857,
            0.00495141,
            0.27258381,
            0.03651912,
        ]
    ),
}


def get_chebyshev_target_polynomial(
    d_cheb: int,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Generate the Chebyshev target function."""
    coeffs = coeffs_dict[d_cheb]
    # coeffs = np.random.random(d_cheb + 1)
    x_vals = np.linspace(-1, 1, 100)
    f = np.polynomial.chebyshev.chebval(x_vals, coeffs)
    f_max = np.max(f)
    f_min = np.min(f)

    def target_polynomial(x: NDArray[np.float64]) -> NDArray[Any]:
        """Compute a normalized Chebyshev polynomial."""
        return (
            (np.sign(x)) ** d_cheb
            * (np.polynomial.chebyshev.chebval(np.abs(x), coeffs) - f_min)
            / (f_max - f_min)
        )

    return target_polynomial


def get_normalized_fourier_coeffs(d_max: int, d_min: int) -> dict[int, np.complex128]:
    """Auxiliary function that generates a random Fourier function s.t. |f(x)| <= 1."""
    f_coeffs = {
        k: np.complex128(np.random.random() + 1j * np.random.random())
        for k in range(d_min, d_max + 1)
    }

    def eval_fourier(x: NDArray[np.float64]) -> NDArray[np.complex128]:
        powers = np.arange(d_min, d_max + 1)
        s = np.zeros_like(x, dtype=np.complex128)
        for n in powers:
            s += f_coeffs[n] * np.exp(1j * n * np.pi * x)
        return s

    # normalize
    x_vals = np.linspace(-1, 1, 2000)
    norm = 2 * np.max(
        np.abs(eval_fourier(x_vals))
    )  # we multiply by 2 to make sure |f(x)| <= 1

    f_coeffs_norm = {k: f_coeffs[k] / norm for k in f_coeffs}
    return f_coeffs_norm


def apply_gqsp_single(
    phase_factors: NDArray[np.float64], x: np.float64, d_max: int, d_min: int = 0
) -> np.complex128:
    """Evaluate GQSP on one variable."""
    theta_0, phi_0, lam = phase_factors[0]

    A = np.array(
        [[np.exp(1j * np.pi * x), 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )
    Ap = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.exp(-1j * np.pi * x), 0], [0, 0, 0, 1]]
    )

    u = np.kron(GQSPAngleFinder._arbitrary_su2(theta_0, phi_0, lam), np.identity(2))
    for d in range(1, d_max + 1):
        theta_d, phi_d, _ = phase_factors[d]
        u = (
            np.kron(GQSPAngleFinder._arbitrary_su2(theta_d, phi_d, 0), np.identity(2))
            @ A
            @ u
        )

    for d in range(d_max + 1, d_max - d_min + 1):
        theta_d, phi_d, _ = phase_factors[d]
        u = (
            np.kron(GQSPAngleFinder._arbitrary_su2(theta_d, phi_d, 0), np.identity(2))
            @ Ap
            @ u
        )
    return u[0, 0]


@pytest.mark.parametrize("d_phi", [4, 5, 6, 7, 8, 9, 10])
def test_chebyshev_unnormalized(d_phi: int) -> None:
    """Test that unnormalized functions fail for ChebyshevPolynomial."""
    target_polynomial = get_chebyshev_target_polynomial(d_phi)

    def unnormalized_poly(x: NDArray[np.float64]) -> NDArray[Any]:
        return 2 * target_polynomial(x)

    with pytest.raises(AssertionError, match=r"Input function is not normalized."):
        ChebyshevPolynomial(unnormalized_poly, d_phi)


@pytest.mark.parametrize("d_phi", [4, 5, 6, 7, 8, 9, 10])
def test_chebyshev__parity_and_degree(d_phi: int) -> None:
    """Test that the degree corresponds to the parity for ChebyshevPolynomial."""
    target_polynomial = get_chebyshev_target_polynomial(d_phi)
    chebyshev_polynomial = ChebyshevPolynomial(target_polynomial, d_phi)
    if d_phi % 2 == 0:
        assert chebyshev_polynomial.parity == FunctionParity.EVEN
        with pytest.raises(
            AssertionError, match=r"Even function requires even degree."
        ):
            ChebyshevPolynomial(target_polynomial, d_phi + 1)
    else:
        assert chebyshev_polynomial.parity == FunctionParity.ODD
        with pytest.raises(AssertionError, match=r"Odd function requires odd degree."):
            ChebyshevPolynomial(target_polynomial, d_phi + 1)


@pytest.mark.parametrize("d_phi", [4, 6, 8])
def test_chebyshev_no_parity(d_phi: int) -> None:
    """Test that functions without parity fail for ChebyshevPolynomial."""
    target_polynomial_even = get_chebyshev_target_polynomial(d_phi)
    target_polynomial_odd = get_chebyshev_target_polynomial(d_phi + 1)

    def target_polynomial_noparity(x: NDArray[np.float64]) -> NDArray[Any]:
        return (target_polynomial_even(x) + target_polynomial_odd(x)) / 2

    with pytest.raises(
        AssertionError, match=r"Input function does not have defined parity."
    ):
        ChebyshevPolynomial(target_polynomial_noparity, d_phi)


@pytest.mark.parametrize("d_phi", [4, 6, 8, 10, 12])
def test_qsp_angles_even(d_phi: int) -> None:
    """Test that the angles optimized obtain a good approximation for even functions."""
    target_polynomial = get_chebyshev_target_polynomial(d_phi)

    chebyshev_polynomial = ChebyshevPolynomial(target_polynomial, d_phi)
    qsp_optimizer = QSPAngleFinder(d_phi=d_phi, target_polynomial=chebyshev_polynomial)
    x_vals = np.linspace(-1, 1, 100)
    f_vals_qsp = qsp_optimizer(x=x_vals)
    f_vals_target = target_polynomial(x_vals)

    # against the target polynomial
    np.testing.assert_allclose(f_vals_qsp, f_vals_target, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("d_phi", [5, 7, 9, 11, 13])
def test_qsp_angles_odd(d_phi: int) -> None:
    """Test that the angles optimized obtain a good approximation for odd functions."""
    target_polynomial = get_chebyshev_target_polynomial(d_phi)

    chebyshev_polynomial = ChebyshevPolynomial(target_polynomial, d_phi)
    qsp_optimizer = QSPAngleFinder(d_phi=d_phi, target_polynomial=chebyshev_polynomial)
    x_vals = np.linspace(-1, 1, 100)
    f_vals_qsp = qsp_optimizer(x=x_vals)
    f_vals_target = qsp_optimizer.target_polynomial(x_vals)

    # against the target polynomial
    np.testing.assert_allclose(f_vals_qsp, f_vals_target, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("d_phi", [4, 5, 10, 13])
def test_qsp_angles_numba(d_phi: int) -> None:
    """Test that the angles obtained with numba are the same."""
    d_cheb = d_phi

    target_polynomial = get_chebyshev_target_polynomial(d_cheb)

    chebyshev_polynomial = ChebyshevPolynomial(target_polynomial, d_cheb)
    qsp_optimizer = QSPAngleFinder(d_phi=d_phi, target_polynomial=chebyshev_polynomial)
    qsp_optimizer_nb = QSPAngleFinder(
        d_phi=d_phi,
        target_polynomial=chebyshev_polynomial,
        compiler=CompilerPhasesNumba(),
    )
    np.testing.assert_allclose(qsp_optimizer.phi, qsp_optimizer_nb.phi)


def test_qsp_compiler() -> None:
    """Test the compiler is correct."""
    d_phi = 10
    d_cheb = d_phi
    target_polynomial = get_chebyshev_target_polynomial(d_cheb)
    chebyshev_polynomial = ChebyshevPolynomial(target_polynomial, d_cheb)
    qsp_optimizer = QSPAngleFinder(d_phi=d_phi, target_polynomial=chebyshev_polynomial)
    qsp_optimizer_nb = QSPAngleFinder(
        d_phi=d_phi,
        target_polynomial=chebyshev_polynomial,
        compiler=CompilerPhasesNumba(),
    )

    assert qsp_optimizer.compiler.__class__.__name__ == "CompilerPhasesNumpy"
    assert qsp_optimizer_nb.compiler.__class__.__name__ == "CompilerPhasesNumba"


@pytest.mark.parametrize("degree", [8, 16, 20, 26])
def test_unitary_complementary_coeffs(degree: int) -> None:
    """Test that the obtained polynomial satisfies the unitary condition for GQSP."""
    np.random.seed(0)
    d_max = degree // 2
    d_min = -degree // 2

    f_coeffs_norm = get_normalized_fourier_coeffs(d_max, d_min)
    f = FourierPolynomial(f_coeffs_norm, d_max, d_min)
    angle_finder = GQSPAngleFinder(f)
    q = angle_finder.complementary_polynomial

    x_vals = np.linspace(-1, 1, 200)
    res = np.square(np.abs(f(x_vals))) + np.square(np.abs(q(x_vals)))
    np.testing.assert_allclose(res, np.ones(len(res)), rtol=1e-07, atol=1e-07)


@pytest.mark.parametrize("degree", [8, 16, 20, 26])
def test_fourier_unnormalized(degree: int) -> None:
    """Test that unnormalized coeffs fail for FourierPolynomial."""
    np.random.seed(0)
    d_max = degree // 2
    d_min = -degree // 2
    f_coeffs_norm = get_normalized_fourier_coeffs(d_max, d_min)

    f_coeffs_unnorm = {n: f_coeffs_norm[n] * 10 for n in f_coeffs_norm}

    with pytest.raises(AssertionError, match=r"Input coeffs are not normalized."):
        FourierPolynomial(f_coeffs_unnorm, d_max, d_min)


@pytest.mark.parametrize("degree", [8, 16, 20, 26])
def test_phase_factors(degree: int) -> None:
    """Test that the obtained phase factors approximate target function for GQSP."""
    np.random.seed(0)
    d_max = degree // 2
    d_min = -degree // 2
    f_coeffs_norm = get_normalized_fourier_coeffs(d_max, d_min)
    f = FourierPolynomial(f_coeffs_norm, d_max, d_min)

    angle_finder = GQSPAngleFinder(f)
    phase_factors = angle_finder.phase_factors

    x_vals = np.linspace(-1, 1, 20)
    y_vals_gqsp = [apply_gqsp_single(phase_factors, x, d_max, d_min) for x in x_vals]
    y_vals_target = f(x_vals)

    np.testing.assert_allclose(y_vals_gqsp, y_vals_target, rtol=1e-07, atol=1e-07)
