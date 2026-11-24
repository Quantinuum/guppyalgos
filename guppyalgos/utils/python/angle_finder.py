"""Utils functions to compute phases for QSP and GQSP."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import Any

import numpy as np
from numpy import complexfloating
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyfromroots
from numpy.typing import NDArray
from scipy.fft import dct
from scipy.linalg import expm
from scipy.optimize import (
    OptimizeResult,
    minimize,
)

from guppyalgos.errors.import_errors import NumbaImportError


class FunctionParity(IntEnum):
    """Parity of function."""

    EVEN = 0
    ODD = 1
    NONE = 2


class ChebyshevPolynomial:
    """Chebyshev class.

    Given an input vectorized function computes the coefficients of the Chebyshev
    polynomial with the desired degree that approximates it. The input function must
    be normalized, i.e. it's maximum (in absolute value) in [-1, 1] has to be 1 at most.
    Furthermore, even (odd) parity functions require an even (odd) degree.

    Args:
        fun (Callable[[NDArray], NDArray]): vectorized function to be approximated
        degree (int): degree of the polynomial
        rtol (float): tolerance for checking normalization and parity

    """

    def __init__(
        self,
        fun: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        degree: int,
        rtol: float = 1e-3,
    ):
        """Initialize Chebyshev object."""
        self._fun = fun

        # Check normalization and parity
        data_points = np.linspace(-1, 1, 10000)
        self._norm = max(abs(self._fun(data_points)))
        assert self._norm <= 1 + rtol, "Input function is not normalized."

        if np.allclose(self._fun(data_points), self._fun(-data_points), rtol=rtol):
            self._parity = FunctionParity.EVEN
            assert degree % 2 == 0, "Even function requires even degree."
        elif np.allclose(self._fun(data_points), -self._fun(-data_points), rtol=rtol):
            self._parity = FunctionParity.ODD
            assert degree % 2 == 1, "Odd function requires odd degree."
        else:
            self._parity = FunctionParity.NONE
        assert self._parity != FunctionParity.NONE, (
            "Input function does not have defined parity."
        )

        self._degree = degree
        self._roots = np.cos(
            (np.arange(self._degree + 1) + 0.5) * np.pi / (self._degree + 1)
        )
        self._extrema = np.cos(np.arange(self._degree + 1) * np.pi / (self._degree + 1))
        self._coeffs = self._compute_coeffs()
        # self._norm = max(abs(self(extrema)) for extrema in self._extrema)
        # self._coeffs = self._coeffs / self._norm
        self._error = self._compute_error(data_points)

    @property
    def coeffs(self) -> NDArray[np.float64]:
        """Return the coefficients."""
        return self._coeffs

    @property
    def norm(self) -> np.float64:
        """Return the norm."""
        return self._norm

    @property
    def degree(self) -> int:
        """Return the degree."""
        return self._degree

    @property
    def parity(self) -> FunctionParity:
        """Return the parity of the function."""
        return self._parity

    @property
    def roots(self) -> NDArray[np.float64]:
        """Return the roots."""
        return self._roots

    @property
    def extrema(self) -> NDArray[np.float64]:
        """Return the extrema."""
        return self._extrema

    @property
    def fun(self) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """Return the original function."""
        return self._fun

    @property
    def error(self) -> np.float64:
        """Return the maximum absolute error of the polynomial approximation."""
        return self._error

    def _compute_coeffs(self) -> NDArray[np.float64]:
        """Compute coefficients of Chebyshev polynomial approximation.

        Returns:
            NDArray[np.float64]: coefficients

        """
        y = self._fun(self._roots)
        coeffs = dct(y) / (self._degree + 1)
        coeffs[0] = coeffs[0] / 2
        return coeffs

    def _compute_error(self, x: NDArray[np.float64]) -> np.float64:
        """Compute the maximum absolute error of the polynomial approximation."""
        return max(np.abs(self._fun(x) - self(x)))

    def __call__(
        self, x: NDArray[np.float64] | NDArray[np.complex128]
    ) -> NDArray[np.complex128]:
        """Evaluate Chebyshev polynomial.

        Args:
            x (NDArray[np.complex128]): x values where to evaluate the polynomial

        """
        return np.polynomial.chebyshev.chebval(x, self._coeffs)


class BaseCompilePhases(ABC):
    """Abstract class for Phase Compilers for QSPAngleFinder."""

    @abstractmethod
    def _construct_loss_function(
        self,
        x: NDArray[np.float64],
        fun_vals: NDArray[np.complex128] | NDArray[np.float64],
        d_phi: int,
    ) -> Callable[[NDArray[np.float64]], np.float64]:
        """Construct loss function for the QSP angle optimization accelerated with njit.

        Args:
            x (NDArray[np.float64]): x positions where the function is evaluated.
            fun_vals (NDArray[np.complex128] | NDArray[np.float64]): Values of the
                target function.
            d_phi (int): Degree of the QSP.

        Returns:
            Callable[[NDArray[np.float64]], np.float64]: Loss function.

        """
        pass

    @staticmethod
    def _f_phi(
        phi: Sequence[float] | NDArray[np.float64], x: NDArray[np.float64]
    ) -> NDArray[Any] | NDArray[complexfloating[Any, Any]]:
        """Compute the QSP unitary given x and the phases.

        Args:
            phi (list[np.float64] | NDArray[np.float64]): Standard phases.
            x (NDArray[np.float64]): x positions where to evaluate the function.

        Returns:
            NDArray[Any] | NDArray[complexfloating[Any, Any]]: Values of the function.

        """
        # Construct U_phi
        W_x = np.array([[x, 1j * np.sqrt(1 - x**2)], [1j * np.sqrt(1 - x**2), x]])
        U_phi = np.array([[np.exp(1j * phi[0]), 0], [0, np.exp(-1j * phi[0])]])
        for p in phi[1:]:
            U_phi = U_phi @ W_x @ np.array([[np.exp(1j * p), 0], [0, np.exp(-1j * p)]])
        return np.real(U_phi[0, 0])


class CompilerPhasesNumpy(BaseCompilePhases):
    """Compiler based on numpy operations."""

    def __init__(self) -> None:
        """Initialize CompilerPhasesNumpy."""
        pass

    def _construct_loss_function(
        self,
        x: NDArray[np.float64],
        fun_vals: NDArray[np.complex128] | NDArray[np.float64],
        d_phi: int,
    ) -> Callable[[NDArray[np.float64]], np.float64]:
        def loss_function(
            phi_hat: list[np.float64] | NDArray[np.float64],
        ) -> np.float64:
            if d_phi % 2 == 0:
                phi = np.concatenate([phi_hat, phi_hat[-2::-1]])
            else:
                phi = np.concatenate([phi_hat, phi_hat[::-1]])
            return np.sum(
                [
                    np.abs(self._f_phi(phi, x[i]) - fun_vals[i]) ** 2
                    for i in range(len(x))
                ]
            )

        return loss_function


class CompilerPhasesNumba(BaseCompilePhases):
    """Compiler based on numba operations."""

    def __init__(self) -> None:
        """Initialize CompilerPhasesNumba."""
        try:
            from numba import njit
        except ImportError as e:
            raise NumbaImportError("accelerated QSP angle optimization") from e

        self._f_phi_nb = njit(self._f_phi)

    def _construct_loss_function(
        self,
        x: NDArray[np.float64],
        fun_vals: NDArray[np.complex128] | NDArray[np.float64],
        d_phi: int,
    ) -> Callable[[NDArray[np.float64]], np.float64]:
        from numba import njit

        f_phi_nb = self._f_phi_nb

        def loss_function(
            phi_hat: list[np.float64] | NDArray[np.float64],
        ) -> np.float64:
            phi = np.zeros(d_phi + 1, dtype=np.float64)
            if (d_phi + 1) % 2 == 0:
                phi[0 : len(phi_hat)] = phi_hat
                phi[len(phi_hat) :] = phi_hat[::-1]
            else:
                phi[0 : len(phi_hat)] = phi_hat
                phi[len(phi_hat) :] = phi_hat[-2::-1]

            total = np.abs(f_phi_nb(phi, x[0]) - fun_vals[0]) ** 2
            for i in range(1, len(x)):
                total += np.abs(f_phi_nb(phi, x[i]) - fun_vals[i]) ** 2
            return total

        return njit(loss_function)


class QSPAngleFinder:
    """Find QSP angles via optimization.

    Based on https://arxiv.org/abs/2002.11649. Given a target polynomial ``f(x)``,
    this protocol minimizes the loss function
    ``L(phi) = dist(Re(<0|U_phi(x)|0>), f(x))``. It uses phase symmetry to reduce
    the complexity of the problem. For degree ``d``, there are ``d + 1`` standard
    phases, but only ``d_hat = ceil((d + 1) / 2)`` independent symmetric phases.
    If ``phi_hat = (phi_hat_0, ..., phi_hat_{d_hat - 1})``, the mappings are:

    * For odd ``d``: ``phi = (phi_hat_0, ..., phi_hat_{d_hat - 1},
      phi_hat_{d_hat - 1}, ..., phi_hat_0)``.
    * For even ``d``: ``phi = (phi_hat_0, ..., phi_hat_{d_hat - 2},
      phi_hat_{d_hat - 1}, phi_hat_{d_hat - 2}, ..., phi_hat_0)``.

    There is also the option to use numba to accelerate the evaluation of the loss
    function.

    Args:
        d_phi (int): Degree of the QSP.
        target_polynomial (ChebyshevPolynomial):
            Target Chebyshev polynomial.
        phi_0 (NDArray[np.float64]):  Set of initial phases.
        compiler (BaseCompilePhases): Compiler used for generating the loss function.
            Availables: 'numpy' and 'numba'. Defaults to numpy.

    """

    def __init__(
        self,
        d_phi: int,
        target_polynomial: ChebyshevPolynomial,
        phi_0: NDArray[np.float64] | None = None,
        compiler: BaseCompilePhases = CompilerPhasesNumpy(),  # noqa: B008
    ):
        """Initialize QSPAngleFinder."""
        self._norm = target_polynomial.norm
        self._d_phi = d_phi
        self._target_polynomial = target_polynomial

        self._compiler = compiler

        self._phi_0 = phi_0 if phi_0 is not None else self._get_phi_0()
        self._phi_hat_0 = self._convert_phi_to_phi_hat(self._phi_0)

        self._d_tilde = int(np.ceil((self._d_phi + 1) / 2))
        self._x_Chebyshev_roots = self._get_x_Chebyshev_roots()
        self._fun_vals = self._target_polynomial(self._x_Chebyshev_roots)

        self._loss = self._compiler._construct_loss_function(
            self._x_Chebyshev_roots, self._fun_vals, self._d_phi
        )
        self._res = self._minimize()

        self._opt_error = self._res.fun
        self._phi_hat = self._res.x
        self._phi = self._convert_phi_hat_to_phi(self._phi_hat)

    @property
    def phi(self) -> Sequence[float]:
        """Return the phases."""
        return qsp_phase_reflection(self._phi)

    @property
    def target_polynomial(
        self,
    ) -> ChebyshevPolynomial:
        """Return the target polynomial."""
        return self._target_polynomial

    @property
    def x_Chebyshev_roots(self) -> NDArray[np.float64]:
        """Return the roots."""
        return self._x_Chebyshev_roots

    @property
    def fun_vals(self) -> NDArray[np.complex128] | NDArray[np.float64]:
        """Return the value of the target function."""
        return self._fun_vals

    @property
    def norm(self) -> np.float64:
        """Return the value of the target function."""
        return self._norm

    @property
    def compiler(self) -> BaseCompilePhases:
        """Return the compiler."""
        return self._compiler

    @property
    def opt_error(self) -> np.float64:
        """Return the optimization error."""
        return self._opt_error

    def _get_phi_0(self) -> NDArray[np.float64]:
        """Compute initial value for phi."""
        phi_0 = np.zeros(self._d_phi + 1)
        phi_0[0] = np.pi / 4
        phi_0[-1] = np.pi / 4
        return phi_0

    def _get_x_Chebyshev_roots(self) -> NDArray[np.float64]:
        """Compute the Chebyshev roots."""
        return np.array(
            [
                np.cos((2 * i - 1) * np.pi / 4 / self._d_tilde)
                for i in range(1, self._d_tilde + 1)
            ]
        )

    def _minimize(self) -> OptimizeResult:
        """Minimization process using scipy.

        Returns:
            OptimizeResult: result of scipy minimize

        """
        bounds = [(-np.pi, np.pi) for _ in range(self._d_tilde)]

        return minimize(
            self._loss,
            self._phi_hat_0,
            # tol = 1E-12,
            # ftol = 1E-12,
            bounds=bounds,
            method="L-BFGS-B",
            # options={"ftol": 1E-12, "gtol": 1E-12}
        )

    def __call__(self, x: NDArray[np.float64]) -> NDArray[Any]:
        """Call optimizer and evaluate the approximation function.

        Args:
            x (NDArray[np.float64]): x positions where to evaluate the function.

        Returns:
            NDArray[np.float64]: Values of the approximation function.

        """
        return np.array([self._compiler._f_phi(self._phi, x_i) for x_i in x])

    def _convert_phi_hat_to_phi(
        self, phi_hat: list[np.float64] | NDArray[np.float64]
    ) -> Sequence[float]:
        """Convert phi_hat format to standard phi format.

        Args:
            phi_hat (list[np.float64] | NDArray[np.float64]): Symmetric phases.

        Returns:
            list[np.float64] | NDArray[np.float64]: Standard phases.

        """
        if self._d_phi % 2 == 0:
            phi = np.concatenate([phi_hat, phi_hat[-2::-1]])
        else:
            phi = np.concatenate([phi_hat, phi_hat[::-1]])
        return list(phi)

    @staticmethod
    def _convert_phi_to_phi_hat(
        phi: list[np.float64] | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Convert standard phi format to phi_hat format.

        Args:
            phi (list[np.float64] | NDArray[np.float64]): Standard phases.

        Returns:
            list[np.float64] | NDArray[np.float64]: Symmetric phases.

        """
        d_tilde = int(np.ceil((len(phi)) / 2))
        phi_hat = phi[:d_tilde]
        return np.array(phi_hat)


def qsp_phase_reflection(phi_list: Sequence[float]) -> Sequence[float]:
    """QSP reflection phase convention conversion from pyqsp.

    Converts a list of qsp phases to radians for use in guppy
    in the convention used in Appendix A2 of
    https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.040203

    Args:
        phi_list (list[float]): The list of phases to be converted

    Returns:
        np.array: The converted phases

    """
    phi_array = np.array(phi_list)
    d = len(phi_list) - 1
    phi_1 = phi_array[0] + phi_array[-1] + (d - 1) * (np.pi / 2)
    phi_2_d = phi_array[1:-1] - np.pi / 2
    new_phi = [phi_1, *phi_2_d]
    phi_list_rev = new_phi[::-1]

    return (-2 * np.array(phi_list_rev) / np.pi).tolist()


class FourierPolynomial:
    r"""Fourier class representing a Fourier Polynomial.

    Given coefficients :math:`\{n: c_n\}` for
    :math:`d_{\min} \leq n \leq d_{\max}`, this class represents
    :math:`f(x) = \sum_n c_n \exp(i n \pi x)`. The coefficients must define a
    normalized function.

    Args:
        coeffs (dict): dictionary of coefficients.
        d_max (Optional[int]): maximum degree.
        d_min (int): minimum degree.
        rtol (float): tolerance for checking normalization.

    """

    def __init__(
        self,
        coeffs: dict[int, np.complex128],
        d_max: int | None = None,
        d_min: int = 0,
        rtol: float = 1e-3,
    ):
        """Initialize Fourier class."""
        self._coeffs = coeffs
        self._d_max = len(coeffs) - 1 if d_max is None else d_max
        self._d_min = d_min

        # Check normalization
        data_points = np.linspace(-1, 1, 10000)
        self._norm = max(abs(self(data_points)))
        assert self._norm <= 1 + rtol, "Input coeffs are not normalized."

    @property
    def coeffs(self) -> dict[int, np.complex128]:
        """Return dict of coefficients."""
        return self._coeffs

    @property
    def coeffs_list(self) -> NDArray[np.complex128]:
        """Return list of coefficients."""
        return np.array(list(self._coeffs.values()))

    @property
    def d_max(self) -> int:
        """Return maximum degree."""
        return self._d_max

    @property
    def d_min(self) -> int:
        """Return minimum degree."""
        return self._d_min

    @property
    def norm(self) -> float:
        """Return norm of function."""
        return self._norm

    def eval_mat(self, mat: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Given matrix mat, evaluates mat * pi."""
        powers = np.arange(self._d_min, self._d_max + 1)
        s = np.zeros_like(mat, dtype=np.complex128)
        for n in powers:
            s += self._coeffs[n] * expm(1j * n * np.pi * mat)
        return s

    def __call__(
        self, x: NDArray[np.float64], matrix: bool = False
    ) -> NDArray[np.complex128]:
        """Evaluate Fourier polynomial.

        Given a 1 or 2 dimensional array x, evaluates x * pi.

        Args:
            x (NDArray[np.complex128]): x values where to evaluate the polynomial.
            matrix (bool): If True, uses scipy expm to compute the exponential of the
                matrix.

        """
        exp_fun = np.exp
        if matrix:
            exp_fun = expm

        powers = np.arange(self._d_min, self._d_max + 1)
        s = np.zeros_like(x, dtype=np.complex128)
        for n in powers:
            s += self._coeffs[n] * exp_fun(1j * n * np.pi * x)
        return s


class GQSPAngleFinder:
    r"""Find GQSP angles via exact methods.

    Given a target Fourier series P, this class computes the phase factors to implement
    the GQSP circuit for a given operator. It consists of two steps:

    * Find the complementary series :math:`Q` such that
      :math:`|P(x)|^2 + |Q(x)|^2 = 1`, following Proof of Lemma 4 in
      https://arxiv.org/pdf/2206.02826.
    * Compute phase factors using the constructive method in
      https://arxiv.org/pdf/2308.01501.

    Although the procedure is exact, when considering large degrees numerical errors may
    arise.

    Args:
        target_polynomial (FourierPolynomial): Target Fourier series.

    """

    def __init__(self, target_polynomial: FourierPolynomial):
        """Initialize GQSPAngleFinder."""
        self._target_polynomial = target_polynomial
        self._degree = len(target_polynomial.coeffs) - 1
        self._complementary_polynomial = self._get_complementary_polynomial()

        S = np.array(
            [
                self._target_polynomial.coeffs_list,
                self._complementary_polynomial.coeffs_list,
            ],
            order="F",
        )
        self._phase_factors = self._get_phase_factors(S, self._degree)

    @property
    def degree(self) -> int:
        """Return the degree."""
        return self._degree

    @property
    def target_polynomial(self) -> FourierPolynomial:
        """Return the target polynomial."""
        return self._target_polynomial

    @property
    def complementary_polynomial(self) -> FourierPolynomial:
        """Return the complementary polynomial."""
        return self._complementary_polynomial

    @property
    def phase_factors(self) -> NDArray[np.float64]:
        """Return the phases."""
        return self._phase_factors

    @property
    def G_roots(self) -> NDArray[np.complex128]:
        """Return roots of the Laurent poly sorted by increasing magn."""
        return self._G_roots

    @property
    def prefactor(self) -> np.float64:
        """Return the prefactor of the Laurent poly, must be real."""
        return self._prefactor

    def _get_complementary_polynomial(self) -> FourierPolynomial:
        """Return the auxiliary polynomial Q satisfying unitary condition.

        Follows Proof of Lemma 4 in https://arxiv.org/pdf/2206.02826.
        """
        f_coeffs = self._target_polynomial.coeffs
        # Find coefficients of Laurent polynomial
        laurent_coeff: list[np.complex128] = []
        # k<0
        for k in range(-int(self._degree), 0):
            tmp = np.complex128(0)
            for i in range(-int(self._degree / 2), int(self._degree / 2) + k + 1):
                tmp += f_coeffs[i] * np.conj(f_coeffs[i - k])
            laurent_coeff.append(-1.0 * tmp)
        # k=0
        tmp = np.complex128(0)
        for i in range(-int(self._degree / 2), int(self._degree / 2) + 1):
            tmp += np.square(np.abs(f_coeffs[i]))
        laurent_coeff.append(1.0 - 1 * tmp)
        # k>0
        for k in range(1, int(self._degree) + 1):
            tmp = np.complex128(0)
            for i in range(-int(self._degree / 2) + k, int(self._degree / 2) + 1):
                tmp += f_coeffs[i] * np.conj(f_coeffs[i - k])
            laurent_coeff.append(-1 * tmp)

        # Find roots of Laurent polynomial and sort them by magnitude
        p = Polynomial(np.array(laurent_coeff))
        G_roots = p.roots()
        mod_G_roots = np.abs(G_roots)
        idxs = mod_G_roots.argsort()
        G_roots = G_roots[idxs]
        prefactor = np.sqrt(laurent_coeff[-1] * np.prod(G_roots[0 : self._degree]))

        # Vieta's formula
        roots = G_roots[: self._degree]
        h_poly_coeffs = polyfromroots(1 / np.conj(roots))
        h_coeff = {
            k - int(self._degree / 2): prefactor * complex(h_poly_coeffs[k])
            for k in range(len(h_poly_coeffs))
        }

        assert np.abs(np.imag(prefactor)) < 1e-12, (
            f"Prefactor must be real, but {prefactor} obtained."
        )

        self._G_roots = G_roots
        self._prefactor = prefactor

        complementary_polynomial = FourierPolynomial(
            h_coeff, self._target_polynomial.d_max, self._target_polynomial.d_min
        )
        return complementary_polynomial

    def _get_phase_factors(
        self, S: NDArray[np.complex128 | np.float64], d: int
    ) -> NDArray[np.float64]:
        """Return phase factors for GQSP given polynomials coeffs.

        From https://arxiv.org/pdf/2308.01501.
        """
        a_d, b_d = S[0][d], S[1][d]
        theta_d: np.float64 = np.arctan(np.abs(b_d) / np.abs(a_d))
        phi_d: np.float64 = np.angle(np.array([a_d / b_d]))[0]
        if d == 0:
            lam = np.angle(b_d)
            return np.array([[theta_d, phi_d, lam]])
        else:
            new_S = self._arbitrary_su2(theta_d, phi_d, 0.0).conj().T @ S
            new_S = np.array([new_S[0][1:], new_S[1][0:d]])
            return np.append(
                self._get_phase_factors(new_S, d - 1),
                np.array([(theta_d, phi_d, 0)]),
                axis=0,
            )

    @staticmethod
    def _arbitrary_su2(
        theta: np.float64, phi: np.float64, lam: float
    ) -> NDArray[np.complex128]:
        """Return R(theta, phi, lambda)."""
        return np.array(
            [
                [
                    np.exp(1j * (lam + phi)) * np.cos(theta),
                    np.exp(1j * phi) * np.sin(theta),
                ],
                [np.exp(1j * lam) * np.sin(theta), -np.cos(theta)],
            ]
        )
