# optional argument: all_prices (if we need to calculate the Markowitz parameters)

import cvxpy as cp
import numpy as np


def get_optimization_input(price_array: np.ndarray, hyperparams):
    v = hyperparams["v"]
    def update_hyperparams(hyperparams):
        nonlocal v
        v = hyperparams["v"]


    def max_return_with_turnover(mu: np.ndarray,
                                        w_prev: np.ndarray,
                                        turnover_penalty: float = 1.0,
                                        long_only: bool = True) -> np.ndarray:
        """
        Solve:
            max_w   mu.T @ w  -  turnover_penalty * ||w - w_prev||^2
            s.t.    sum(w) == 1
                    w >= 0            (if long_only)
        Args:
            mu               : (n,) vector of expected returns
            w_prev           : (n,) vector of current weights
            turnover_penalty : penalty coefficient (higher ⇒ smaller moves)
            long_only        : if True, enforces w >= 0
        Returns:
            w_opt : (n,) optimal new weights
        """
        n = mu.shape[0]
        w = cp.Variable(n)

        # Objective: maximize return minus turnover penalty
        objective = cp.Maximize(
            mu @ w
            - turnover_penalty * cp.sum_squares(w - w_prev)
        )

        # Constraints
        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= 0)

        # Solve
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)  # OSQP handles quadratic objectives well

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Solver did not converge: {prob.status}")

        return w.value


    def optimize_mean_variance(mu: np.ndarray,
                            Sigma: np.ndarray,
                            risk_aversion: float = 1.0,
                            long_only: bool = True) -> np.ndarray:
        """
        Solve the Markowitz mean–variance utility:
            max_w   mu^T w - (risk_aversion/2) * w^T Sigma w
            s.t.    sum(w) == 1
                    w >= 0    (if long_only)

        Args:
            mu             : (n,) vector of expected returns
            Sigma          : (n,n) covariance matrix
            risk_aversion  : lambda parameter (higher => more penalty on variance)
            long_only      : if True, enforces w >= 0

        Returns:
            w_opt : (n,) optimal portfolio weights
        """
        n = mu.shape[0]
        w = cp.Variable(n)

        # Objective: maximize return minus variance penalty
        objective = cp.Maximize(
            mu @ w
            - (risk_aversion / 2) * cp.quad_form(w, Sigma)
        )

        # Constraints: fully invested, optionally long-only
        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= 0)

        # Solve the QP
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP)  # or cp.SCS, cp.ECOS

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Solver status: {prob.status}")

        return w.value

    def max_sharpe_ratio(mu: np.ndarray,
                        Sigma: np.ndarray,
                        rf: float = 0.0) -> np.ndarray:
        """
        Compute the portfolio weights w that maximize the Sharpe ratio:
            max_w   (mu^T w - rf) / sqrt(w^T Sigma w)
            s.t.    sum(w) = 1

        Under no other constraints, the solution is the "tangency portfolio":
            w ∝ Sigma^{-1} (mu - rf*1)
            then scaled to sum to 1.

        Args:
            mu    : (n,) array of expected returns
            Sigma : (n,n) covariance matrix (must be invertible)
            rf    : risk-free rate (scalar)

        Returns:
            w     : (n,) array of portfolio weights summing to 1
        """
        # 1) Compute excess returns
        excess = mu - rf

        # 2) Invert the covariance
        inv_S = np.linalg.inv(Sigma)

        # 3) Raw (unnormalized) weights
        w_unnormalized = inv_S.dot(excess)

        # 4) Normalize to sum to 1
        w = w_unnormalized / np.sum(w_unnormalized)
        return w



    functions =  {
        'max_return_with_turnover': max_return_with_turnover,
        'optimize_mean_variance': optimize_mean_variance,
        'max_sharpe_ratio': max_sharpe_ratio,
        'update_hyperparams': hyperparams
    }

    return functions