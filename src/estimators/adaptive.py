import sys
from scipy import stats
import numpy as np
import pandapower as pp
import os
import math

import multiprocess as multiprocessing

from copy import deepcopy

sys.path.append("..")
from src.estimators.static import SecurityAssessmentEstimator
from src.samplers.sampler import Sampler
from src.estimators.feasibility import *


class AdaptiveEstimator:
    """This class describes adaptive estimator, which can check feasibility of current point, estimate feasibility probability, adapt parameters"""

    def __init__(
        self,
        fluct_gens_idxs,
        fluct_loads_idxs,
        mu_init,
        sigma_init,
        net=None,
        functions=None,  # [lambda x: np.linalg.norm(x) - 25, lambda x: x[0] - x[1] ** 2],
        batch_size=16,
    ):
        """Initialization of instance of this class

        Args:
            net (pandapower.auxiliary.pandapowerNet): the system
            functions (list): functions that define feasibility set as `max(f(x) for f in functions) <= 0`
            fluct_gens_idxs (list): list of generators' indexes that are fluctuating
            fluct_loads_idxs (list): list of loads' indexes that are fluctuating
            mu_init (np.ndarray): initial for mu for importance distribution
            sigma_init (tuple): initial vector of diagonal elements for sigma matrix. first element is for generators, second is for loads
            batch_size (int, optional): number of samples that are used to estimate stochastic gradient. Defaults to 16.
        """

        self.net = net
        self.functions = functions
        self.fluct_gens = fluct_gens_idxs
        self.fluct_loads = fluct_loads_idxs
        assert (len(self.fluct_gens) > 0) or (
            len(self.fluct_loads) > 0
        ), "something must fluctuate"
        assert len(self.fluct_gens) == len(sigma_init[0]) and len(
            self.fluct_loads
        ) == len(
            sigma_init[1]
        ), "cov diagonal size must match number of fluctuating units"
        if self.net is None:
            assert (
                len(self.fluct_loads) == 0
            ), "For non power grid cases define fluctuations via `fluct_gens_idxs`"
        assert (self.net is not None and self.functions is None) or (
            self.net is None and self.functions is not None
        ), "choose between power grid or analytical formulation"
        # assert len(mu_init) == len(self.fluct_gens) + len(self.fluct_loads),
        self.mu = mu_init
        self.sigma = sigma_init[0]
        # Assembling nominal and importance (to be optimized) distribution

        if len(self.fluct_gens) > 0:
            self.nominal_d = stats.multivariate_normal(
                mean=mu_init, cov=np.diag(sigma_init[0])
            )
            self.importance_d = stats.multivariate_normal(
                mean=mu_init, cov=np.diag(sigma_init[0])
            )
        if len(self.fluct_loads) > 0:
            self.nominal_d_load = stats.multivariate_normal(
                mean=np.zeros(len(self.fluct_loads)), cov=np.diag(sigma_init[1])
            )
            self.importance_d_load = (
                stats.multivariate_normal(
                    mean=np.zeros(len(self.fluct_loads)), cov=np.diag(sigma_init[1])
                )
                if len(self.fluct_loads) > 0
                else None
            )
        # Wrapping into sampler instance
        self.Nsampler = Sampler(
            len(self.fluct_gens),
            len(self.fluct_loads),
            lambda: self.nominal_d.rvs() if len(self.fluct_gens) > 0 else None,
            lambda: {"P": self.nominal_d_load.rvs(), "Q": self.nominal_d_load.rvs()}
            if len(self.fluct_loads) > 0
            else None,
        )
        self.Isampler = Sampler(
            len(self.fluct_gens),
            len(self.fluct_loads),
            lambda: self.importance_d.rvs() if len(self.fluct_gens) > 0 else None,
            lambda: {"P": self.nominal_d_load.rvs(), "Q": self.nominal_d_load.rvs()}
            if len(self.fluct_loads) > 0
            else None,
        )
        # Saving logging data
        if self.net is None:
            self.violation_history = []  # for logging which functions were violated
        self.weightes_outcomes = []
        self.mu_history = [mu_init]
        self.sigma_history = [sigma_init]
        self.grad_history = []
        self.n_steps = 0
        self.batch_size = batch_size
        # Saving reference values - current operating point, limits that define feasibility set apart from Power Flow Equations (PFE)
        if self.net is not None:
            if len(net["res_gen"]) == 0:
                pp.runopp(net)
            self.Pg = net["res_gen"]["p_mw"]
            self.Pl = net["res_load"]["p_mw"]
            self.Ql = net["res_load"]["q_mvar"]
            self.Pg_lims = [
                net["gen"]["max_p_mw"].values,
                net["gen"]["min_p_mw"].values,
            ]
            # Zeroing cost function, since we only need to check feasibility
            for i in range(len(net.poly_cost)):
                self.net["poly_cost"]["cp1_eur_per_mw"].iloc[i] = 0.0
                self.net["poly_cost"]["cp2_eur_per_mw2"].iloc[i] = 0.0
        # feasibility function
        if self.net is not None:
            self.check_feasibility = lambda x: check_feasibility_grid(self, x)
        else:
            self.check_feasibility = lambda x: check_feasibility_analytic(self, x)

    def estimate(self):
        """Estimates current feasibility probability based on history stored
        Returns:
            float: feasibility probability estimate
        """
        return sum(self.weightes_outcomes) / self.n_steps

    def estimate_batch(self):
        """Estimate gradient on the batch"""
        samples_foos = [
            (
                next(self.Isampler.sample()),
                self.check_feasibility,
                deepcopy(self.nominal_d.pdf),
                deepcopy(self.importance_d.pdf),
                deepcopy(self.mu),
                deepcopy(self.sigma),
            )
            for j in range(self.batch_size)
        ]
        curr_grads_outcomes = [
            estimate_grad(v[0], v[1], v[2], v[3], v[4], v[5]) for v in samples_foos
        ]
        curr_grads = [v[0] for v in curr_grads_outcomes]
        self.weightes_outcomes += [v[1] for v in curr_grads_outcomes]
        curr_grad = np.mean(curr_grads, axis=0)  # curr_grad_tmp / (self.batch_size)
        # indicator
        self.grad_history.append(np.copy(curr_grad))
        mu_new = self.mu - 1e-3 * curr_grad[: len(self.mu)]
        sigma_new = self.sigma - 1e-3 * curr_grad[len(self.mu) :]
        self.mu = mu_new
        self.sigma = sigma_new
        self.importance_d.mean = self.mu
        self.importance_d.cov = np.diag(self.sigma)
        self.mu_history.append(np.copy(self.mu))

        self.n_steps += 1

    def test_samples(self, N):
        """Estimate on N samples and store the progress in the corresponding fields of this class
        Args:
            N (int): Number of steps
        """
        # for i in range(N):
        for i in range(N):
            self.estimate_batch()


def estimate_grad(
    s, check_feasibility, nominal_pdf, importance_pdf, mu, sigma,
):
    """Estimates gradient based on sample `s`
    Args:
        s (Dict): Formatted sample
        check_feasibility (function): returns result of feasibility check: True - infeasible, False - feasible
        nominal_pdf (function): pdf of nominal distribution
        importance_pdf (function): pdf of optimized importance distribution
        mu (np.ndarray): current mean vector for importance distribution that is being optimized
        sigma (np.ndarray): current vector of diagonal elements of covariance matrix for importance distribution
    Returns:
        np.ndarray: estimated stochastic gradient of variance of the estimate
    """
    # for j in range(self.batch_size):
    indicator = int(check_feasibility(s))
    weighted_outcomes = (
        indicator * nominal_pdf(s["Gen"]) / (importance_pdf(s["Gen"]) + 1e-8)
    )
    curr_grad_mu = (
        -indicator
        * nominal_pdf(s["Gen"]) ** 2
        / (importance_pdf(s["Gen"]) ** 2 + 1e-8)
        * np.diag([1 / x for x in sigma]).dot((s["Gen"] - mu))
    )

    grad_sigma = lambda sigma_i: (2 * np.pi) ** (-0.5 * len(sigma)) * (
        -2 * sigma_i ** -3 * np.prod(sigma) ** -2
        + np.exp(
            -0.5
            * (
                float(
                    np.dot((s["Gen"] - mu), np.diag([1 / x for x in sigma])).dot(
                        (s["Gen"] - mu)[np.newaxis].T
                    )
                )
            )
        )
        * sigma_i ** -3
    )

    curr_grad_sigma = np.array([grad_sigma(sigma_i) for sigma_i in sigma])

    grads = [curr_grad_mu, curr_grad_sigma]
    curr_grad = np.concatenate(grads)
    return curr_grad, weighted_outcomes

