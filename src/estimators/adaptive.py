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


class AdaptiveEstimator:
    """This class describes adaptive estimator, which can check feasibility of current point, estimate feasibility probability, adapt parameters"""

    def __init__(
        self,
        fluct_gens_idxs,
        fluct_loads_idxs,
        mu_init,
        sigma_init,
        net=None,
        batch_size=16,
    ):
        """Initialization of instance of this class

        Args:
            net (pandapower.auxiliary.pandapowerNet): the system
            fluct_gens_idxs (list): list of generators' indexes that are fluctuating
            fluct_loads_idxs (list): list of loads' indexes that are fluctuating
            mu_init (np.ndarray): initial for mu for importance distribution
            sigma_init (np.ndarray): initial vector of diagonal elements for sigma matrix.
            batch_size (int, optional): number of samples that are used to estimate stochastic gradient. Defaults to 16.
        """

        self.net = net

        self.fluct_gens = fluct_gens_idxs
        self.fluct_loads = fluct_loads_idxs
        if self.net is None:
            assert (
                len(self.fluct_loads) == 0
            ), "For non power grid problem, define fluctuations via `fluct_gens`"
        # assert len(mu_init) == len(self.fluct_gens) + len(self.fluct_loads),
        self.mu = mu_init
        self.sigma = sigma_init
        # Assembling nominal and importance (to be optimized) distribution
        self.nominal_d = stats.multivariate_normal(
            mean=mu_init, cov=np.diag(sigma_init)
        )
        self.importance_d = stats.multivariate_normal(
            mean=mu_init, cov=np.diag(sigma_init)
        )
        self.nominal_d_load = stats.multivariate_normal(
            mean=np.zeros(len(self.fluct_loads)),
            cov=np.eye(len(self.fluct_loads)) * sigma_init,
        )
        self.importance_d_load = (
            stats.multivariate_normal(
                mean=np.zeros(len(self.fluct_loads)),
                cov=np.eye(len(self.fluct_loads)) * sigma_init,
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
        self.weightes_outcomes = []
        self.mu_history = [mu_init]
        self.sigma_history = [sigma_init]
        self.grad_history = []
        self.n_steps = 0
        self.batch_size = batch_size
        # Saving reference values - current operating point, limits that define feasibility set apart from Power Flow Equations (PFE)
        if net is not None:
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

    def estimate(self):
        """Estimates current feasibility probability based on history stored
        Returns:
            float: feasibility probability estimate
        """
        return sum(self.weightes_outcomes) / self.n_steps

    def check_feasibility(self, sample):
        """True - system is not feasible with given sample,
           False - system is feabile with given sample
        Args:
            sample (Dict): formatted sample -- see Sampler from src.samplers.sampler
        Returns:
            bool: if system is infeasible
        """
        # local copy of the system
        local_net = deepcopy(self.net)
        # introduce sample into the system
        if self.fluct_gens is not None and sample["Gen"] is not None:
            for idx, i in enumerate(self.fluct_gens):
                local_net["gen"]["p_mw"].iloc[i] = self.Pg[i] + sample["Gen"][idx]
        if self.fluct_loads is not None and sample["Load"] is not None:
            for idx, i in enumerate(self.fluct_loads):
                local_net["load"]["p_mw"].iloc[i] = (
                    self.Pl[i] + sample["Load"]["P"][idx]
                )
                local_net["load"]["q_mvar"].iloc[i] = (
                    self.Ql[i] + sample["Load"]["Q"][idx]
                )
        # Solve PFE
        pp.runpp(local_net)
        # Check operating limits for violation
        # line currents
        curr_from = local_net.res_line["i_from_ka"].values
        curr_to = local_net.res_line["i_to_ka"].values
        load_perc = local_net.res_line["loading_percent"]
        lines_satisfied = (
            (local_net.line["max_i_ka"] >= curr_from).all()
            and (local_net.line["max_i_ka"] >= curr_to).all()
            and (local_net.line["max_loading_percent"] >= load_perc).all()
        )
        # buses
        Vm = local_net.res_bus["vm_pu"].values
        bus_satisfied = (Vm <= local_net.bus["max_vm_pu"]).all() and (
            Vm >= local_net.bus["min_vm_pu"]
        ).all()
        # generatos
        pmw = local_net.res_gen["p_mw"]
        qmvar = local_net.res_gen["q_mvar"]
        gen_satisfied = (
            (pmw <= local_net.gen["max_p_mw"]).all()
            and (pmw >= local_net.gen["min_p_mw"]).all()
            and (qmvar <= local_net.gen["max_q_mvar"]).all()
            and (qmvar >= local_net.gen["min_q_mvar"]).all()
        )
        cond_feasible = lines_satisfied and bus_satisfied and gen_satisfied
        return not cond_feasible

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

