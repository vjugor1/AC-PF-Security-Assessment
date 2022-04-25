import sys
from scipy import stats
import numpy as np
import pandapower as pp
import os

import multiprocess as multiprocessing

from copy import deepcopy

sys.path.append("..")
from src.estimators.static import SecurityAssessmentEstimator
from src.samplers.sampler import Sampler


class AdaptiveEstimator:
    def __init__(
        self,
        net,
        fluct_gens_idxs,
        fluct_loads_idxs,
        mu_init,
        sigma_level=50,
        batch_size=16,
    ):
        self.net = net
        self.fluct_gens = fluct_gens_idxs
        self.fluct_loads = fluct_loads_idxs
        # assert len(mu_init) == len(self.fluct_gens) + len(self.fluct_loads),
        self.mu = mu_init
        self.nominal_d = stats.multivariate_normal(
            mean=mu_init, cov=np.eye(len(mu_init)) * sigma_level
        )
        self.importance_d = stats.multivariate_normal(
            mean=mu_init, cov=np.eye(len(mu_init)) * sigma_level
        )
        self.nominal_d_load = stats.multivariate_normal(
            mean=np.zeros(len(self.fluct_loads)),
            cov=np.eye(len(self.fluct_loads)) * sigma_level,
        )
        self.importance_d_load = (
            stats.multivariate_normal(
                mean=np.zeros(len(self.fluct_loads)),
                cov=np.eye(len(self.fluct_loads)) * sigma_level,
            )
            if len(self.fluct_loads) > 0
            else None
        )
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
        self.weightes_outcomes = []
        self.mu_history = [mu_init]
        self.grad_history = []
        self.n_steps = 0
        self.batch_size = batch_size
        # try:
        #     self.Pg = net['res_gen']
        #     self.Pl = net['res_load']['p_mw']
        #     self.Ql = net['res_load']['q_mvar']
        #     self.Pg_lims = [net['gen']['max_p_mw'].values, net['gen']['min_p_mw'].values]
        # except KeyError:
        if len(net["res_gen"]) == 0:
            pp.runopp(net)
        self.Pg = net["res_gen"]["p_mw"]
        self.Pl = net["res_load"]["p_mw"]
        self.Ql = net["res_load"]["q_mvar"]
        self.Pg_lims = [net["gen"]["max_p_mw"].values, net["gen"]["min_p_mw"].values]
        # Zeroing cost function, since we only need to check feasibility
        for i in range(len(net.poly_cost)):
            self.net["poly_cost"]["cp1_eur_per_mw"].iloc[i] = 0.0
            self.net["poly_cost"]["cp2_eur_per_mw2"].iloc[i] = 0.0

    def estimate(self):
        return sum(self.weightes_outcomes) / self.n_steps

    def check_feasibility(self, sample):
        local_net = deepcopy(self.net)
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
        pp.runpp(local_net)

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

    # def estimate_grad(self, s):
    #     # for j in range(self.batch_size):
    #     indicator = int(self.check_feasibility(s))
    #     # self.weightes_outcomes.append(
    #     #     indicator
    #     #     * self.nominal_d.pdf(s["Gen"])
    #     #     / (self.importance_d.pdf(s["Gen"]) + 1e-8)
    #     # )
    #     weighted_outcomes = (
    #         indicator
    #         * self.nominal_d.pdf(s["Gen"])
    #         / (self.importance_d.pdf(s["Gen"]) + 1e-8)
    #     )
    #     curr_grad = (
    #         -indicator
    #         * self.nominal_d.pdf(s["Gen"]) ** 2
    #         / (self.importance_d.pdf(s["Gen"]) ** 2 + 1e-8)
    #         * (s["Gen"] - self.mu)
    #     )
    #     return curr_grad, weighted_outcomes

    def estimate_batch(self):
        # s = next(self.Isampler.sample())
        # indicator_tmp = int(self.check_feasibility(s))

        # curr_grad_tmp = - indicator_tmp * self.nominal_d.pdf(s['Gen']) ** 2 / (self.importance_d.pdf(s['Gen']) ** 2 + 1e-8) * (s['Gen'] - self.mu)
        # this guy should be parallelized.... not the outer loop - since the updates
        # s = next(self.Isampler.sample())
        samples_foos = [
            (
                next(self.Isampler.sample()),
                self.check_feasibility,
                deepcopy(self.nominal_d.pdf),
                deepcopy(self.importance_d.pdf),
                deepcopy(self.mu),
            )
            for j in range(self.batch_size)
        ]
        # with multiprocessing.Pool() as pool:
        #     curr_grads_outcomes = pool.starmap(
        #         estimate_grad,
        #         samples_foos,
        #     )
        curr_grads_outcomes = [estimate_grad(v[0], v[1], v[2], v[3], v[4]) for v in samples_foos]
        curr_grads = [v[0] for v in curr_grads_outcomes]
        self.weightes_outcomes += [v[1] for v in curr_grads_outcomes]
        # for j in range(self.batch_size):
        #     s = next(self.Isampler.sample())
        #     indicator_tmp += int(self.check_feasibility(s))
        #     self.weightes_outcomes.append(indicator_tmp * self.nominal_d.pdf(s['Gen']) / (self.importance_d.pdf(s['Gen']) + 1e-8))
        #     curr_grad_tmp += - indicator_tmp * self.nominal_d.pdf(s['Gen']) ** 2 / (self.importance_d.pdf(s['Gen']) ** 2 + 1e-8) * (s['Gen'] - self.mu)

        curr_grad = np.mean(curr_grads, axis=0)  # curr_grad_tmp / (self.batch_size)
        # indicator
        self.grad_history.append(np.copy(curr_grad))
        mu_new = self.mu - 1e-3 * curr_grad
        # self.sampler.gen_sample_foo = lambda x: np.random.multivariate_normal(mu_new, np.eye(len(self.fluct_gens)) * 50)
        self.mu = mu_new
        self.importance_d.mean = self.mu
        self.mu_history.append(np.copy(self.importance_d.mean))
        self.n_steps += 1

    def test_samples(self, N):
        # for i in range(N):
        for i in range(N):
            self.estimate_batch()


def estimate_grad(
    s,
    check_feasibility,
    nominal_pdf,
    importance_pdf,
    mu,
):
    # for j in range(self.batch_size):
    indicator = int(check_feasibility(s))
    weighted_outcomes = (
        indicator * nominal_pdf(s["Gen"]) / (importance_pdf(s["Gen"]) + 1e-8)
    )
    curr_grad = (
        -indicator
        * nominal_pdf(s["Gen"]) ** 2
        / (importance_pdf(s["Gen"]) ** 2 + 1e-8)
        * (s["Gen"] - mu)
    )
    return curr_grad, weighted_outcomes
