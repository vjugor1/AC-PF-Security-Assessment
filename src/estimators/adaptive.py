
import sys
from scipy import stats
import numpy as np
import pandapower as pp
import os

sys.path.append('..')
from src.estimators.static import SecurityAssessmentEstimator
from src.samplers.sampler import Sampler


class AdaptiveEstimator(SecurityAssessmentEstimator):
    def __init__(self, net, fluct_gens_idxs, fluct_loads_idxs, mu_init, sigma_level=50):
        self.net = net
        self.fluct_gens = fluct_gens_idxs
        self.fluct_loads = fluct_loads_idxs
        #assert len(mu_init) == len(self.fluct_gens) + len(self.fluct_loads), 
        self.mu = mu_init
        self.nominal_d = stats.multivariate_normal(mean=mu_init, cov=np.eye(len(mu_init)) * sigma_level)
        self.importance_d = stats.multivariate_normal(mean=mu_init, cov=np.eye(len(mu_init)) * sigma_level)
        self.nominal_d_load = stats.multivariate_normal(mean=np.zeros(len(self.fluct_loads)), cov=np.eye(len(self.fluct_loads)) * sigma_level)
        self.importance_d_load = stats.multivariate_normal(mean=np.zeros(len(self.fluct_loads)), cov=np.eye(len(self.fluct_loads)) * sigma_level) if len(self.fluct_loads) > 0 else None
        self.Nsampler = Sampler(len(self.fluct_gens), len(self.fluct_loads), lambda: self.nominal_d.rvs() if len(self.fluct_gens) > 0  else None, lambda: {"P": self.nominal_d_load.rvs(), "Q": self.nominal_d_load.rvs()} if len(self.fluct_loads) > 0 else None)
        self.Isampler = Sampler(len(self.fluct_gens), len(self.fluct_loads), lambda: self.importance_d.rvs() if len(self.fluct_gens) > 0  else None, lambda: {"P": self.nominal_d_load.rvs(), "Q": self.nominal_d_load.rvs()} if len(self.fluct_loads) > 0  else None)
        self.weightes_outcomes = []
        self.mu_history = [mu_init]
        self.grad_history = []
        self.n_steps = 0
        # try:
        #     self.Pg = net['res_gen']
        #     self.Pl = net['res_load']['p_mw']
        #     self.Ql = net['res_load']['q_mvar']
        #     self.Pg_lims = [net['gen']['max_p_mw'].values, net['gen']['min_p_mw'].values]
        # except KeyError:
        if len(net['res_gen']) == 0:
            pp.runopp(net)
        self.Pg = net['res_gen']['p_mw']
        self.Pl = net['res_load']['p_mw']
        self.Ql = net['res_load']['q_mvar']
        self.Pg_lims = [net['gen']['max_p_mw'].values, net['gen']['min_p_mw'].values]
        # Zeroing cost function, since we only need to check feasibility
        for i in range(len(net.poly_cost)):
            self.net['poly_cost']['cp1_eur_per_mw'].iloc[i] = 0.0
            self.net['poly_cost']['cp2_eur_per_mw2'].iloc[i] = 0.0
    def estimate(self):
        return sum(self.weightes_outcomes) / self.n_steps
    def test_samples(self, N):
        for i in range(N):
            s = next(self.Isampler.sample())
            indicator_tmp = int(self.check_feasibility(s))
            
            curr_grad_tmp = - indicator_tmp * self.nominal_d.pdf(s['Gen']) ** 2 / (self.importance_d.pdf(s['Gen']) ** 2 + 1e-8) * (s['Gen'] - self.mu)
            for j in range(N // 16):
                s = next(self.Isampler.sample())
                indicator_tmp += int(self.check_feasibility(s))
                self.weightes_outcomes.append(indicator_tmp * self.nominal_d.pdf(s['Gen']) / (self.importance_d.pdf(s['Gen']) + 1e-8))
                curr_grad_tmp += - indicator_tmp * self.nominal_d.pdf(s['Gen']) ** 2 / (self.importance_d.pdf(s['Gen']) ** 2 + 1e-8) * (s['Gen'] - self.mu)
            curr_grad = curr_grad_tmp / (N // 16)
            #indicator
            
            self.grad_history.append(np.copy(curr_grad))
            mu_new = self.mu - 1e-3 * curr_grad
            #self.sampler.gen_sample_foo = lambda x: np.random.multivariate_normal(mu_new, np.eye(len(self.fluct_gens)) * 50)
            self.mu = mu_new
            self.importance_d.mean = self.mu
            self.mu_history.append(np.copy(self.importance_d.mean))
            self.n_steps += 1
        