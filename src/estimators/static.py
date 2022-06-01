from copy import deepcopy
import pandapower as pp
from pandapower import OPFNotConverged
import multiprocessing
from src.estimators.feasibility import *
from tqdm import tqdm


class SecurityAssessmentEstimator:
    """Monte-Carlo feasibiliy of a power grid estimator"""

    def __init__(self, fluct_gens_idxs, fluct_loads_idxs, net=None, functions=None):
        """Initialization of an instance of this class

        Args:
            net (pandapower.auxiliary.pandapowerNet): power grid
            functions (list): functions that define feasibility set as `max(f(x) for f in functions) <= 0`
            fluct_gens_idxs (list): list of generators' indexes that are fluctuating
            fluct_loads_idxs (list): list of loads' indexes that are fluctuating
        """
        self.net = net
        self.functions = functions
        self.fluct_gens = fluct_gens_idxs
        self.fluct_loads = fluct_loads_idxs
        if self.net is None:
            assert (
                len(self.fluct_loads) == 0
            ), "For non power grid cases define fluctuations via `fluct_gens_idxs`"
        assert (self.net is not None and self.functions is None) or (
            self.net is None and self.functions is not None
        ), "choose between power grid or analytical formulation"
        if self.net is not None:
            # Saving reference values - current operating point, limits that define feasibility set apart from Power Flow Equations (PFE)
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

        if self.net is None:
            self.violation_history = []  # for logging which functions were violated

        # feasibility function
        if self.net is not None:
            self.check_feasibility = check_feasibility_grid
        else:
            self.check_feasibility = check_feasibility_analytic

    def estimate(self, samples):
        """Make estimation samples on given samples

        Args:
            samples (List): List of samples in a form [(sample, ) for sample in samples] -- to be able to being run in parallel

        Returns:
            List: Feasibility checks of each sample in list
        """
        self_samples = [(self, s[0]) for s in samples]
        with multiprocessing.Pool() as pool:
            est = pool.starmap(self.check_feasibility, self_samples,)
        # est = [0]
        # for s in tqdm(samples):
        #     est.append(self.check_feasibility(s[0]))

        return est
