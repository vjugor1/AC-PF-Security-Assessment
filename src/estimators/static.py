from copy import deepcopy
import pandapower as pp
from pandapower import OPFNotConverged
import multiprocessing


class SecurityAssessmentEstimator:
    """Monte-Carlo feasibiliy of a power grid estimator"""

    def __init__(self, net, fluct_gens_idxs, fluct_loads_idxs):
        """Initialization of an instance of this class

        Args:
            net (pandapower.auxiliary.pandapowerNet): power grid
            fluct_gens_idxs (list): list of generators' indexes that are fluctuating
            fluct_loads_idxs (list): list of loads' indexes that are fluctuating
        """
        self.net = net
        self.fluct_gens = fluct_gens_idxs
        self.fluct_loads = fluct_loads_idxs
        # Saving reference values - current operating point, limits that define feasibility set apart from Power Flow Equations (PFE)
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

    def estimate(self, samples, parallel=True):
        """Make estimation samples on given samples

        Args:
            samples (List): List of samples in a form [(sample, ) for sample in samples] -- to be able to being run in parallel

        Returns:
            List: Feasibility checks of each sample in list
        """
        if parallel:
            self_samples = [(s) for s in samples]
            with multiprocessing.Pool() as pool:
                est = pool.starmap(self.check_feasibility, self_samples)
        else:
            est = []
            for s in samples:
                est.append(self.check_feasibility(s[0]))

        return est
