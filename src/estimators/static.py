from copy import deepcopy
import pandapower as pp
from pandapower import OPFNotConverged
import multiprocessing


class SecurityAssessmentEstimator:
    def __init__(self, net, fluct_gens_idxs, fluct_loads_idxs):
        self.net = net
        self.fluct_gens = fluct_gens_idxs
        self.fluct_loads = fluct_loads_idxs
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

    def check_feasibility(self, sample):
        # if self.fluct_gens is not None and sample["Gen"] is not None:
        #     for idx, i in enumerate(self.fluct_gens):
        #         self.net["gen"]["max_p_mw"].iloc[i] = self.Pg[i] + sample["Gen"][idx]
        #         self.net["gen"]["min_p_mw"].iloc[i] = self.Pg[i] + sample["Gen"][idx]
        # gen_lims_exceed_cond = (
        #     self.net["gen"]["max_p_mw"] > self.Pg_lims[0]
        # ).any() or (self.net["gen"]["max_p_mw"] < self.Pg_lims[1]).any()
        # if gen_lims_exceed_cond:
        #     return True
        # if self.fluct_loads is not None and sample["Load"] is not None:
        #     for idx, i in enumerate(self.fluct_loads):
        #         self.net["load"]["p_mw"].iloc[i] = self.Pl[i] + sample["Load"]["P"][idx]
        #         self.net["load"]["min_p_mw"].iloc[i] = (
        #             self.Ql[i] + sample["Load"]["Q"][idx]
        #         )
        # try:
        #     pp.runopp(self.net, init="results")
        #     cond_feasible = False
        # except OPFNotConverged:
        #     cond_feasible = True
        # print(sample)
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

    def estimate(self, samples):
        # est = []
        with multiprocessing.Pool() as pool:
            est = pool.starmap(
                self.check_feasibility,
                samples,
            )
        # for s in samples:
        #     est.append(int(self.check_feasibility(s)))
        return est
