from copy import deepcopy
import numpy as np
import pandapower as pp


def check_feasibility_grid(self, sample):
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
            local_net["load"]["p_mw"].iloc[i] = self.Pl[i] + sample["Load"]["P"][idx]
            local_net["load"]["q_mvar"].iloc[i] = self.Ql[i] + sample["Load"]["Q"][idx]
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


def check_feasibility_analytic(self, sample):
    """True - system is not feasible with given sample,
        False - system is feabile with given sample
    Args:
        sample (Dict): formatted sample -- see Sampler from src.samplers.sampler
    Returns:
        bool: if system is infeasible
    """

    self.violation_history.append(
        np.array([f(sample["Gen"]) > 0 for f in self.functions])
    )
    cond_infeasible = (self.violation_history[-1] == True).any()

    return cond_infeasible
