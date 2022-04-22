import math

import numpy as np

from scipy import optimize as optim
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 18})
import scipy.linalg as la
import pandapower as pp
from pandapower import networks as nw


from sys import stderr
import sys
from numpy import array, zeros, ones, any, diag, r_, pi, Inf, isnan, arange, c_, dot

from numpy import flatnonzero as find

from scipy.sparse import vstack, hstack, csr_matrix as sparse

from pandapower.pypower.idx_bus import BUS_TYPE, REF, VA, LAM_P, LAM_Q, MU_VMAX, MU_VMIN
from pandapower.pypower.idx_gen import PG, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN
from pandapower.pypower.idx_brch import PF, PT, QF, QT, RATE_A, MU_SF, MU_ST
from pandapower.pypower.idx_cost import MODEL, POLYNOMIAL, PW_LINEAR, NCOST, COST
from pandapower.pypower.util import sub2ind

from scipy.linalg import pinvh

sys.path.append("..")
from src.solvers import utils as SU

grid_name_map = {}
#!!! nw is somehow invisible, so I make it thourgh converter
grid_name_map["grid4"] = nw.case4gs
grid_name_map["grid6"] = nw.case6ww
grid_name_map["grid14"] = nw.case14
grid_name_map["grid24"] = nw.case24_ieee_rts
grid_name_map["grid30"] = nw.case_ieee30
grid_name_map["grid39"] = nw.case39
grid_name_map["grid57"] = nw.case57
grid_name_map["grid89"] = nw.case89pegase
grid_name_map["grid118"] = nw.case118
grid_name_map["grid118i"] = nw.iceland
grid_name_map["grid145"] = nw.case145
grid_name_map["grid200"] = nw.case_illinois200
grid_name_map["grid300"] = nw.case300
grid_name_map["grid1354"] = nw.case1354pegase
grid_name_map["grid1888"] = nw.case1888rte
grid_name_map["grid2224"] = nw.GBnetwork
grid_name_map["grid2848"] = nw.case2848rte
grid_name_map["grid2869"] = nw.case2869pegase
grid_name_map["grid3120"] = nw.case3120sp
grid_name_map["grid6470"] = nw.case6470rte
grid_name_map["grid6495"] = nw.case6495rte
grid_name_map["grid6515"] = nw.case6515rte
grid_name_map["grid9241"] = nw.case9241pegase
grid_name_map["grid4"] = nw.case4gs
grid_name_map["grid6"] = nw.case6ww
grid_name_map["grid14"] = nw.case14
grid_name_map["grid24"] = nw.case24_ieee_rts
grid_name_map["grid30"] = nw.case_ieee30
grid_name_map["grid39"] = nw.case39
grid_name_map["grid57"] = nw.case57
grid_name_map["grid89"] = nw.case89pegase
grid_name_map["grid118"] = nw.case118
grid_name_map["grid118i"] = nw.iceland
grid_name_map["grid145"] = nw.case145
grid_name_map["grid200"] = nw.case_illinois200
grid_name_map["grid300"] = nw.case300
grid_name_map["grid1354"] = nw.case1354pegase
grid_name_map["grid1888"] = nw.case1888rte
grid_name_map["grid2224"] = nw.GBnetwork
grid_name_map["grid2848"] = nw.case2848rte
grid_name_map["grid2869"] = nw.case2869pegase
grid_name_map["grid3120"] = nw.case3120sp
grid_name_map["grid6470"] = nw.case6470rte
grid_name_map["grid6495"] = nw.case6495rte
grid_name_map["grid6515"] = nw.case6515rte
grid_name_map["grid9241"] = nw.case9241pegase


def get_grid_name_map():
    return grid_name_map


def available_grids():
    print(list(grid_name_map.keys()))


def get_linear_constraints(grid_name, check_pp_vs_new_form=False):
    """The function produces matrix A and vector b of linear constraint of DC OPF approximation

    Args:
        grid_name (string): grid name see `grid_name_map` above
        cov_std (float): variance of Gaussian fluctuations
        max_angle (float): max angle difference between buses
        shf_eps (float): thresold on shift significance in DC-PF Eqs pwr = pwr_shf + np.real(bbus)*va

    Returns:
        A_n (ndarray(lines, buses)): matrix of linear constraints
        b_n (ndarray(lines,)): vector of linear constraints
        gens (ndarray(buses,)): generator values
        cost_coeffs (ndarray(buses,))
    """
    net = get_grid_name_map()[grid_name]()
    pp.auxiliary._init_rundcopp_options(
        net,
        check_connectivity=False,
        switch_rx_ratio=0.5,
        delta=1e-10,
        trafo3w_losses="hv",
    )
    ac = net["_options"]["ac"]
    init = net["_options"]["init"]
    ppopt = pp.pypower.ppoption.ppoption(
        VERBOSE=False, OPF_FLOW_LIM=2, PF_DC=not ac, INIT=init
    )
    net["OPF_converged"] = False
    net["converged"] = False
    ppc, ppci = pp.pd2ppc._pd2ppc(net)
    ppc, ppopt = pp.pypower.opf_args.opf_args2(ppci, ppopt)
    om = pp.pypower.opf_setup.opf_setup(ppc, ppopt)
    om.build_cost_params()
    # om.get_cost_params()
    x0_pp, xmin_pp, xmax_pp = om.getv()

    ## unpack data
    ppc = om.get_ppc()
    baseMVA, bus, gen, branch, gencost = (
        ppc["baseMVA"],
        ppc["bus"],
        ppc["gen"],
        ppc["branch"],
        ppc["gencost"],
    )
    cp = om.get_cost_params()
    N, H, Cw = cp["N"], cp["H"], cp["Cw"]
    fparm = array(c_[cp["dd"], cp["rh"], cp["kk"], cp["mm"]])
    Bf = om.userdata("Bf")
    Pfinj = om.userdata("Pfinj")
    vv, ll, _, _ = om.get_idx()

    ## problem dimensions
    ipol = find(gencost[:, MODEL] == POLYNOMIAL)  ## polynomial costs
    ipwl = find(gencost[:, MODEL] == PW_LINEAR)  ## piece-wise linear costs
    nb = bus.shape[0]  ## number of buses
    nl = branch.shape[0]  ## number of branches
    nw = N.shape[0]  ## number of general cost vars, w
    ny = om.getN("var", "y")  ## number of piece-wise linear costs
    nxyz = om.getN("var")  ## total number of control vars of all types

    ## linear constraints & variable bounds
    A, l, u = om.linear_constraints()
    x0, xmin, xmax = om.getv()

    ## set up objective function of the form: f = 1/2 * X'*HH*X + CC'*X
    ## where X = [x;y;z]. First set up as quadratic function of w,
    ## f = 1/2 * w'*HHw*w + CCw'*w, where w = diag(M) * (N*X - Rhat). We
    ## will be building on the (optionally present) user supplied parameters.

    ## piece-wise linear costs
    any_pwl = int(ny > 0)
    if any_pwl:
        # Sum of y vars.
        Npwl = sparse(
            (ones(ny), (zeros(ny), arange(vv["i1"]["y"], vv["iN"]["y"]))), (1, nxyz)
        )
        Hpwl = sparse((1, 1))
        Cpwl = array([1])
        fparm_pwl = array([[1, 0, 0, 1]])
    else:
        Npwl = None  # zeros((0, nxyz))
        Hpwl = None  # array([])
        Cpwl = array([])
        fparm_pwl = zeros((0, 4))

    ## quadratic costs
    npol = len(ipol)
    if any(len(gencost[ipol, NCOST] > 3)) and sum(
        gencost[find(gencost[ipol, NCOST] > 3)][:][NCOST + 1 :]
    ):
        stderr.write(
            "DC opf cannot handle polynomial costs with higher "
            "than quadratic order.\n"
        )
    iqdr = find(gencost[ipol, NCOST] == 3)
    ilin = find(gencost[ipol, NCOST] == 2)
    polycf = zeros((npol, 3))  ## quadratic coeffs for Pg
    if len(iqdr) > 0:
        polycf[iqdr, :] = gencost[ipol[iqdr], COST : COST + 3]
    if npol:
        polycf[ilin, 1:3] = gencost[ipol[ilin], COST : COST + 2]
    polycf = dot(polycf, diag([baseMVA**2, baseMVA, 1]))  ## convert to p.u.
    if npol:
        Npol = sparse(
            (ones(npol), (arange(npol), vv["i1"]["Pg"] + ipol)), (npol, nxyz)
        )  # Pg vars
        Hpol = sparse((2 * polycf[:, 0], (arange(npol), arange(npol))), (npol, npol))
    else:
        Npol = None
        Hpol = None
    Cpol = polycf[:, 1]
    fparm_pol = ones((npol, 1)) * array([[1, 0, 0, 1]])

    ## combine with user costs
    NN = vstack([n for n in [Npwl, Npol, N] if n is not None and n.shape[0] > 0], "csr")
    # FIXME: Zero dimension sparse matrices.
    if (Hpwl is not None) and any_pwl and (npol + nw):
        Hpwl = hstack([Hpwl, sparse((any_pwl, npol + nw))])
    if Hpol is not None:
        if any_pwl and npol:
            Hpol = hstack([sparse((npol, any_pwl)), Hpol])
        if npol and nw:
            Hpol = hstack([Hpol, sparse((npol, nw))])
    if (H is not None) and nw and (any_pwl + npol):
        H = hstack([sparse((nw, any_pwl + npol)), H])
    HHw = vstack(
        [h for h in [Hpwl, Hpol, H] if h is not None and h.shape[0] > 0], "csr"
    )
    CCw = r_[Cpwl, Cpol, Cw]
    ffparm = r_[fparm_pwl, fparm_pol, fparm]

    ## transform quadratic coefficients for w into coefficients for X
    nnw = any_pwl + npol + nw
    M = sparse((ffparm[:, 3], (range(nnw), range(nnw))))
    MR = M * ffparm[:, 1]
    HMR = HHw * MR
    MN = M * NN
    HH = MN.T * HHw * MN
    CC = MN.T * (CCw - HMR)
    C0 = 0.5 * dot(MR, HMR) + sum(polycf[:, 2])  # Constant term of cost.

    Varefs = bus[bus[:, BUS_TYPE] == REF, VA] * (pi / 180.0)

    lb, ub = xmin.copy(), xmax.copy()
    lb[xmin == -Inf] = -1e10  ## replace Inf with numerical proxies
    ub[xmax == Inf] = 1e10
    x0 = (lb + ub) / 2
    # angles set to first reference angle
    x0[vv["i1"]["Va"] : vv["iN"]["Va"]] = Varefs[0]
    if ny > 0:
        ipwl = find(gencost[:, MODEL] == PW_LINEAR)
        # largest y-value in CCV data
        c = gencost.flatten("F")[
            sub2ind(gencost.shape, ipwl, NCOST + 2 * gencost[ipwl, NCOST])
        ]
        x0[vv["i1"]["y"] : vv["iN"]["y"]] = max(c) + 0.1 * abs(max(c))

    ###a
    alg = ppopt["OPF_ALG_DC"]
    opt = {"alg": alg, "verbose": False}
    ## set up options
    feastol = ppopt["PDIPM_FEASTOL"]
    gradtol = ppopt["PDIPM_GRADTOL"]
    comptol = ppopt["PDIPM_COMPTOL"]
    costtol = ppopt["PDIPM_COSTTOL"]
    max_it = ppopt["PDIPM_MAX_IT"]
    max_red = ppopt["SCPDIPM_RED_IT"]
    if feastol == 0:
        feastol = ppopt["OPF_VIOLATION"]  ## = OPF_VIOLATION by default
    opt["pips_opt"] = {
        "feastol": feastol,
        "gradtol": gradtol,
        "comptol": comptol,
        "costtol": costtol,
        "max_it": max_it,
        "max_red": max_red,
        "cost_mult": 1,
    }

    for i in range(len(l)):
        if np.isinf(l[i]):
            l[i] = -1e10
    # sanity check with pandapower solver
    x, f, info, output, lmbda = pp.pypower.qps_pypower.qps_pypower(
        None, CC, A, l, u, xmin, xmax, x0, opt
    )
    A = np.array(A.todense())
    # eliminating Va's via Pmis equality constraints
    i1_eq = om.lin["idx"]["i1"]["Pmis"]
    iN_eq = om.lin["idx"]["iN"]["Pmis"]
    b_Pmis_eq = np.array(np.real(om.lin["data"]["l"]["Pmis"]))[1:]
    k1_Va = om.var["idx"]["i1"]["Va"]
    kN_Va = om.var["idx"]["iN"]["Va"]
    k1_Pg = om.var["idx"]["i1"]["Pg"]
    kN_Pg = om.var["idx"]["iN"]["Pg"]
    # In the following code the following happens
    # Let us denote inequalities as
    # A_{Pmis} (Va Pg)^T = b_{Pmis} (in code it is as double inequalities with equal lower and upper bounds) (*)
    # Next, A_{Pmis} = [AVa_{Pmis} APg_{Pmis}]  - horizontal stack
    # Thus, (*) can be rewritten as follows
    # (AVa_{Pmis} Va) +  (APg_{Pmis} Pg) = b_{Pmis}
    # From here, we can solve for Va - AVa_{Pmis} is invertible
    # Va = (AVa_{Pmis})^{-1} (b_{Pmis} - (APg_{Pmis} Pg))

    # NB: the reason for dropping reference bus is rank!
    # including first constraint and Va[0] will make not full rank compromising ability to take inverse

    # No, I am assigning 0 to be slack, but, e.g., for pegase 1354 reference one is 369
    # see np.argmax(bus[:, BUS_TYPE] == REF)
    A_Va_Pmis = np.array(
        np.real(om.lin["data"]["A"]["Pmis"][:, k1_Va:kN_Va].todense())
    )[1:, 1:]

    A_Va_Pmis_inv = np.linalg.inv(A_Va_Pmis)
    A_Pg_Pmis = np.array(
        np.real(om.lin["data"]["A"]["Pmis"][:, k1_Pg:kN_Pg].todense())
    )[1:, :]
    A_Va_inv_A_Pg = np.dot(A_Va_Pmis_inv, A_Pg_Pmis)
    A_Va_inv_b_eq = np.dot(A_Va_Pmis_inv, b_Pmis_eq)

    left_ineqs = om.lin["order"][:]
    left_ineqs.remove("Pmis")
    A_new = np.zeros((om.lin["N"], om.var["idx"]["N"]["Pg"]))

    # Here equalities are dropped and the expression for Va from Pmis is being inserted into inequalities
    # l <= [A_{Va} A_{Pg}] (Va Pg)^T <= u
    # now substitute Va = (AVa_{Pmis})^{-1} (b_{Pmis} - (APg_{Pmis} Pg))
    # l - AVa_{Pmis})^{-1} b_{Pmis} <= (A_{Pg} - A_{Va} * AVa_{Pmis})^{-1} * (APg_{Pmis}) Pg
    # <= u - AVa_{Pmis})^{-1} b_{Pmis}
    A_Va = A[iN_eq:, k1_Va + 1 : kN_Va]
    A_Pg = A[iN_eq:, k1_Pg:kN_Pg]
    A_new_ = A_Pg - np.dot(A_Va, A_Va_inv_A_Pg)
    u_new_ = u[iN_eq:] - np.dot(A_Va, A_Va_inv_b_eq)
    l_new_ = l[iN_eq:] - np.dot(A_Va, A_Va_inv_b_eq)

    # load gen balance (sum loads = sum gens) - it is the first constraint in A
    # the same substituion as previous
    # the purpose of this part is to enforce balancing: Pg[0] = sum loads - sum_{i != 0} Pg[i]
    A_ref_eq = (
        -A[0, k1_Va + 1 : kN_Va].dot(A_Va_Pmis_inv).dot(A_Pg_Pmis) + A[0, k1_Pg:kN_Pg]
    )
    ul_ref_eq = u[0] - A[0, k1_Va + 1 : kN_Va].dot(A_Va_Pmis_inv).dot(
        b_Pmis_eq
    )  # .reshape(1,-1)
    # Now, having got Pg[0] = A_ref_eq[:, 1:] Pg[1:] + u[0] - A[0, k1_Va + 1 : kN_Va].dot(A_Va_Pmis_inv).dot(b_Pmis_eq)
    # We simply substitute Pg[0] with this in all of the inequalities
    A_new = (
        -A_new_[:, 0].reshape(-1, 1).dot(A_ref_eq[1:].reshape(1, -1)) / A_ref_eq[0]
        + A_new_[:, 1:]
    )
    u_new = u_new_ - A_new_[:, 0] * (ul_ref_eq / A_ref_eq[0])
    l_new = l_new_ - A_new_[:, 0] * (ul_ref_eq / A_ref_eq[0])
    # additional gen lim (mitigated from Pg[0] elimination)
    # Now we must add generation limita that were imposed for Pg[0], now they take form of linear combination of Pg[1:]
    A_new = np.vstack((A_new, -A_ref_eq[1:].reshape(1, -1) / A_ref_eq[0]))
    u_new = np.hstack((u_new, xmax_pp[k1_Pg] - ul_ref_eq / A_ref_eq[0]))
    l_new = np.hstack((l_new, xmin_pp[k1_Pg] - ul_ref_eq / A_ref_eq[0]))
    # adding generation limits
    for i in range(kN_Pg - (k1_Pg + 1)):
        unit_vec = np.zeros(kN_Pg - (k1_Pg + 1))
        unit_vec[i] = 1.0
        lower = xmin_pp[i + k1_Pg + 1]
        upper = xmax_pp[i + k1_Pg + 1]
        A_new = np.vstack((A_new, unit_vec.reshape(1, -1)))
        u_new = np.hstack((u_new, upper))
        l_new = np.hstack((l_new, lower))

    # Transform lower bounds into upper bounds
    A_full_new = np.vstack((A_new, -A_new))
    u_full_new = np.hstack((u_new, -l_new))

    # changes in cost due to the load gen balance!
    # Again, we must substitute expression Pg[0] into cost function now
    # c_new = CC[k1_Pg+1:kN_Pg]
    c_new = CC[k1_Pg + 1 : kN_Pg] - CC[k1_Pg] / A_ref_eq[0] * A_ref_eq[1:]
    c_correction_term = (
        CC[k1_Pg] * ul_ref_eq / A_ref_eq[0]
    )  # arises due to the elimination of gen = demand constraint

    if check_pp_vs_new_form:
        res_hands_new_x, status_sol = SU.solve_glpk(
            A_full_new, u_full_new, c_new, x0_pp[k1_Pg + 1 : kN_Pg]
        )
        res_hands_new_x = res_hands_new_x.flatten()
        res_hands_new_fun = np.dot(c_new, res_hands_new_x)
        # sanity check continuation
        all_gens_restorated = np.array(
            [(ul_ref_eq - A_ref_eq[1:].dot(res_hands_new_x)) / A_ref_eq[0]]
            + list(res_hands_new_x)
        )
        print(
            "sol from pandapower proximity:",
            np.max(x[k1_Pg:kN_Pg] - all_gens_restorated),
        )

        print(
            "cost from this method:",
            res_hands_new_fun + CC[k1_Pg] * ul_ref_eq / A_ref_eq[0],
        )
    print("cost from pandapower:", f)
    return A_full_new, u_full_new, x[k1_Pg:kN_Pg], c_new, c_correction_term, f


def get_linear_constraints_(grid_name, max_angle):
    """The function produces matrix A and vector b of linear constraint of DC OPF approximation

    Args:
        grid_name (string): grid name see `grid_name_map` above
        cov_std (float): variance of Gaussian fluctuations
        max_angle (float): max angle difference between buses
        shf_eps (float): thresold on shift significance in DC-PF Eqs pwr = pwr_shf + np.real(bbus)*va

    Returns:
        A_n (ndarray(lines, buses)): matrix of linear constraints
        b_n (ndarray(lines,)): vector of linear constraints
        gens (ndarray(buses,)): generator values
        cost_coeffs (ndarray(buses,))
    """
    net = grid_name_map[grid_name]()
    pp.rundcpp(net)
    ppc = net["_ppc"]
    m = net.line["to_bus"].size
    n = net.res_bus["p_mw"].size
    ### controllable are gens
    controllable_idxs = net.gen["bus"].values
    # Construct adjacency matrix
    adj = np.zeros((2 * m, n))
    for i in range(0, m):
        adj[i, net.line["to_bus"][i]] = 1
        adj[i, net.line["from_bus"][i]] = -1
        adj[i + m, net.line["to_bus"][i]] = -1
        adj[i + m, net.line["from_bus"][i]] = 1

    ### DC power flow equations have a form:
    ### pwr = pwr_shf + np.real(bbus)*va
    ### (compute all parameters)

    bbus = np.real(ppc["internal"]["Bbus"])
    va = math.pi * net.res_bus["va_degree"] / 180
    pwr = -net.res_bus["p_mw"]
    # print("va:", va)

    ### Phase angle differences:
    ### va = pinv(bbus)*(pwr - pwr_shf)
    ### va_d = adj*va = adj*pinv(bbus)*(pwr - pwr_shf)
    ### va_d = pf_mat*pwr - va_shf

    bbus_pinv = pinvh(bbus.todense())
    pf_mat = adj @ bbus_pinv

    ### slack bus
    ### !NB! ext_grid is an EXTERNAL bus, it does not presented in the grid!
    ### thus, slack is to be set differently, e.g., as the generator with maximum capacity
    ### moreover, we should put the sum of demands into the right hand sight of neg sum of generators -- it is
    ###   P_s + \sum_i (P_{g_i} + \xi) - P_{d_i}  = 0 => P_s^{min} \leq -(\sum_i (P_{g_i} + \xi) - P_{d_i}) \leq P_s^{max}
    # slck = net.ext_grid['bus']
    max_capacity_loc = net["gen"]["max_p_mw"].argmax()
    slck = max_capacity_loc
    slck_mat = np.eye(n)

    slck_mat[controllable_idxs[slck]] = -1  ## assign values to the whole array
    slck_mat[
        controllable_idxs[slck], controllable_idxs[slck]
    ] = 0  # and zero out for the slack itself

    ### set fluctuating components: either loads or gens or both
    ###

    loads = np.zeros(n)
    gens = np.zeros(n)
    ctrls = np.zeros(n)  ## controllable loads + gens

    loads[net.load["bus"]] = -net.res_load["p_mw"]
    gens[net.gen["bus"]] = net.res_gen["p_mw"]
    ctrls = loads + gens

    ### assume only loads are fluctuating

    xi = loads

    ### Assembling matrix
    ### A, -A for phase angle differences maximum and minimum
    ### slck_mat, -slck_mat for maximum and minimum generation
    A = pf_mat @ slck_mat
    A = np.vstack(
        (
            A[:, controllable_idxs],
            -A[:, controllable_idxs],
            slck_mat[np.ix_(controllable_idxs, controllable_idxs)],
            -slck_mat[np.ix_(controllable_idxs, controllable_idxs)],
        )
    )

    ### Assembling vector
    ### ones x max_angle is for maximum and minimum values of phase angle difference
    p_upper = np.zeros(n)
    p_upper[net.gen["bus"]] = net.gen["max_p_mw"]
    ### balancing for the slack bus
    p_upper[controllable_idxs[slck]] += +sum(loads)
    p_lower = np.zeros(n)
    p_lower[net.gen["bus"]] = net.gen["min_p_mw"]
    ### balancing for the slack bus
    p_lower[controllable_idxs[slck]] += sum(loads)
    b = np.hstack(
        (
            (np.ones(2 * m) * max_angle),
            (np.ones(2 * m) * max_angle),
            p_upper[controllable_idxs],
            -p_lower[controllable_idxs],
        )
    )

    assert A.shape[0] == len(b)

    ### find linear cost fuction coefficients
    cost_coeffs = np.zeros(n)
    cost_coeffs[net.gen["bus"]] = net.poly_cost.cp1_eur_per_mw[1:]
    cost_coeffs = cost_coeffs[controllable_idxs]
    ### check feasibility of gens
    gens = gens[controllable_idxs]

    # removing slack bus from the variables

    gens = np.hstack((gens[:slck], gens[slck + 1 :]))
    cost_coeffs = np.hstack((cost_coeffs[:slck], cost_coeffs[slck + 1 :]))
    controllable_idxs = np.hstack(
        (controllable_idxs[:slck], controllable_idxs[slck + 1 :])
    )
    A = np.hstack((A[:, :slck], A[:, slck + 1 :]))

    print("Number of constraints violated:", (A @ gens - b > 0).sum())

    ##### Assess equations feasibility #######
    ### Power balance check

    print("The RHS (phase angle diff max) = ", max_angle)

    return A, b, gens, cost_coeffs, controllable_idxs, slck
