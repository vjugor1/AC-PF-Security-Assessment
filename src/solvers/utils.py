from black import out
import numpy as np
import cvxpy as cp


import sys

sys.path.append("..")
import os

# print(os.listdir())
# from src.samplers import utils as sampling
# from src.solvers import scenario_approx as SA


def solve_glpk(
    scenario_Gamma: np.ndarray, scenario_Beta: np.ndarray, c: np.ndarray, x0: np.ndarray
):
    """Solves linear program
        min c^T x
        x
        s.t. scenario_Gamma.dot(x) <= scenario_Beta

    Args:
        scenario_Gamma (np.ndarray): matrix of linear constraints - upper bounds
        scenario_Beta (np.ndarray): vector of linear constraints - upper bounds
        c (np.ndarray): cost function vector
        x0 (np.ndarray): initial guess

    Returns:
        tuple: solution and solution status from GLPK
    """
    x = cp.Variable(scenario_Gamma.shape[1])
    x.value = x0
    obj = cp.Minimize(cp.sum(cp.multiply(x, c)))
    constraints = [
        cp.sum(cp.multiply(scenario_Gamma[i], x)) <= scenario_Beta[i]
        for i in range(scenario_Gamma.shape[0])
    ]
    prob = cp.Problem(obj, constraints)
    prob.solve(
        solver=cp.GLPK, verbose=False, glpk={"msg_lev": "GLP_MSG_OFF"}, warm_start=True
    )
    # print("opt val = ", prob.value)
    sol = prob.solution.primal_vars
    x_opt = np.array(list(sol.values()))
    return x_opt, prob.solution.status


def solve_approximations(
    Gamma: np.ndarray,
    Beta: np.ndarray,
    A: np.ndarray,
    N: int,
    c: np.ndarray,
    eta: float,
    x0: dict,
    optimize_samples: bool,
):
    """Utils function that solves with N scenarios
        1) Ordinary scenario approx
        2) Scenario approx with slack constraints
        3) SAIMIN scenario approx
        for
        min c^T x
        x
        s.t. Prob{Gamma.dot(x) + A.dot(chi) <= Beta} >= 1 - eta
    Args:
        Gamma (np.ndarray): Matrix of linear inequalities - upper bound
        Beta (np.ndarray): Vector of linear inequalities - upper bound
        A (np.ndarray): Matrix before standard random vector
        N (int): Number of samples
        c (np.ndarray): Cost function vector
        eta (float): Reliability level
        x0 (dict): initial guesses for methods - use for warm start
        optimize_samples (bool): for each unique normal in Gamma, keeps only the constant Beta that can be potentially active -- exclude planes that are 100% inactive

    Returns:
        dict: dictionary of optimal points and their statuses
    """
    # Gamma, Beta, A, N, c, eta, x0 = args
    assert np.allclose(
        np.linalg.norm(A, axis=1), np.ones(A.shape[0])
    ), "Planes (chi - random vector) normal vectors must have unit lengths"
    # Scenario approx with slack constraints O
    samples = np.random.multivariate_normal(
        np.zeros(A.shape[1]), np.eye(A.shape[1]), size=N
    )
    SCSA_Gamma, SCSA_Beta = SA.get_scenario_approx_constraints(
        Gamma=Gamma,
        Beta=Beta,
        A=A,
        samples=samples,
        optimize_samples=optimize_samples,
        include_slack=True,
        eta=eta,
    )
    SCSA_sol, SCSA_status = solve_glpk(SCSA_Gamma, SCSA_Beta, c, x0["SCSA"])
    # Ordinary scenario approx
    SCSA_Gamma_no_O, SCSA_Beta_no_O = SA.get_scenario_approx_constraints(
        Gamma=Gamma,
        Beta=Beta,
        A=A,
        samples=samples,
        optimize_samples=optimize_samples,
        include_slack=False,
        eta=eta,
    )
    SCSA_noO_sol, SCSA_noO_status = solve_glpk(
        SCSA_Gamma_no_O, SCSA_Beta_no_O, c, x0["SCSA_no_O"]
    )
    # SAIMIN
    samples_SAIMIN = sampling.get_samples_SAIMIN(N, eta, len(Beta), A)
    SAIMIN_Gamma, SAIMIN_Beta = SA.get_scenario_approx_constraints(
        Gamma=Gamma,
        Beta=Beta,
        A=A,
        samples=samples_SAIMIN,
        optimize_samples=optimize_samples,
        include_slack=False,
        eta=eta,
    )
    SAIMIN_sol, SAIMIN_status = solve_glpk(SAIMIN_Gamma, SAIMIN_Beta, c, x0["SAIMIN"])

    out_dict = {
        "SCSA": [SCSA_sol.flatten(), SCSA_status],
        "SCSA_no_O": [SCSA_noO_sol.flatten(), SCSA_noO_status],
        "SAIMIN": [SAIMIN_sol.flatten(), SAIMIN_status],
    }
    # json compatibility
    for k__ in out_dict.keys():
        out_dict[k__][0] = [float(v) for v in out_dict[k__][0]]
    status = True
    for v in out_dict.values():
        status = status and v[1]
    assert status
    return out_dict
