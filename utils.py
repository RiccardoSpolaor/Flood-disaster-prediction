import time

import numpy as np
import pandas as pd
from pgmpy.factors.discrete.CPD import TabularCPD
from typing import Dict, List, Optional

from pgmpy.inference.ExactInference import VariableElimination

from extended_classes import ExtendedApproxInference


def __get_state_names(variable: str, state_names_dictionary: Dict[str, List[str]],
                      evidence: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Return a subset of the dictionary of state names where keys correspond to the given `variable` of the  Bayesian
    Network and its evidence.

    Parameters
    ----------
    variable : str
        Variable of the Bayesian Network from which the subset of `state_names_dictionary` with the corresponding key is
        taken.
    state_names_dictionary : Dict[str, List[str]]
        Dictionary which keys are variables of the Bayesian Network and which values are the respective discrete states.
    evidence : Optional[List[str]] (default = None)
        Optional list of variables corresponding to the evidence of `variable` in the Bayesian Network.

    Returns
    -------
    Dict[str, List[str]]
        Subset of `state_names_dictionary` where keys correspond to the given `variable` of the Bayesian Network and its
        evidence.
    """

    if evidence is None:
        variable_and_evidence_list = [variable]
    else:
        variable_and_evidence_list = [variable] + evidence

    return dict((v, state_names_dictionary[v]) for v in variable_and_evidence_list if v in state_names_dictionary)


def __get_evidence_card(state_names_dictionary: Dict[str, List[str]],
                        evidence: Optional[List[str]] = None) -> Optional[List[int]]:
    """Return a list containing in each position the cardinality of the discrete states of the corresponding evidence
    variable in the list `evidence`.

    Parameters
    ----------
    state_names_dictionary : Dict[str, List[str]]
        Dictionary which keys are variables of the Bayesian Network and which values are the respective discrete states.
    evidence : Optional[List[str]] (default = None).
        Optional list of variables corresponding to the evidence of `variable` in the Bayesian Network.

    Returns
    -------
    Optional[List[int]]
        A list containing in each position the cardinality of the discrete states of the corresponding evidence variable
        in the list `evidence`. If the list `evidence` is equal to None, None is returned.
    """

    if evidence is None:
        return None
    else:
        return [len(state_names_dictionary[e]) for e in evidence]


def get_tabular_cpd(variable: str, state_names_dictionary: Dict[str, List[str]],
                    values_dictionary: Dict[str, List[List[float]]],
                    evidence_dictionary: Dict[str, List[str]]) -> TabularCPD:
    """Return the Conditional Probability Distribution (CPD) table of a variable according to its parent nodes in the
    Bayesian Network.

    Parameters
    ----------
    variable : str
        Variable of the Bayesian Network for which the CPD table is computed.
    state_names_dictionary : Dict[str, List[str]]
        Dictionary which keys are variables of the Bayesian Network and which values are the respective discrete states.
    values_dictionary : Dict[str, List[List[float]]]
        Dictionary which keys are variables of the Bayesian Network and which values are their respective conditional
        probability distribution values with respect to their evidence.
    evidence_dictionary : Dict[str, List[str]] (default = None).
        Dictionary which keys are variables of the Bayesian Network and which values are their respective evidence.

    Returns
    -------
    TabularCPD
        The CPD table of a variable according to its parent nodes in the Bayesian Network.
    """

    evidence: Optional[List[str]] = evidence_dictionary[variable]

    return TabularCPD(
        variable=variable,
        variable_card=len(state_names_dictionary[variable]),
        values=values_dictionary[variable],
        evidence=evidence,
        evidence_card=__get_evidence_card(state_names_dictionary, evidence),
        state_names=__get_state_names(variable, state_names_dictionary, evidence)
    )


def cpd_to_pandas(cpd: TabularCPD) -> pd.DataFrame:
    """Return the given Conditional Probability Distribution (CPD) table represented as a pandas DataFrame for
    readability.

    Parameters
    ----------
    cpd : TabularCPD
        A CPD table.

    Returns
    -------
    DataFrame
        `cpd` represented as a pandas DataFrame.
    """

    variable = cpd.variable

    evidence = cpd.get_evidence()
    evidence.reverse()

    state_names = cpd.state_names

    if not evidence:
        columns = [['']]
    else:
        columns = pd.MultiIndex.from_product(
            [['{} ({})'.format(e, n) for n in state_names[e]] for e in evidence]
        )

    values = cpd.values

    if values.ndim > 1:
        values = values.reshape(
            values.shape[0], (np.prod(np.array([i for i in values.shape[1:]])))
        )

    return pd.DataFrame(
        values,
        index=['{} ({})'.format(variable, n) for n in state_names[variable]],
        columns=columns
    )


def print_exact_inference(variable: str, infer: VariableElimination,
                          evidence: Optional[Dict[str, List[str]]] = None) -> None:
    """Print the exact inference table of a variable of a Bayesian Network given a specific evidence.

    Parameters
    ----------
    variable : str
        Variable of the Bayesian Network for which the exact inference on its discrete states is computed.
    infer : VariableElimination
        Object to apply exact inference on `variable` with the Variable Elimination method.
    evidence : Optional[Dict[str, List[str]]] (default: None)
        Dictionary which keys are evidence of `variable` in the Bayesian Network and which values are their selected
        state.
    """

    start_time = time.time_ns()

    evidence_str = '' if evidence is None else f" | {', '.join([f'{k} = {v}' for k, v in evidence.items()])}"

    print(f"Exact Inference to find P({variable}{evidence_str})\n")
    print(infer.query([variable], show_progress=False, evidence=evidence))
    print(f"----- {(time.time_ns() - start_time) / 1_000_000_000} seconds -----")


def print_approximate_inference(variable: str, infer: ExtendedApproxInference, n_samples=1_000,
                                evidence: Optional[Dict[str, List[str]]] = None, use_weighted_likelihood=False,
                                random_state: int = None) -> None:
    """Print the approximate inference table of a variable of a Bayesian Network given a specific evidence.

    Parameters
    ----------
    variable : str
        Variable of the Bayesian Network for which the approximate inference on its discrete states is computed.
    infer : ExtendedApproxInference
        Object to apply approximate inference on `variable` with the Rejection Sampling or Weighted Likelihood methods.
    n_samples : int (default: 1_000)
        Number of samples to use to compute the approximate inference.
    evidence : Optional[Dict[str, List[str]]] (default: None)
        Dictionary which keys are evidence of `variable` in the Bayesian Network and which values are their selected
        state.
    use_weighted_likelihood : bool (default: False)
        If true, sample by Weighted Likelihood. If false, sample by Rejection Sampling.
    random_state : int (default: None)
        Set a seed for deterministic results on the sampling.
    """

    start_time = time.time_ns()

    evidence_str = '' if evidence is None else f" | {', '.join([f'{k} = {v}' for k, v in evidence.items()])}"
    use_weighted_likelihood_str = 'rejection sampling' if not use_weighted_likelihood else 'weighted likelihood'

    print(f"Approximate Inference with {use_weighted_likelihood_str} to find P({variable}{evidence_str})\n")
    print(infer.query(variables=[variable], n_samples=n_samples, show_progress=False, seed=random_state,
                      evidence=evidence, use_weighted_sampling=use_weighted_likelihood))
    print(f"----- {(time.time_ns() - start_time) / 1_000_000_000} seconds -----")


def apply_discrete_values(variable: str, df: pd.DataFrame, quantiles: List[float],
                          state_names_dictionary: Dict[str, List[str]]) -> None:
    """Discretize the values of a column in a DataFrame according to its discrete state names and given their exact
    inference values.

    Parameters
    ----------
    variable : str
        Column of the DataFrame which its continuous values are discretized.
    df : DataFrame
        pandas Dataframe which column `variable` is discretized.
    quantiles : List[float]
        Quantiles used for the discretization of `variable`.
    state_names_dictionary : Dict[str, List[str]]
        Dictionary which keys are variables of the Bayesian Network and which values are the respective discrete states.
        It is used to select the states in which `variable` is discretized according to `quantiles`.
    """

    state_names = state_names_dictionary[variable]
    state_names = state_names.copy()
    state_names.reverse()

    df[variable] = pd.cut(x=df[variable],
                          bins=quantiles,
                          labels=state_names,
                          include_lowest=True
                          )
