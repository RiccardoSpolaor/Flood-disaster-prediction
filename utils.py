import numpy as np
import pandas as pd
from io import StringIO
from pgmpy.factors.discrete.CPD import TabularCPD
from typing import Dict, List, Optional


def __get_state_names(variable: str, state_names_dictionary: Dict[str, List[str]],
                      evidence: Optional[List[str]]) -> Dict[str, List[str]]:
    if evidence is None:
        variable_and_evidence_list = [variable]
    else:
        variable_and_evidence_list = [variable] + evidence

    return dict((v, state_names_dictionary[v]) for v in variable_and_evidence_list if v in state_names_dictionary)


def __get_evidence_card(state_names_dictionary: Dict[str, List[str]],
                        evidence: Optional[List[str]]) -> Optional[List[int]]:
    if evidence is None:
        return None
    else:
        return [len(state_names_dictionary[e]) for e in evidence]


def get_tabular_cpd(variable: str, state_names_dictionary: Dict[str, List[str]],
                    values_dictionary: Dict[str, List[List[float]]],
                    evidence_dictionary: Dict[str, List[str]]) -> TabularCPD:
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
    variable = cpd.variable

    evidence = cpd.get_evidence()
    evidence.reverse()

    state_names = cpd.state_names

    if len(evidence) == 0:
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


def apply_discrete_values(variable: str, df: pd.DataFrame, quantiles: List[float],
                          state_names_dictionary: Dict[str, List[str]]) -> None:
    state_names = state_names_dictionary[variable]
    state_names = state_names.copy()
    state_names.reverse()

    df[variable] = pd.cut(x=df[variable],
                          bins=quantiles,
                          labels=state_names,
                          include_lowest=True
                          )
