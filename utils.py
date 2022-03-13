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


def cpd_to_pandas(tabular_cpd: TabularCPD):
    # String representing the pretty print format of the PCD table
    tabulate_string = str(tabular_cpd)

    # Set the header according to the number of parents of the variable
    header: Optional[List[int]]
    if len(tabular_cpd.state_names) == 1:
        header = None
    else:
        header = [i for i in range(0, len(tabular_cpd.state_names) - 1)]

    # Create the Pandas Dataframe
    data = (pd.read_csv(StringIO(tabulate_string),
                        sep=r'\|',
                        comment='+',
                        engine='python',
                        skipinitialspace=True,
                        header=header).dropna(how='all', axis=1)
            )

    # Remove the column name if the variable has no parents
    if header is None:
        data.columns = [data.columns[0], '']

    # Turn the first column into the index of the Dataframe
    data.set_index(data.columns[0], inplace=True, drop=True)
    data.index.name = None

    return data
