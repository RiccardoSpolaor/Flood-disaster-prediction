from typing import List, Optional, Dict, Set, Union, Any

from matplotlib.lines import Line2D

from variables import *

from pgmpy.models import BayesianNetwork
import networkx as nx
import matplotlib.pyplot as plt

__node_positions = {
    PER_UNIT_GDP: (10, 12), ELEVATION: (22, 12), RAINFALL_FREQUENCY: (30, 12), RIVER_DENSITY: (42, 12),
    POPULATION_DENSITY: (2, 9),
    ROAD_DENSITY: (10, 6), SLOPE: (22, 6), RAINFALL_AMOUNT: (34, 6),
    FLOOD: (22, 0)
}

__GREEN = '#9BBB59'
__WHITE = 'white'
__BLUE = '#cccce6'
__YELLOW = '#ffffcc'


def __plot_bayesian_network(model: BayesianNetwork, title: str, color_map: List[str] = None,
                            legend: Optional[Dict[str, Any]] = None) -> None:
    """Plot a Bayesian Network.

    Parameters
    ----------
    model : BayesianNetwork
        Bayesian Network to plot.
    title : str
        Title of the plot.
    color_map : List[str] (default = None)
        Optional list of colors to assign to the respective nodes of the Network.
    color_map : List[str] (default = None)
        Optional list of colors to assign to the respective nodes of the Network. If None, the nodes are all filled with
        the color 'white'.
    legend : Optional[Dict[str, Any]] (default = None)
        Optional dictionary containing information to plot the legend. If None, the legend is not plotted.
    """

    plt.figure(figsize=(10, 6))

    node_names = {n: '\n'.join(n.split(' ')) for n in model.nodes()}

    if color_map is None:
        color_map = ['white'] * len(model.nodes)

    nx.draw(model, pos=__node_positions, node_size=5000, with_labels=True, labels=node_names,
            linewidths=1, node_color=color_map)

    # Set the color of the nodes edges as black
    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_edgecolor('black')

    if legend is not None:
        ax.legend(legend['handles'], legend['labels'], prop={'size': 15}, loc='lower right')

    plt.title(title)

    plt.show()


def plot_simple_bayesian_network(model: BayesianNetwork) -> None:
    """Plot a Bayesian Network without highlighted nodes or legends.

        Parameters
        ----------
        model : BayesianNetwork
            Bayesian Network to plot.
    """

    __plot_bayesian_network(model, "Flood Risk Bayesian Network")


def plot_markov_blanket(model: BayesianNetwork, variable: str, markov_blanket: List[str]) -> None:
    """Plot a Bayesian Network showing information about the Markov Blanket of a given node.

        Parameters
        ----------
        model : BayesianNetwork
            Bayesian Network to plot.
        variable : str
            Node for which the Markov Blanket is shown.
        markov_blanket : List[str]
            Markov Blanket of `variable`.
    """

    color_map = []
    legend = {
        'handles': [
            Line2D([0], [0], marker='o', color='black', label='Circle', markerfacecolor=__GREEN, markersize=15),
            Line2D([0], [0], marker='o', color='black', label='Circle', markerfacecolor=__YELLOW, markersize=15)
        ],
        'labels': ['Selected variable', 'Markov blanket node']

    }
    for node in model.nodes:
        if node == variable:
            color_map.append(__GREEN)
        elif node in markov_blanket:
            color_map.append(__YELLOW)
        else:
            color_map.append(__WHITE)

    __plot_bayesian_network(model, "Markov Blanket for variable: {}".format(variable), color_map, legend)


def plot_v_structure(model: BayesianNetwork, variable: str, evidence: Union[str, List[str]],
                     active_trail: Dict[str, Set[str]]) -> None:
    """Plot a Bayesian Network showing an active v-structure given a variable and a certain evidence.

        Parameters
        ----------
        model : BayesianNetwork
            Bayesian Network to plot.
        variable : str
            Given node.
        evidence : Union[str, List[str]]
            Evidence of `variable` that activates the v-structure.
        active_trail : Dict[str, Set[str]]
            Active trail of the v-structure.
    """

    if type(evidence) == str:
        evidence = [evidence]
    active_trail_variables = active_trail[variable]
    color_map = []
    legend = {
        'handles': [
            Line2D([0], [0], marker='o', color='black', label='Circle', markerfacecolor=__GREEN, markersize=15),
            Line2D([0], [0], marker='o', color='black', label='Circle', markerfacecolor=__YELLOW, markersize=15),
            Line2D([0], [0], marker='o', color='black', label='Circle', markerfacecolor=__BLUE, markersize=15)
        ],
        'labels': ['Selected variable', 'Evidence', 'Activated node']

    }
    for node in model.nodes:
        if node == variable:
            color_map.append(__GREEN)
        elif node in evidence:
            color_map.append(__YELLOW)
        elif node in active_trail_variables:
            color_map.append(__BLUE)
        else:
            color_map.append(__WHITE)

    __plot_bayesian_network(model, f'Trail of influence for variable {variable}', color_map, legend)
