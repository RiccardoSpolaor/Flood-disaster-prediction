from pgmpy.models import BayesianNetwork
from pgmpy.inference import ApproxInference
from pgmpy.factors.discrete import TabularCPD

import numpy as np


class ExtendedBayesianNetwork(BayesianNetwork):
    def simulate_by_weighted_likelihood(
        self,
        n_samples=10,
        do=None,
        evidence=None,
        virtual_evidence=None,
        virtual_intervention=None,
        include_latents=False,
        partial_samples=None,
        seed=None,
        show_progress=True,
    ):
        """
        Simulates data from the given model. Internally uses methods from
        pgmpy.sampling.Sampling.likelihood_weighted_sample to generate the data.

        Parameters
        ----------
        n_samples: int
            The number of data samples to simulate from the model.

        do: dict
            The interventions to apply to the model. dict should be of the form
            {variable_name: state}

        evidence: dict
            Observed evidence to apply to the model. dict should be of the form
            {variable_name: state}

        virtual_evidence: list
            Probabilistically apply evidence to the model. `virtual_evidence` should
            be a list of `pgmpy.factors.discrete.TabularCPD` objects specifying the
            virtual probabilities.

        virtual_intervention: list
            Also known as soft intervention. `virtual_intervention` should be a list
            of `pgmpy.factors.discrete.TabularCPD` objects specifying the virtual/soft
            intervention probabilities.

        include_latents: boolean
            Whether to include the latent variable values in the generated samples.

        partial_samples: pandas.DataFrame
            A pandas dataframe specifying samples on some of the variables in the model. If
            specified, the sampling procedure uses these sample values, instead of generating them.
            partial_samples.shape[0] must be equal to `n_samples`.

        seed: int (default: None)
            If a value is provided, sets the seed for numpy.random.

        show_progress: bool
            If True, shows a progress bar when generating samples.

        Returns
        -------
        A dataframe with the simulated data: pd.DataFrame

        Examples
        --------
        >>> from pgmpy.utils import get_example_model

        Simulation without any evidence or intervention:

        >>> model = get_example_model('alarm')
        >>> model.simulate(n_samples=10)

        Simulation with the hard evidence: MINVOLSET = HIGH:

        >>> model.simulate(n_samples=10, evidence={"MINVOLSET": "HIGH"})

        Simulation with hard intervention: CVP = LOW:

        >>> model.simulate(n_samples=10, do={"CVP": "LOW"})

        Simulation with virtual/soft evidence: p(MINVOLSET=LOW) = 0.8, p(MINVOLSET=HIGH) = 0.2,
        p(MINVOLSET=NORMAL) = 0:

        >>> virt_evidence = [TabularCPD("MINVOLSET", 3, [[0.8], [0.0], [0.2]], state_names={"MINVOLSET": ["LOW", "NORMAL", "HIGH"]})]
        >>> model.simulate(n_samples, virtual_evidence=virt_evidence)

        Simulation with virtual/soft intervention: p(CVP=LOW) = 0.2, p(CVP=NORMAL)=0.5, p(CVP=HIGH)=0.3:

        >>> virt_intervention = [TabularCPD("CVP", 3, [[0.2], [0.5], [0.3]], state_names={"CVP": ["LOW", "NORMAL", "HIGH"]})]
        >>> model.simulate(n_samples, virtual_intervention=virt_intervention)
        """
        from pgmpy.sampling import BayesianModelSampling

        self.check_model()
        model = self.copy()

        evidence = {} if evidence is None else evidence
        do = {} if do is None else do
        virtual_intervention = (
            [] if virtual_intervention is None else virtual_intervention
        )
        virtual_evidence = [] if virtual_evidence is None else virtual_evidence

        if set(do.keys()).intersection(set(evidence.keys())):
            raise ValueError("Variable can't be in both do and evidence")

        # Step 1: If do or virtual_intervention is specified, modify the network structure.
        if (do != {}) or (virtual_intervention != []):
            virt_nodes = [cpd.variables[0] for cpd in virtual_intervention]
            model = model.do(list(do.keys()) + virt_nodes)
            evidence = {**evidence, **do}
            virtual_evidence = [*virtual_evidence, *virtual_intervention]

        # Step 2: If virtual_evidence; modify the network structure
        if virtual_evidence != []:
            for cpd in virtual_evidence:
                var = cpd.variables[0]
                if var not in model.nodes():
                    raise ValueError(
                        "Evidence provided for variable which is not in the model"
                    )
                elif len(cpd.variables) > 1:
                    raise (
                        "Virtual evidecne should be defined on individual variables. Maybe your are looking for soft evidence."
                    )
                elif self.get_cardinality(var) != cpd.get_cardinality([var])[var]:
                    raise ValueError(
                        "The number of states/cardinality for the evideence should be same as the nubmer fo states/cardinalit yof teh variable in the model"
                    )

            for cpd in virtual_evidence:
                var = cpd.variables[0]
                new_var = "__" + var
                model.add_edge(var, new_var)
                values = np.vstack((cpd.values, 1 - cpd.values))
                new_cpd = TabularCPD(
                    variable=new_var,
                    variable_card=2,
                    values=values,
                    evidence=[var],
                    evidence_card=[model.get_cardinality(var)],
                    state_names={new_var: [0, 1], var: cpd.state_names[var]},
                )
                model.add_cpds(new_cpd)
                evidence[new_var] = 0

        # Step 3: If no evidence do a forward sampling
        if len(evidence) == 0:
            samples = BayesianModelSampling(model).forward_sample(
                size=n_samples,
                include_latents=include_latents,
                seed=seed,
                show_progress=show_progress,
                partial_samples=partial_samples,
            )

        # Step 4: If evidence; do a rejection sampling
        else:
            samples = BayesianModelSampling(model).likelihood_weighted_sample(
                size=n_samples,
                evidence=[(k, v) for k, v in evidence.items()],
                include_latents=include_latents,
                seed=seed,
                show_progress=show_progress,
            )

        # Step 5: Postprocess and return
        if include_latents:
            return samples
        else:
            return samples.loc[:, set(self.nodes()) - self.latents]


class ExtendedApproxInference(ApproxInference):
    def query(
            self,
            variables,
            n_samples=int(1e4),
            evidence=None,
            virtual_evidence=None,
            joint=True,
            show_progress=True,
            use_weighted_sampling=False,
    ):
        """
        Method for doing approximate inference based on sampling in Bayesian
        Networks and Dynamic Bayesian Networks.

        Parameters
        ----------
        variables: list
            List of variables for which the probability distribution needs to be calculated.

        n_samples: int
            The number of samples to generate for computing the distributions. Higher `n_samples`
            results in more accurate results at the cost of more computation time.

        evidence: dict (default: None)
            The observed values. A dict key, value pair of the form {var: state_name}.

        virtual_evidence: list (default: None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual/soft
            evidence.

        show_progress: boolean (default: True)
            If True, shows a progress bar when generating samples.

        sampling_type: str (default: 'weighted')
            The type of sampling to perform: ['weighted', 'negation'].

        Returns
        -------
        Probability distribution: pgmpy.factors.discrete.TabularCPD
            The queried probability distribution.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.inference import ApproxInference
        >>> model = get_example_model("alarm")
        >>> infer = ApproxInference(model)
        >>> infer.query(variables=["HISTORY"])
        <DiscreteFactor representing phi(HISTORY:2) at 0x7f92d9f5b910>
        >>> infer.query(variables=["HISTORY", "CVP"], joint=True)
        <DiscreteFactor representing phi(HISTORY:2, CVP:3) at 0x7f92d9f77610>
        >>> infer.query(variables=["HISTORY", "CVP"], joint=False)
        {'HISTORY': <DiscreteFactor representing phi(HISTORY:2) at 0x7f92dc61eb50>,
         'CVP': <DiscreteFactor representing phi(CVP:3) at 0x7f92d915ec40>}
        """
        # Step 1: Generate samples for the query
        if use_weighted_sampling:
            samples = self.model.simulate(
                n_samples=n_samples,
                evidence=evidence,
                virtual_evidence=virtual_evidence,
                show_progress=show_progress,
            )
        else:
            samples = self.model.simulate_by_weighted_likelihood(
                n_samples=n_samples,
                evidence=evidence,
                virtual_evidence=virtual_evidence,
                show_progress=show_progress,
            )


        # Step 2: Compute the distributions and return it.
        return self.get_distribution(samples, variables=variables, joint=joint)
