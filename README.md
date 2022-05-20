# :earth_africa: Flood Disaster Prediction :ocean:
Due to recent *climate change* related issues critical floods of urban areas have started becoming more
and more common. Hence, a software able to estimate the likelihood of such an event could be a useful
tool in order to prevent the most dramatic scenarios.
For this reason we proposed the implementation of a *Bayesian Network* to model the likelihood of
floods in various municipalities of the Italian region Veneto. The structure of the network was inspired
by the paper [_Assessing urban flood disaster risk using Bayesian network model and GIS applications_](https://www.tandfonline.com/doi/full/10.1080/19475705.2019.1685010), although quite deeply modified for didactic reasons.
The project was developed in *Python* using the `pgmpy` library.

## Repository structure

    .
    ├── code
    │   ├── data                       
    │   │   ├── 05_Veneto_Allegato-statistico.xlsx    # Data regarding general statistics on the region Veneto
    │   │   ├── Elab_Altimetrie_DEM.xlsx              # Data regarding the elevation of the italian municipalities
    │   │   ├── Redditi_e_principali_variabili...     # Data regarding the Italian population economy
    │   │   └── codiceISTAT_schedaLR14_2017.ods       # Data regarding general statistics on the territory of the region Veneto
    │   ├── Flood Disaster Prediction.ipynb           # Notebook containing the execution of the project
    │   ├── extended_classes.py                       # Script extending the classes BayesianNetwork and ApproxInference
    │   ├── graphics.py                               # Scripts containing graphical utils functions
    │   ├── utils.py                                  # Script containing utils functions
    │   └── variables.py                              # Variable nodes of the Bayesian Network
    ├── report
    │   └── Flood disaster prediction.pdf             # Report about the project 
    ├── .gitignore                             
    ├── LICENSE
    └── README.md

## Versioning

Git is used for versioning.

## Group members

|  Name     |  Surname  |     Email                              |    Username                                             |
| :-------: | :-------: | :------------------------------------: | :-----------------------------------------------------: |
| Antonio   | Politano  | `antonio.politano2@studio.unibo.it`    | [_S1082351_](https://github.com/S1082351)               |
| Francesco | Pieroni   | `francesco.pieroni3@studio.unibo.it`   | [_HumidBore_](https://github.com/HumidBore)             |
| Riccardo  | Spolaor   | `riccardo.spolaor@studio.unibo.it`     | [_RiccardoSpolaor_](https://github.com/RiccardoSpolaor) |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

Social preview image licensed by [Flood Vectors by Vecteez](https://www.vecteezy.com/free-vector/flood)
