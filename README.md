# COURSEWORK Alberto Plebani (ap2387)

README containing instructions on how to run the code for the coursework.

The repository can be cloned with 
```shell
git clone git@gitlab.developers.cam.ac.uk:phy/data-intensive-science-mphil/s2_assessment/ap2387.git
```

# Anaconda 

The conda environment can be created using the [environment.yml](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/S2_Assessment/ap2387/-/blob/main/environment.yml?ref_type=heads) file, which contains all the packages needed to run the code:
```shell
conda env create --name CDT_ML --file environment.yml
```

# Report

The final report is presented in [ap2387.pdf](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/S2_Assessment/ap2387/-/blob/main/report/ap2387.pdf?ref_type=heads). The file is generated using LaTeX, but all LaTeX-related files are not being committed as per the instructions on the [.gitignore](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/S2_Assessment/ap2387/-/blob/main/.gitignore?ref_type=heads) file.


# Code structure

The codes to run the exercises can be found in the ```src``` folder, whereas the file [Helpers/HelperFunctions.py](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/S2_Assessment/ap2387/-/blob/main/Helpers/HelperFunctions.py?ref_type=heads) contains the definition of the classes used in the code.

The code can be run with multiple ```parser``` options, which can be accessed with the following command
```shell
python src/main.py -h
```

The parser options are the following:
- ```--plots```: If selected, this flag displays the plots instead of only saving them
- ```-n, --nsamples```: Number (int) of samples to be generated in the Markov Chain. By default this number is set to 10000, because running for 500000 takes roughly 5 minutes
- ```-b, --burnin```: Number (int) of samples to be considered for the burn-in. By default this number is set to 1000.
- ```--a_range```: List of 2 space-separated values (float) for the range of alpha, which defines in which range the prior distribution for alpha is uniform. The two parameters must be in increasing order. By default, these values are set to -2 and 2
- ```b_range```: : List of 2 space-separated values (float) for the range of beta, which defines in which range the prior distribution for alpha is uniform. The two parameters must be in increasing order, with the first one which needs to be positive. By default, these values are set to 0.1 and 4.
- ```-s, --starting```: List of 3 space-separated values (float) which are the starting points for alpha, beta and I0 in the chain. By default, all three values are set to 0.3
- ```-i, --intensity```: Mean (float) of the log-I distribution for I0. By default, this parameter is set to 1.1
- ```-c, --custom_name```: Custom name to give to the plots when saving them, to be used to run different tests and change configurations

The plots will be saved in the ```Plots``` folder.