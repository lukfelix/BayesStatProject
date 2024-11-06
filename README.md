# BayesStatProject
This repository contains the group project created for the course "402-0738-10L  Bayesian Statistical Methods and Data Analysis" in the fall semester of 2024 at ETH.
[Link to course information](https://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=186258&semkez=2024W&ansicht=ALLE&lang=de)

Our project is based on recent work by [Coulombe et al 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240903812C/abstract) and aims to reproduce their result and further explore its implications.

Contributors to the project are:\
Céline Nussbaumer\
Maximilian Reiter\
Ryan Meierhofer\
Kevin Kohli\
and Lukas Felix

[View-only link to the report on Overleaf](https://www.overleaf.com/read/kmzwjsrvgrmv#078b9e)

All necessary packages can be installed e.g. with pip using:
```
pip install numpy matplotlib emcee batman-package corner tqdm
```

# FILESTRUCTURE

"analysis.py" contains the entire analysis, from simulation, to modelling, all the way to mcmc. It is the "mother file". 

The other files contain certain functions which are being used via the analysis.py file. Those are all the "children files" and do not perform anything on their own. They are simply placeholders for certain functions. 