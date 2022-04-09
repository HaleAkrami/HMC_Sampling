This is the starter code for homework 2 (Sampling). 

This provides some utility functions for getting started and is based on JAX. 
JAX makes it easy to compute fn and its grad (see `main.py`), but you can use any other framework.
If you run into problem with other frameworks/library, please reach out to me. 
The discrete -> continuous density code provided in `utils/density.py` is just one way to convert discrete to continuous. 
Using any other approach for this is also fine. 

## Setup

- See `requirements.txt` for required packages. Higher version of packages should also work
- Tested with python3.8
- We have provided NPEET as a submodule. To fetch the NPEET code correctly, you may have to pass `--recurse_submodule` flag with git pull/clone command.

## How to use?
To run code for GMM run 
```
python main_GMM.py
```
which gives you sampling for GMM using NUTS and HMC for different L and epsilon( the L and epsilon is hard-coded) 
![plot](./directory_1/directory_2/.../directory_n/plot.png)

