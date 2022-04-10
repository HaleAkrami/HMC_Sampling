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
which gives you sampling for GMM using NUTS and HMC for different L and epsilon.
<img width="601" alt="Screen Shot 2022-04-09 at 7 46 29 PM" src="https://user-images.githubusercontent.com/25341241/162599035-4eaff375-4fc1-40b4-80e4-03a15c1b1bae.png">

