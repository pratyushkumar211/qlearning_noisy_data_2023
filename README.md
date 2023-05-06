This repository contains source code for the simulation studies presented in the paper

Unconstrained feedback controller design using Q-learning from noisy process data
Pratyush Kumar, James B. Rawlings
Under review, Computers and Chemical Engineering

To reproduce the results, run the following python scrips in order:- 

% Perfom the calculations.

python hvac_parameters.py 

python hvac_sfeed_optlqg.py 

python hvac_sfeed_lspi_kqw.py 

python hvac_sfeed_lspi_uqw.py 

python hvac_sfeed_lspi_regl.py 

python hvac_sfeed_sysid.py 

python hvac_ofeed_optlqg.py

python hvac_ofeed_lspi.py

python hvac_ofeed_sysid.py

% Make the plots.

python hvac_sfeed_traindata_plots.py 

python hvac_sfeed_clanalysis_plots.py 

python hvac_sfeed_dataeff_plots.py 

python hvac_ofeed_traindata_plots.py 

python hvac_ofeed_clanalysis_plots.py 

python hvac_ofeed_dataeff_plots.py 

Feel free to contact Pratyush Kumar (pratyushkumar211@gmail.com) for any questions regarding the source code.
