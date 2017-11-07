# MORSE
MORSE is a code that infers equation of state (EoS) parameters from a joint posterior distribution on neutron star masses and radii. 
Effectively there are two parts to the code:
 - The first part calculates a mass-radius curve given an EoS by solving the relativistic stellar structure equations (TOV-     equations)
 - The second part transforms a joint posterior distribution on neutron star masses and radii to a posterior distribution on EoS parameters in a Bayesian framework. 

In the "Example" Ipython Notebook we show how to use MORSE to calculate mass, radius and moment of inertia from a set of EoS parameters and how to create input joint posteriors on mass and radius. 

To infer a posterior distribution on EoS parameters, the following codes have to be executed: 
- python Generate_Params_Par.py -n 20 
  The "-n 20" means that for each of the EoS parameters, 20 different values are calculated. 
- python Create_MRcurves_Par.py -f1 MRIcurves20 -f2 Parameters20
  This will create the mass-radius curves and the corresponding parameters and saves them as "MRIcurves20.npy" and     "Parameters20.npy" respectively.
  
