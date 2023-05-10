# PhD-code

Functions written and used during the course of my PhD research in astrophysics at the University of Chicago. The code provided here is intended for educational purposes.

## Functions

This module collects miscellaneous functions that I found myself using somewhat frequently across my work. There are 5 packages within this module: `Kepler_solver`, `orbits`, `pdfs`, `stats_fitting`, and `LL_secular`. Descriptions and demonstrations of the functions in each of these packages are found in the iPython notebooks `Using the Functions Module` (first 4 pacakages) and `Using the Functions Module - LL_secular` (the last package).

## NP

This module collects miscellaneous functions I created while working on a paper about nodal precession (hence NP). There are 5 packages within this module: `eigenfrequencies` (this was later superseded by the LL_secular package), `functions` (all save 2 functions within were superseded by the LL_secular package), `invariable_plane`, `LW11`, and `MD4_derivation`. Descriptions and demonstrations of the functions in each of these packages are found in the iPython notebook `Using the NP Module`.

## Stability

This module contains only one package, `spectral_fraction`. Descriptions and demonstrations of the functions in this package are found in the iPython notebook `Using the Stability Module`.

## TTV

This module incorporates various capabilities relating to transit timing variations in 3 packages: `Hadden2019`, `REBOUND_TTV`, and `TTV_fitting`. Descriptions and demonstrations of the functions in each of these packages are found in the iPython notebook `Using the TTV Module`. The folder `emcee fitting example` demonstrates a different method that I primarily used for TTV fitting, implemented using [the `emcee` package](https://github.com/dfm/emcee), though note the code within is working code and is not well documented.
