## HMcode

To whom it may concern,

I coded this Python version of HMcode up quite quickly before leaving academia. 
It is written in pure Python and doesn't use any of the original Fortran code whatsoever.
There is something amazing/dispiriting about coding something up in 3 days that previously took 5 years.
A tragic last hoorah! At least I switched to Python eventually...

I tested it against the CAMB HMcode version for 100 random sets of cosmological parameters (k < 10 h/Mpc; z < 3). 
The accuracy is as follows:
    - LCDM: Mean error: 0.10%; Std error: 0.03%; Worst error; 0.21%
    - k-LCDM: Mean error: 0.11%; Std error: 0.03%; Worst error; 0.23%
    - w-CDM: Mean error: 0.10%; Std error: 0.03%; Worst error; 0.20%
    - w(a)-CDM: Mean error: 0.13%; Std error: 0.06%; Worst error; 0.48%
    - nu-LCDM: Mean error: 0.47%; Std error: 0.44%; Worst error; 2.01% (larger errors strongly correlated with neutrino mass)
    - nu-k-w(a)CDM: Mean error: 0.42%; Std error: 0.43%; Worst error; 2.02% (larger errors strongly correlated with neutrino mass)
These tests can be reproduced using the tests/test_HMcode.py script.

Note that the quoted accuracy of HMcode relative to simulations is RMS ~2.5%.
Note also that the accuracy is anti-correlated with neutrino masses (cf. Fig. 2 of https://arxiv.org/abs/2009.01858).
The larger discrepancies for massive neutrinos (2% for ~1eV) may seem worrisome, but here are some reasons that I am not that worried:
    - Here neutrinos are treated as completely cold matter when calculating the linear growth factors
    - In CAMB-HMcode the transition from behaving like radiation to behaving like matter is accounted for in the linear growth
    - Here the cold matter power spectrum is taken directly from CAMB.
    - In CAMB-HMcode the *cold* matter power spectrum is calculated approximately using Eisenstein & Hu (1999)
Using the actual cold matter spectrum is definitely a (small) improvement.
Ignoring the actual energy-density scaling of massive neutrinos might seem to be a (small) problem,
but keep in mind the comments below regarding the linear growth factor.

I think any residual differences must stem from:
    - The BAO de-wiggling process
    - The accuracy of the sigmaV numerical integration
    - The accuracy of the sigmaR numerical integration (using CAMB here; done internally in CAMB-HMcode)
    - The accuracy of the linear growth ODE solutions
    - Root finding for the halo-collapse redshift and for Rnl
But I didn't have time to investigate these differences more thoroughly.
Note that there are accuracy parameters in CAMB-HMcode fixed at the 1e-4 level, so you would never expect better than 0.01% agreement.
Although given that HMcode is only accurate at the ~2.5% level compared to simulations the level of agreement seems okay.

While writing this code I had a few ideas for future improvements:
    - Add the baryon-feedback model; this would not be too hard for the enthusiastic student/postdoc.
    - The predictions are a bit sensitive to the smoothing sigma used for the dewiggling. This should probably be a fitted parameter.
    - It's annoying having to calculate linear growth functions (all, LCDM), especially since the linear growth doesn't really exist. 
One should probably should use the P(k) amplitude evolution over time at some cleverly chosen scale instead,
or instead the evolution of sigma(R) over time at some pertinent R value.
Note that the growth factors are *only* used to calculate the Dolag correction and Mead (2017) delta_c, Delta_v.
    - I never liked the halo bloating parameter, it's hard to understand the effect of modifying halo profiles in Fourier space.
Someone should get rid of this (maybe modify the mass function instead?).
    - Redshift 'infinity' for the Dolag correction is actually z = 10. Predictions *are* sensitive to this (particularly w(a)CDM)
Either the redshift 'infinity' should be fitted or the halo-concentration model for beyond-LCDM should be improved somehow.
    - The massive neutrino correction for the Mead (2017) dc, Dv formula (appendix A of https://arxiv.org/abs/2009.01858) is crude.
This should be improved somehow, I guess using the intuition that hot neutrinos are ~smoothly distributed on halo scales.
Currently neutrinos are treated as cold matter in the linear/accumulated growth calculation (used by Mead 2017), which sounds a bit wrong.
    - I haven't checked how fast this code is, but there are a couple of TODO in the code that might improve speed if necessary.
    - The choices regarding how to account for massive neutrinos could usefully be revisited. This whole subject is a bit confusing
and the code doesn't help to alleviate the confusion. Choices like what to use for: dc; Dv; sigma; Rnl; neff; c(M).
    - Refit model (including sigma for BAO and zc for Dolag) to new emulator(s) (e.g., https://arxiv.org/abs/2207.12345).
    - Don't be under any illusions that the HMcode parameters, or the forms of their dependence on the underlying power spectrum,
are special in any particular way. A lot of experimentation went into finding these, but it was by no means exhaustive.
Obviously these parameters should only depend on the underlying spectrum though (rather than being random functions of z or whatever).

Have fun,
Alexander Mead (2023/02/21)