# HMcode

![image](https://user-images.githubusercontent.com/9140961/228345397-f33d2f94-e8e4-4eb0-9fc9-9b27df407fbc.png)

A pure-`Python` implementation of the `HMcode-2020` method ([Mead et al. 2021](https://arxiv.org/abs/2009.01858)) for computing accurate non-linear matter power spectra across a wide range of cosmological parameters for $w(a)$-CDM models including curvature and massive neutrinos.

## Installation

Either
```bash
> pip install hmcode
```
or, if you want to edit the code, use the demo notebook, or run the tests or consistency checks, then clone the repository, `cd` into the directory, and then
```bash
> poetry install
```
to create a virtual environment with everything you need to get going.

## Dependencies

- `numpy`
- `scipy`
- `camb`
- `pyhalomodel`

## Use

```
import numpy as np
import camb
import hmcode

# Ranges
k = np.logspace(-3, 1, 100) # Wavenumbers [h/Mpc]
zs = [3., 2., 1., 0.5, 0.]  # Redshifts
T_AGN = 10**7.8             # Feedback temperature [K]

# Run CAMB
parameters = camb.CAMBparams(WantCls=False)
parameters.set_cosmology(H0=70.)
parameters.set_matter_power(redshifts=zs, kmax=100.) # kmax should be much larger than the wavenumber of interest
results = camb.get_results(parameters)

# HMcode
Pk = hmcode.power(k, zs, results, T_AGN)
```

## Note

To whom it may concern,

I coded this `Python` version of `HMcode-2020` ([Mead et al. 2021](https://arxiv.org/abs/2009.01858)) up quite quickly before leaving academia in February 2023. It is written in pure `Python` and doesn't use any of the original Fortran code whatsoever. There is something amazing/dispiriting about coding something up in 3 days that previously took 5 years. A tragic last hoorah! At least I switched to `Python` eventually...

You might also be interested in [`pyhalomodel`](https://pypi.org/project/pyhalomodel/), upon which this code depends, which implements a vanilla halo-model calculation for any desired large-scale-structure tracer. Alternatively, and very confusingly, you might be interested in this [`pyhmcode`](https://pypi.org/project/pyhmcode/), which provides a wrapper around the original `Fortran` `HMcode` implementation.

I compared it against the `CAMB-HMcode` version for 100 random sets of cosmological parameters ($k < 10 h\mathrm{Mpc}^{-1}$; $z < 3$). The level of agreement between the two codes is as follows:
- LCDM: Mean error: 0.10%; Standard deviation of error: 0.03%; Worst-case error; 0.28%
- k-LCDM: Mean error: 0.12%; Standard deviation of error: 0.04%; Worst-case error; 0.32%
- w-CDM: Mean error: 0.11%; Standard deviation of error: 0.03%; Worst-case error; 0.31%
- w(a)-CDM: Mean error: 0.14%; Standard deviation of error: 0.08%; Worst-case error; 0.77%
- nu-LCDM: Mean error: 0.45%; Standard deviation of error: 0.43%; Worst-case error; 1.97% (larger errors strongly correlated with neutrino mass)
- nu-k-w(a)-CDM: Mean error: 0.41%; Standard deviation of error: 0.42%; Worst-case error; 1.99% (larger errors strongly correlated with neutrino mass)

These comparisons can be reproduced using the `comparisons/CAMB.py` script.

The power-spectrum suppression caused by baryonic feedback has also been implemented, and the level of agreement between the *suppression* predicted by this code and the same implementation in `CAMB` is as follows:
- LCDM: Mean error: 0.02%; Standard deviation of error: <0.01%; Worst-case error; 0.03%
- nu-k-w(a)-CDM: Mean error: 0.06%; Standard deviation of error: 0.07%; Worst-case error; 0.49% (larger errors strongly correlated with neutrino mass)

The comparisons can be reproduced using the `comparisons/CAMB_feedback.py` script.

While the quoted accuracy of `HMcode-2020` relative to simulations is RMS ~2.5%, note that the accuracy is anti-correlated with neutrino masses (cf. Fig. 2 of [Mead et al. 2021](https://arxiv.org/abs/2009.01858)). The larger discrepancies between the codes for massive neutrinos (2% for ~1eV) may seem worrisome, but here are some reasons why I am not that worried:
- Here, neutrinos are treated as completely cold matter when calculating the linear growth factors, whereas in `CAMB-HMcode` the transition from behaving like radiation to behaving like matter is accounted for in the linear growth.
- Here the *cold* matter power spectrum is taken directly from `CAMB` whereas in `CAMB-HMcode` the *cold* spectrum is calculated approximately from the total matter power spectrum using approximations for the scale-dependent growth rate from [Eisenstein & Hu (1999)](https://arxiv.org/abs/astro-ph/9710252).
- If I resort to the old approximation for the *cold* matter power spectrum (and therefore the cold $\sigma(R)$) then the level of agreement between the codes for nu-k-w(a)-CDM improves to: Mean error: 0.15%; Std error: 0.11%; Worst error; 1.15%.

Using the actual cold matter spectrum is definitely an improvement from a theoretical perspective (and for speed), so I decided to keep that at the cost of the disagreement between this code at `CAMB-HMcode` for models with very high neutrino mass. If better agreement between this code and `CAMB-HMcode` is important to you then the old approximate cold spectrum can be used by changing the `sigma_cold_approx` flag at the top of `hmcode.py`. Note that while ignoring the actual energy-density scaling of massive neutrinos might seem to be a (small) problem, keep in mind the comments below regarding the linear growth factor.

I think any residual differences between codes must therefore stem from:
- The BAO de-wiggling process (different `k` grids)
- The $\sigma_\mathrm{v}$ numerical integration
- The $n_\mathrm{eff}$ calculation (numerical differentiation here; numerical integration in `CAMB-HMcode`)
- The $\sigma(R)$ numerical integration (using `CAMB` here; done internally in `CAMB-HMcode`)
- The linear growth ODE solution
- Root finding for the halo-collapse redshift and for $R_\mathrm{nl}$

But I didn't have time to investigate these differences more thoroughly. Note that there are accuracy parameters in `CAMB-HMcode` fixed at the $10^{-4}$ level, so you would never expect better than 0.01% agreement. Given that `HMcode` is only accurate at the ~2.5% level compared to simulated power spectra, the level of agreement between the codes seems okay to me, with the caveats above regarding very massive neutrinos.

While writing this code I had a few ideas for future improvements:
- ~~Add the `HMcode-2020` baryon-feedback model; this would not be too hard for the enthusiastic student/postdoc.~~ (Thanks Laila Linke!)
- The predictions are a bit sensitive to the smoothing $\sigma$ used for the dewiggling. This should probably be a fitted parameter.
- It's annoying having to calculate linear growth functions (all, LCDM), especially since the linear growth doesn't really exist. One should probably use the $P(k)$ amplitude evolution over time at some cleverly chosen scale instead, or instead the evolution of $\sigma(R)$ over time at some pertinent $R$ value. Note that the growth factors are *only* used to calculate the [Dolag et al. (2004)](https://arxiv.org/abs/astro-ph/0309771) correction and [Mead (2017)](https://arxiv.org/abs/1606.05345) $\delta_\mathrm{c}$, $\Delta_\mathrm{v}$.
- I never liked the halo bloating parameter, it's hard to understand the effect of modifying halo profiles in Fourier space. Someone should get rid of this (maybe modify the halo mass function instead?).
- Redshift 'infinity' for the Dolag correction is actually $z_\infty = 10$. `HMcode` predictions *are* sensitive to this (particularly w(a)CDM). Either the redshift 'infinity' should be fitted or the halo-concentration model for beyond-LCDM should be improved.
- The massive neutrino correction for the [Mead (2017)](https://arxiv.org/abs/1606.05345) $\delta_\mathrm{c}$, $\Delta_\mathrm{v}$ formulae (appendix A of [Mead et al. 2021](https://arxiv.org/abs/2009.01858)) is crude and should be improved. I guess using the intuition that hot neutrinos are ~smoothly distributed on halo scales. Currently neutrinos are treated as cold matter in the linear/accumulated growth calculation (used by [Mead 2017](https://arxiv.org/abs/1606.05345)), which seems a bit wrong.
- I haven't checked how fast this code is, but there are a couple of TODO in the code that might improve speed if necessary.
- The choices regarding how to account for massive neutrinos could usefully be revisited. This whole subject is a bit confusing and the code doesn't help to alleviate the confusion. Choices like what to use for: $\delta_\mathrm{c}$; $\Delta_\mathrm{v}$; $\sigma(R)$; $R_\mathrm{nl}$; $n_\mathrm{eff}$; $c(M)$.
- Refit model (including $\sigma$ for BAO smoothing and $z_\infty$ for [Dolag et al. 2004](https://arxiv.org/abs/astro-ph/0309771)) to new emulator(s) (e.g., [Mira Titan IV](https://arxiv.org/abs/2207.12345)).
- Don't be under any illusions that the `HMcode` parameters, or the forms of their dependence on the underlying power spectrum, are special in any particular way. A lot of experimentation went into finding these, but it was by no means exhaustive. However, please note that obviously these parameters should only depend on the underlying linear spectrum (rather than being random functions of $z$, $\Omega_\mathrm{m}$, $w$, or whatever).

Have fun,

Alexander Mead (2023/02/28)
