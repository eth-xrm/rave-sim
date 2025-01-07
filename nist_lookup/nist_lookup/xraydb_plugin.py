import sys
from math import pi

from .physical_constants import R_ELECTRON_CM, AVOGADRO, PLANCK_HC
from .chemparser import chemparse
from .xraydb import xrayDB

'''
Functions for accessing and using data from X-ray Databases and
Tables.  Many of these take an element as an argument -- this
can be either the atomic symbol or atomic number Z.

The data and functions here include (but are not limited too):

member name     descrption
------------    ------------------------------
materials       dictionary of composition of common materials
chemparse       parse a Chemical formula to compositiondictionary.
atomic_mass     return atomic mass for an element
f0              Thomson X-ray scattering factor
f1f2_cl         Anomalous scattering factors from Cromer-Libermann
mu_elam         X-ray attenuation coefficients from Elam etal
mu_chantler     X-ray attenuation coefficients from Chantler
xray_edges      X-ray absorption edges for an element
xray_lines      X-ray emission lines for an element
'''


def xray_line(element, line='Ka', xdb=xrayDB()):
    """returns data for an  x-ray emission lines of an element, given
    the siegbahn notation for the like (Ka1, Lb1, etc).  Returns:
         energy (in eV), intensity, initial_level, final_level

    arguments
    ---------
    element:   atomic number, atomic symbol for element
    line:      siegbahn notation for emission line

    if line is 'Ka', 'Kb', 'La', 'Lb', 'Lg', without number,
    the weighted average for this family of lines is returned.

    Data from Elam, Ravel, and Sieber.
    """
    lines = xdb.xray_lines(element)

    family = line.lower()
    if family == 'k':
        family = 'ka'
    if family == 'l':
        family = 'la'
    if family in ('ka', 'kb', 'la', 'lb', 'lg'):
        scale = 1.e-99
        value = 0.0
        linit, lfinal = None, None
        for key, val in lines.items():
            if key.lower().startswith(family):
                value += val[0]*val[1]
                scale += val[1]
                if linit is None:
                    linit = val[2]
                if lfinal is None:
                    lfinal = val[3][0]
        return (value/scale, scale, linit, lfinal)
    else:
        return lines.get(line.title(), None)


def fluo_yield(symbol, edge, emission, energy,
               energy_margin=-150, xdb=xrayDB()):
    """Given
         atomic_symbol, edge, emission family, and incident energy,

    where 'emission' is the family of emission lines ('Ka', 'Kb', 'Lb', etc)
    returns

    fluorescence_yield, weighted-average fluorescence energy, net_probability

    fyield = 0  if energy < edge_energy + energy_margin (default=-150)

    > fluo_yield('Fe', 'K', 'Ka', 8000)
    0.350985, 6400.752419799043, 0.874576096

    > fluo_yield('Fe', 'K', 'Ka', 6800)
    0.0, 6400.752419799043, 0.874576096

    > fluo_yield('Ag', 'L3', 'La', 6000)
    0.052, 2982.129655446868, 0.861899000000000

    compare to xray_lines() which gives the full set of emission lines
    ('Ka1', 'Kb3', etc) and probabilities for each of these.

    Adapted for Larch from code by Yong Choi
    """
    e0, fyield, jump = xdb.xray_edge(symbol, edge)
    trans = xdb.xray_lines(symbol, initial_level=edge)

    lines = []
    net_ener, net_prob = 0., 0.
    for name, vals in trans.items():
        en, prob = vals[0], vals[1]
        if name.startswith(emission):
            lines.append([name, en, prob])

    for name, en, prob in lines:
        if name.startswith(emission):
            net_ener += en*prob
            net_prob += prob
    if net_prob <= 0:
        net_prob = 1
    net_ener = net_ener / net_prob
    if energy < e0 + energy_margin:
        fyield = 0
    return fyield, net_ener, net_prob


class Scatterer:
    """Scattering Element

    lamb=PLANCK_HC /(eV0/1000.)*1e-11    # in cm, 1e-8cm = 1 Angstrom
    Xsection=2* R_ELECTRON_CM *lamb*f2/BARN    # in Barns/atom
    """
    def __init__(self, symbol, energy=10000, xdb=xrayDB()):
        # atomic symbol and incident x-ray energy (eV)
        self.symbol = symbol
        self.number = xdb.atomic_number(symbol)
        self.mass = xdb.atomic_mass(symbol)
        self.f1 = xdb.chantler_data(symbol, energy, 'f1')
        self.f1 = self.f1 + self.number
        self.f2 = xdb.chantler_data(symbol, energy, 'f2')
        self.mu_photo = xdb.chantler_data(symbol, energy, 'mu_photo')
        self.mu_total = xdb.chantler_data(symbol, energy, 'mu_total')


def xray_delta_beta(material, density, energy,
                    photo_only=False, xdb=xrayDB()):
    """
    return anomalous components of the index of refraction for a material,
    using the tabulated scattering components from Chantler.

    arguments:
    ----------
       material:   chemical formula  ('Fe2O3', 'CaMg(CO3)2', 'La1.9Sr0.1CuO4')
       density:    material density in g/cm^3
       energy:     x-ray energy in eV
       photo_only: boolean for returning photo cross-section component only
                   if False (default), the total cross-section is returned
    returns:
    ---------
      (delta, beta, atlen)

    where
      delta :  real part of index of refraction
      beta  :  imag part of index of refraction
      atlen :  attenuation length in cm

    These are the anomalous scattering components of the index of refraction:

    n = 1 - delta - i*beta = 1 - lambda**2 * r0/(2*pi) Sum_j (n_j * fj)

    Adapted for Larch from code by Yong Choi
    """
    lamb_cm = 1.e-8 * PLANCK_HC / energy  # lambda in cm
    elements = []
    for symbol, number in chemparse(material).items():
        elements.append((number, Scatterer(symbol, energy, xdb)))

    total_mass, delta, beta_photo, beta_total = 0, 0, 0, 0
    for (number, scat) in elements:
        weight = density*number*AVOGADRO
        delta += weight * scat.f1
        beta_photo += weight * scat.f2
        beta_total += weight * scat.f2*(scat.mu_total/scat.mu_photo)
        total_mass += number * scat.mass

    scale = lamb_cm * lamb_cm * R_ELECTRON_CM / (2*pi * total_mass)
    delta = delta * scale
    beta = beta_total * scale
    if photo_only:
        beta = beta_photo * scale
    return delta, beta, lamb_cm/(4*pi*beta)

if __name__ == '__main__':
    print(xray_delta_beta('Fe2O3', 11, 1e4))
    import cProfile
    cProfile.run(
        'for _ in range(20): xray_delta_beta("Fe2O3", 11, 1e4)',
        'profile')
