# NIST database data

Example:

```
import nist_lookup.xraydb_plugin as xdb
xdb.xray_delta_beta("SiO2", 2, 40e3)
# material or compound, density (g/cm3), energy (eV)
(2.5912407417839865e-07, 2.3106297250405836e-10, 1.067495673748253)
# > delta, beta, attenuation length (cm)
```
