"""Field names shared by project readers and IC/merge writers."""
import numpy as np

FIELD_ALIASES = {
    "SmoothingLength": "KernelMaxRadius",
    "StellarSmoothingLength": "StellarKernelMaxRadius",
    "BH_Mass": "Sink_Mass",
    "BH_Mdot": "Sink_Mdot",
    "BH_Mass_AlphaDisk": "Sink_Mass_Reservoir",
}

def normalize_fields(fields, target="c"):
    if target not in ("c", "cpp"):
        raise ValueError("target must be 'c' or 'cpp'")
    result = dict(fields)
    for old, new in FIELD_ALIASES.items():
        if old in result and new in result:
            a, b = np.asarray(result[old]), np.asarray(result[new])
            if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
                raise ValueError(f"Conflicting aliases: {old} and {new}")
        wanted, other = (old, new) if target == "c" else (new, old)
        if other in result:
            if wanted not in result:
                result[wanted] = result[other]
            del result[other]
    return result
