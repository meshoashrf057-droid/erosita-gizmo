"""Shared, explicit GIZMO unit metadata resolution; never changes input files."""
from __future__ import annotations
from pathlib import Path
import warnings
import h5py
import numpy as np

UNIT_ALIASES = {
    'UnitLength_in_cm': ('UnitLength_in_cm', 'UnitLength_In_CGS'),
    'UnitMass_in_g': ('UnitMass_in_g', 'UnitMass_In_CGS'),
    'UnitVelocity_in_cm_per_s': ('UnitVelocity_in_cm_per_s', 'UnitVelocity_In_CGS'),
}
KPC_CM = 3.0856775814913673e21
MSUN_G = 1.98841586e33
GYR_S = 31557600e9


def metadata(source):
    if isinstance(source, (str, Path)):
        with h5py.File(source, 'r') as handle:
            return metadata(handle)
    if isinstance(source, h5py.File):
        return {name: dict(source[name].attrs) for name in ('Header', 'Parameters', 'Units') if name in source}
    if any(name in source for name in ('Header', 'Parameters', 'Units')):
        return {name: dict(source[name]) for name in ('Header', 'Parameters', 'Units') if name in source}
    return {'metadata': dict(source)}


def resolve_units(source, overrides=None, fallback=None):
    """Return (canonical CGS values, provenance).

    Metadata conflicts are always errors, including with CLI overrides. Missing
    metadata requires explicit caller fallback or overrides, and is reported.
    CLI overrides change the unit declaration only; they never rescale arrays.
    """
    meta = metadata(source)
    overrides, fallback = overrides or {}, fallback or {}
    values, sources = {}, {}
    for key, aliases in UNIT_ALIASES.items():
        found = [(group+'/'+alias, float(np.asarray(attrs[alias]).reshape(())))
                 for group, attrs in meta.items() for alias in aliases if alias in attrs]
        for label, value in found:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid unit {label}={value}')
        if found and any(not np.isclose(value, found[0][1], rtol=1e-10, atol=0) for _, value in found[1:]):
            raise ValueError(f'Conflicting unit metadata for {key}: {found}')
        if key in overrides and overrides[key] is not None:
            value, origin = float(overrides[key]), 'explicit override'
            if found and not np.isclose(value, found[0][1], rtol=1e-10, atol=0):
                warnings.warn(f'{key}: explicit unit override {value} replaces {found}', RuntimeWarning, stacklevel=2)
        elif found:
            value, origin = found[0][1], '; '.join(label for label, _ in found)
        elif key in fallback:
            value, origin = float(fallback[key]), 'explicit caller fallback (metadata absent)'
            warnings.warn(f'{key} missing: using explicitly declared fallback {value}', RuntimeWarning, stacklevel=2)
        else:
            raise ValueError(f'Missing {key}; provide explicit unit declaration or corresponding simulation parameter')
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid unit {key}={value}')
        values[key], sources[key] = value, origin
    return values, sources


def physical_units(cgs):
    length, mass, velocity = (cgs[k] for k in UNIT_ALIASES)
    return dict(length_kpc=length/KPC_CM, mass_msun=mass/MSUN_G,
                velocity_kms=velocity/1e5, time_gyr=length/velocity/GYR_S,
                density_cgs=mass/length**3, specific_energy_cgs=velocity**2)


def require_noncosmological(source):
    meta = metadata(source)
    flags = [(group, bool(np.asarray(attrs['ComovingIntegrationOn']).reshape(())))
             for group, attrs in meta.items() if 'ComovingIntegrationOn' in attrs]
    if any(value for _, value in flags):
        raise ValueError(f'Only non-cosmological data are supported by this diagnostic: {flags}')
    return 'ComovingIntegrationOn=0' if flags else 'caller non-cosmological convention (flag absent)'
