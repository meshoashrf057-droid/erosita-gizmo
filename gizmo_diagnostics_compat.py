"""Thermodynamic reading for project diagnostics (input HDF5 is read-only)."""
from __future__ import annotations
import numpy as np


def gas_temperature(group, velocity_cms, gamma=5.0/3.0, fallback_mu=None):
    """Return (Kelvin array, provenance); prefer GIZMO OUTPUT_TEMPERATURE.

    ElectronAbundance is electrons per H nucleus. The fallback ideal atomic
    mixture ignores the small metal contribution to particle number. FIRE's
    11 columns contain total Z and He at indices 0 and 1, not 11 total metals.
    """
    n = len(group['Coordinates']) if 'Coordinates' in group else len(group['InternalEnergy'])
    if 'Temperature' in group:
        result = np.asarray(group['Temperature'], dtype=float)
        source = 'native GIZMO OUTPUT_TEMPERATURE (K); no numeric rescaling'
    else:
        if 'InternalEnergy' not in group:
            raise ValueError(f'{group.name}: Temperature and InternalEnergy are both missing')
        u = np.asarray(group['InternalEnergy'], dtype=float)
        if not np.isfinite(velocity_cms) or velocity_cms <= 0 or gamma <= 1:
            raise ValueError('Invalid velocity unit or adiabatic index')
        if 'ElectronAbundance' in group:
            ne = np.asarray(group['ElectronAbundance'], dtype=float)
            if ne.shape != (n,) or not np.isfinite(ne).all() or (ne < 0).any():
                raise ValueError(f'{group.name}: invalid ElectronAbundance')
            x = np.full(n, 0.76)
            composition = 'assumed hydrogen mass fraction X=0.76'
            if 'Metallicity' in group:
                z = np.asarray(group['Metallicity'], dtype=float)
                if z.ndim == 2 and z.shape == (n, 11):
                    x = 1.0 - z[:, 0] - z[:, 1]
                    composition = 'X=1-Z_total-He from FIRE 11-column Metallicity'
            if not np.isfinite(x).all() or (x <= 0).any() or (x > 1).any():
                raise ValueError(f'{group.name}: invalid hydrogen mass fraction')
            mu = 4.0 / (1.0 + 3.0*x + 4.0*x*ne)
            source = f'InternalEnergy * UnitVelocity^2, gamma={gamma:g}, ElectronAbundance; {composition}; atomic H/He ideal gas, metal particle-number correction neglected'
        else:
            if fallback_mu is None or not np.isfinite(fallback_mu) or fallback_mu <= 0:
                raise ValueError(f'{group.name}: temperature fallback requires ElectronAbundance or an explicit positive mu')
            mu = fallback_mu
            source = f'InternalEnergy * UnitVelocity^2, gamma={gamma:g}, explicit mu={mu:g}'
        result = (gamma-1.0) * u * velocity_cms**2 * mu * 1.67262192369e-24 / 1.380649e-16
    if result.shape != (n,) or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError(f'{group.name}: invalid temperature shape, nonfinite or negative values')
    return result, source


def is_snapshot_cache(path):
    """Exclude yt sidecars from broad snapshot globs."""
    name = str(path).rsplit('/', 1)[-1]
    return any(tag in name for tag in ('.hsml.', '.kdtree', '.ewah', '.tmp.'))


def require_single_snapshot(handle):
    count = int(handle['Header'].attrs.get('NumFilesPerSnapshot', 1))
    if count != 1:
        raise ValueError(f'{handle.filename}: this direct-array diagnostic requires one complete file; '
                         f'found {count} snapshot parts. Use the shared yt reader or a verified joined analysis copy.')
