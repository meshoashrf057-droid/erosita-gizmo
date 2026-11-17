"""Read-only GIZMO C/C++ adapter for yt 4.4.x.

Native kernel support radii are mapped before particle indexing, including the
coordinate/selection IO path. No installed yt files or original HDF5 are changed.
"""
from __future__ import annotations
import hashlib
import json
import re
import tempfile
import warnings
from pathlib import Path
import h5py
import numpy as np
from yt.frontends.gizmo.data_structures import GizmoDataset
from yt.frontends.gizmo.fields import GizmoFieldInfo
from yt.frontends.gadget.io import IOHandlerGadgetHDF5
from gizmo_units import resolve_units, require_noncosmological
from gizmo_schema import FIELD_ALIASES

def snapshot_parts(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        found = discover_snapshots(path)
        if len(found) != 1:
            raise ValueError(f"{path}: select one snapshot explicitly; found {len(found)}")
        path = found[0]
    if not path.exists():
        choices = [Path(str(path)+ext) for ext in (".hdf5", ".0.hdf5")]
        path = next((p for p in choices if p.is_file()), path)
    if any(x in path.name for x in (".hsml.", ".ewah", ".kdtree", ".tmp.")):
        raise ValueError(f"{path}: cache files are not snapshots")
    with h5py.File(path, "r") as f:
        if "Header" not in f:
            raise ValueError(f"{path}: missing Header")
        count = int(f["Header"].attrs.get("NumFilesPerSnapshot", 1))
    if count == 1:
        return [path]
    match = re.fullmatch(r"(.+)\.\d+\.hdf5", path.name)
    if not match or count < 1:
        raise ValueError(f"{path}: invalid split-snapshot naming/count")
    parts = [path.with_name(f"{match[1]}.{i}.hdf5") for i in range(count)]
    missing = [str(p) for p in parts if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing snapshot parts: {missing}")
    return parts

def discover_snapshots(directory):
    result = {}
    for p in Path(directory).glob("snapshot_*.hdf5"):
        m = re.fullmatch(r"snapshot_(\d+)(?:\.(\d+))?\.hdf5", p.name)
        if not m or m[2] not in (None, "0"):
            continue
        n = int(m[1])
        if n in result:
            raise ValueError(f"Ambiguous single/split snapshot {n}: {directory}")
        result[n] = p.resolve()
    return [result[n] for n in sorted(result)]

def _native_name(group, field):
    if field in group:
        return field
    for old, new in FIELD_ALIASES.items():
        if field == old and new in group:
            return new
        if field == new and old in group:
            return old
    return field

class ProjectGizmoFields(GizmoFieldInfo):
    known_particle_fields = GizmoFieldInfo.known_particle_fields + (
        ("KernelMaxRadius", ("code_length", [], None)),
        ("StellarKernelMaxRadius", ("code_length", [], None)),
        ("StellarSmoothingLength", ("code_length", [], None)),
        ("smoothing_length", ("code_length", [], None)),
        ("Temperature", ("K", ["temperature"], None)),
        ("Sink_Mass", ("code_mass", [], None)),
        ("Sink_Mdot", ("code_mass/code_time", [], None)),
        ("Sink_Mass_Reservoir", ("code_mass", [], None)),
    )


    def setup_gas_particle_fields(self, ptype):
        super().setup_gas_particle_fields(ptype)
        if (ptype, "Temperature") in self.field_list:
            return
        # ElectronAbundance is electrons per hydrogen nucleus. FIRE11 columns
        # 0 and 1 are total Z and He; metals' particle number is neglected here.
        def temperature(field, data):
            if (ptype, "ElectronAbundance") in self.field_list:
                hydrogen = 0.76
                if (ptype, "Metallicity_00") in self.field_list and (ptype, "Metallicity_01") in self.field_list:
                    hydrogen = 1-data[ptype,"Metallicity_00"]-data[ptype,"Metallicity_01"]
                mu = 4/(1+3*hydrogen+4*hydrogen*data[ptype,"ElectronAbundance"])
            else:
                mu = self.ds._compat_temperature_mu
            energy=data[ptype,"InternalEnergy"].to("cm**2/s**2")
            return ((self.ds._compat_gamma-1)*energy*mu
                    *data.ds.quan(1.67262192369e-24,"g")
                    /data.ds.quan(1.380649e-16,"erg/K")).to("K")
        self.add_field((ptype,"temperature"),function=temperature,
                       sampling_type="particle",units="K",force_override=True)
        self.add_field(("gas","temperature"),function=temperature,
                       sampling_type="particle",units="K",force_override=True)


class ProjectGizmoIO(IOHandlerGadgetHDF5):
    _dataset_type = "gizmo_project"

    def _identify_fields(self, data_file):
        fields, units = super()._identify_fields(data_file)
        found = set(fields)
        for ptype, name in list(fields):
            for old, new in FIELD_ALIASES.items():
                if name in (old, new):
                    found.update(((ptype, old), (ptype, new)))
        return sorted(found), units

    def _read_particle_data_file(self, data_file, ptf, selector=None):
        translated, reverse = {}, {}
        with h5py.File(data_file.filename, "r") as f:
            for ptype, names in ptf.items():
                if ptype not in f:
                    translated[ptype] = names
                    continue
                translated[ptype] = []
                for name in names:
                    actual = _native_name(f[ptype], name)
                    if actual not in translated[ptype]:
                        translated[ptype].append(actual)
                    reverse.setdefault((ptype, actual), []).append((ptype, name))
        actual = super()._read_particle_data_file(data_file, translated, selector)
        return {requested: value for key, value in actual.items()
                for requested in reverse.get(key, [key])}

    def _get_smoothing_length(self, data_file, position_dtype, position_shape):
        key = str(Path(data_file.filename).resolve())
        si, ei = data_file.start, data_file.end
        if self.ds.gen_hsmls:
            self._ensure_reconstruction()
            values = self.ds._compat_hsml[key][si:ei]
        else:
            with h5py.File(key, "r") as f:
                g = f[self.ds._sph_ptypes[0]]
                values = g[_native_name(g, "SmoothingLength")][si:ei]
        dtype = np.result_type(values.dtype, position_dtype or np.float64)
        return np.asarray(values, dtype=dtype)

    def _generate_smoothing_length(self, index):
        if self.ds.gen_hsmls:
            self._ensure_reconstruction()

    def _ensure_reconstruction(self):
        if hasattr(self.ds, "_compat_hsml"):
            return
        from scipy.spatial import cKDTree
        positions, counts = [], []
        for path in self.ds._compat_parts:
            with h5py.File(path, "r") as f:
                x = np.asarray(f["PartType0/Coordinates"], dtype=np.float64) if "PartType0" in f else np.empty((0, 3))
                positions.append(x)
                counts.append(len(x))
        xyz = np.concatenate(positions)
        if len(xyz) < 2:
            raise ValueError("At least two gas particles are needed to reconstruct kernel radii")
        k = min(33, len(xyz))
        distances = cKDTree(xyz).query(xyz, k=[k], workers=1)[0][:, 0]
        if not np.isfinite(distances).all() or (distances <= 0).any():
            raise ValueError("Degenerate coordinates prevent positive kernel reconstruction")
        self.ds._compat_hsml = {}
        offset = 0
        for path, count in zip(self.ds._compat_parts, counts):
            self.ds._compat_hsml[str(path)] = distances[offset:offset+count]
            offset += count
        np.save(Path(self.ds.compatibility["cache_dir"])/"reconstructed_kernel_radius.npy", distances)
        self.ds.compatibility["kernel_reconstruction"] = dict(
            method="full-sample nonperiodic exact k-nearest distance",
            neighbors_excluding_self=k-1, unit="code_length")
        warnings.warn("Native kernel radius absent: reconstructed on the complete gas sample; "
                      "this is an analysis approximation, not a GIZMO density solve.", RuntimeWarning)

class ProjectGizmoDataset(GizmoDataset):
    _field_info_class = ProjectGizmoFields

    @classmethod
    def _is_valid(cls, *args, **kwargs):
        # Only the explicit project loader selects this adapter.
        return False

    def _get_hvals(self):
        values = super()._get_hvals()
        with h5py.File(self.parameter_filename, "r") as f:
            if "PartType0" in f and "KernelMaxRadius" in f["PartType0"]:
                self.gen_hsmls = False
        return values

def load_gizmo(path, unit_base=None, cache_dir=None, frontend="auto",
               allow_reconstruction=True, temperature_mu=None, gamma=5/3, **kwargs):
    if frontend not in ("auto", "gizmo"):
        raise ValueError("frontend must be auto or gizmo")
    parts = snapshot_parts(path)
    overrides = {}
    base = dict(unit_base or {})
    for key in ("UnitLength_in_cm", "UnitMass_in_g", "UnitVelocity_in_cm_per_s"):
        if key in base:
            overrides[key] = base[key]
    from unyt import unyt_quantity
    for name, key, unit in (("length", "UnitLength_in_cm", "cm"),
                            ("mass", "UnitMass_in_g", "g"),
                            ("velocity", "UnitVelocity_in_cm_per_s", "cm/s")):
        if name in base:
            val = base[name]
            if isinstance(val[0], str):
                val = (val[1], val[0])
            overrides[key] = float(unyt_quantity(*val).to_value(unit))
    cgs, sources = resolve_units(parts[0], overrides=overrides)
    require_noncosmological(parts[0])
    observed_reference, _ = resolve_units(parts[0], fallback=overrides)
    kinds, native, headers, thermal_schemas = set(), [], [], set()
    lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
    counts_sum = np.zeros(6, dtype=np.uint64)
    for part in parts:
        actual_units, _ = resolve_units(part, fallback=overrides)
        if any(not np.isclose(actual_units[k], observed_reference[k], rtol=1e-10, atol=0) for k in cgs):
            raise ValueError(f"{part}: units differ across snapshot parts")
        require_noncosmological(part)
        with h5py.File(part, "r") as f:
            h = dict(f["Header"].attrs)
            headers.append(h)
            if frontend == "auto" and "GIZMO_version" not in h:
                z = f.get("PartType0/Metallicity")
                if z is None or z.ndim != 2 or z.shape[1] != 11:
                    raise ValueError(f"{part}: ambiguous format; use frontend='gizmo' explicitly")
            kid = h.get("Kernel_Function_ID")
            if kid is not None and int(kid) != 3 and "kernel_name" not in kwargs:
                raise ValueError(f"{part}: kernel ID {kid} needs an explicitly verified yt kernel")
            if kid is None:
                warnings.warn(f"{part}: kernel ID absent; using project cubic-spline convention", RuntimeWarning)
            counts = np.asarray(h["NumPart_ThisFile"], dtype=np.uint64)
            counts_sum += counts
            for i in range(6):
                name = f"PartType{i}"
                if not counts[i]:
                    continue
                if name not in f or "Coordinates" not in f[name]:
                    raise ValueError(f"{part}: missing {name}/Coordinates")
                g = f[name]
                if g["Coordinates"].shape != (int(counts[i]), 3):
                    raise ValueError(f"{part}: coordinate count/shape mismatch")
                for start in range(0, int(counts[i]), 262144):
                    xyz = np.asarray(g["Coordinates"][start:start+262144])
                    if not np.isfinite(xyz).all():
                        raise ValueError(f"{part}: nonfinite coordinates")
                    lower = np.minimum(lower, xyz.min(axis=0))
                    upper = np.maximum(upper, xyz.max(axis=0))
                for old, new in FIELD_ALIASES.items():
                    if old in g and new in g:
                        a, b = np.asarray(g[old]), np.asarray(g[new])
                        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
                            raise ValueError(f"{part}/{name}: conflicting aliases {old}/{new}")
                if i == 0:
                    thermal_schemas.add(tuple(name in g for name in ("Temperature","InternalEnergy","ElectronAbundance")))
                    field = _native_name(g, "SmoothingLength")
                    if field in g:
                        radius = np.asarray(g[field])
                        if radius.shape != (int(counts[i]),) or not np.isfinite(radius).all() or (radius <= 0).any():
                            raise ValueError(f"{part}: invalid native kernel radius")
                        native.append(True)
                        kinds.add(field)
                    else:
                        native.append(False)
    if len(thermal_schemas)>1:
        raise ValueError("Thermodynamic fields differ across gas snapshot parts")
    if not np.isfinite(gamma) or gamma<=1:
        raise ValueError("Temperature gamma must be finite and >1")
    native_t, internal_u, electrons = next(iter(thermal_schemas), (False,False,False))
    if thermal_schemas and not native_t:
        if not internal_u:
            raise ValueError("Gas requires native Temperature or InternalEnergy")
        if not electrons and (temperature_mu is None or not np.isfinite(temperature_mu) or temperature_mu<=0):
            raise ValueError("Temperature fallback without ElectronAbundance requires explicit temperature_mu")
        warnings.warn("Temperature absent: ideal atomic-gas fallback from U*v_unit**2; "
                      "gamma/chemical assumptions are recorded in compatibility metadata.", RuntimeWarning)
    if native and any(native) != all(native):
        raise ValueError("Some gas snapshot parts lack native kernel radius")
    if native and not any(native) and not allow_reconstruction:
        raise ValueError("Native kernel radius missing; reconstruction disabled")
    h = headers[0]
    total = np.asarray(h.get("NumPart_Total", counts_sum), dtype=np.uint64)
    total = total + (np.asarray(h.get("NumPart_Total_HighWord", np.zeros(6)), dtype=np.uint64) << np.uint64(32))
    if not np.array_equal(counts_sum, total):
        raise ValueError("Snapshot total counts do not match all parts")
    for other in headers[1:]:
        if int(other["NumFilesPerSnapshot"]) != len(parts) or float(other["Time"]) != float(h["Time"]):
            raise ValueError("Inconsistent snapshot part headers")
    temp = None
    if cache_dir is None:
        temp = tempfile.TemporaryDirectory(prefix="gizmo_compat_")
        cache_dir = temp.name
    cache = Path(cache_dir).expanduser().resolve()
    if cache in {p.parent for p in parts}:
        raise ValueError("Cache directory must be separate from input directory")
    fingerprint = hashlib.sha256("|".join(f"{p}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in parts).encode()).hexdigest()[:20]
    cache = cache/fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    if "bounding_box" not in kwargs and np.isfinite(lower).all():
        box = float(h.get("BoxSize", 0))
        lower = np.minimum(lower, 0)
        upper = np.maximum(upper, box)
        margin = np.maximum(upper-lower, 1)*1e-6
        kwargs["bounding_box"] = np.column_stack((lower-margin, upper+margin))
    declared = dict(cgs, specific_energy=(cgs["UnitVelocity_in_cm_per_s"]**2, "cm**2/s**2"))
    ds = ProjectGizmoDataset(str(parts[0]), dataset_type="gizmo_project", unit_base=declared,
                            index_filename=str(cache/"index.ewah"),
                            kernel_name=kwargs.pop("kernel_name", "cubic"), **kwargs)
    if native:
        ds.gen_hsmls = not all(native)
    ds.kdtree_filename = str(cache/"tree.kdtree")
    ds._periodicity = (False, False, False)
    ds._compat_temperature_mu = temperature_mu
    ds._compat_gamma = gamma
    ds._compat_parts = parts
    ds._compat_temporary_cache = temp
    ds.compatibility = dict(frontend=type(ds).__name__, source_files=[str(p) for p in parts],
                            unit_cgs=cgs, unit_sources=sources, kernel_source=sorted(kinds) or ["reconstruction"],
                            kernel_definition="compact support radius h, u=r/h; cubic ID3; no numeric rescaling",
                            cache_dir=str(cache), original_inputs_read_only=True,
                            temperature_source=("native Temperature in K" if native_t else
                                "ideal atomic gas from U*v_unit**2; X=1-Z-He when FIRE11 available, otherwise X=0.76; "
                                "mu from electrons/H or explicit value; metal particle counts neglected"),
                            temperature_gamma=gamma, temperature_mu=temperature_mu)
    (cache/"provenance.json").write_text(json.dumps(ds.compatibility, indent=2))
    print("[gizmo-compat]", type(ds).__name__, "native_kernel="+",".join(sorted(kinds)),
          "gen_hsmls="+str(ds.gen_hsmls), "cache="+str(cache), flush=True)
    ds._erosita_reader_info = {
        "frontend": type(ds).__name__, "unit_source": sources,
        "smoothing_length_source": ",".join(sorted(kinds)) if kinds else "full_sample_reconstruction",
        "smoothing_length_units": "code_length",
        "code_length_kpc": float(ds.quan(1, "code_length").to_value("kpc")),
        "code_mass_Msun": float(ds.quan(1, "code_mass").to_value("Msun")),
        "code_velocity_kms": float(ds.quan(1, "code_velocity").to_value("km/s")),
        "code_time_Gyr": float(ds.quan(1, "code_time").to_value("Gyr")),
    }
    return ds
