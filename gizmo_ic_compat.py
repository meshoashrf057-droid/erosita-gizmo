"""Validation and metadata for newly generated files; never call on source snapshots."""
from pathlib import Path
import numpy as np
import h5py
from gizmo_units import resolve_units, require_noncosmological
from gizmo_schema import FIELD_ALIASES

def checked_ids(values, bits=32):
    a=np.asarray(values)
    if a.ndim!=1 or a.dtype.kind not in "iu" or np.any(a<=0):
        raise ValueError("ParticleIDs must be a positive one-dimensional integer array")
    if a.size and int(a.max())>np.iinfo(np.dtype(f"uint{bits}")).max:
        raise ValueError(f"ParticleIDs exceed uint{bits}; use a matching LONGIDS build and explicit 64-bit generator")
    return a.astype(f"uint{bits}",copy=False)

def finalize_new_ic(handle, target, units):
    """Declare existing numeric code units and map names in a NEW output only."""
    if target not in ("c","cpp"): raise ValueError("target must be c or cpp")
    if handle.mode=="r": raise ValueError("finalize_new_ic requires a newly created writable output")
    header=handle["Header"].attrs
    header["TargetFormat"]=target
    header["ComovingIntegrationOn"]=0
    header["ICInitializationWarning"]="flag0 metallicity/age follow global initialization parameters; sink state depends on compiled initialization path"
    for key,value in units.items(): header[key]=float(value)
    for p in range(6):
        g=handle.get(f"PartType{p}")
        if g is None: continue
        for old,new in FIELD_ALIASES.items():
            if old in g and new in g:
                if g[old].shape!=g[new].shape or not np.array_equal(g[old][:],g[new][:],equal_nan=True):
                    raise ValueError(f"Conflicting IC aliases: {old}/{new}")
                del g[new if target=="c" else old]
            source,dest=(new,old) if target=="c" else (old,new)
            if source in g: g.move(source,dest)

def validate_generated_ic(path, require_unique_ids=True):
    result={"path":str(Path(path).resolve()),"counts":{},"mass_code":{}}
    ids=[];identities=[]
    with h5py.File(path,"r") as f:
        result["units"],result["unit_sources"]=resolve_units(f);require_noncosmological(f)
        h=f["Header"].attrs
        counts=np.asarray(h["NumPart_ThisFile"],dtype=np.uint64)
        total=np.asarray(h["NumPart_Total"],dtype=np.uint64)+(np.asarray(h.get("NumPart_Total_HighWord",np.zeros(6)),dtype=np.uint64)<<np.uint64(32))
        mt=np.asarray(h["MassTable"])
        if any(a.shape!=(6,) for a in (counts,total,mt)) or int(h.get("NumFilesPerSnapshot",1))!=1:
            raise ValueError("Invalid single-file IC header")
        for p in range(6):
            key=f"PartType{p}";g=f.get(key);n=0 if g is None else len(g["ParticleIDs"])
            if n!=counts[p] or n!=total[p]:raise ValueError(f"{key}: count mismatch")
            result["counts"][key]=n
            if not n:continue
            for name,shape in [("Coordinates",(n,3)),("Velocities",(n,3)),("ParticleIDs",(n,))]:
                if name not in g or g[name].shape!=shape:raise ValueError(f"{key}/{name}: shape mismatch")
            for name,d in g.items():
                if not isinstance(d,h5py.Dataset) or d.ndim==0 or len(d)!=n:raise ValueError(f"{key}/{name}: alignment mismatch")
                if d.dtype.kind in "fci" and not np.isfinite(d[:]).all():raise ValueError(f"{key}/{name}: nonfinite")
            pid=checked_ids(g["ParticleIDs"][:],64);ids.append(pid)
            child=g["ParticleChildIDsNumber"][:] if "ParticleChildIDsNumber" in g else np.zeros(n,dtype=np.uint64)
            gen=g["ParticleIDGenerationNumber"][:] if "ParticleIDGenerationNumber" in g else np.zeros(n,dtype=np.uint64)
            identities.extend(zip(pid.tolist(),child.tolist(),gen.tolist()))
            mass=g["Masses"][:] if "Masses" in g else np.full(n,mt[p])
            if mass.shape!=(n,) or (mass<=0).any():raise ValueError(f"{key}: invalid mass")
            if mt[p]!=0 and not np.allclose(mass,mt[p],rtol=1e-12,atol=0):raise ValueError(f"{key}: MassTable conflict")
            result["mass_code"][key]=float(np.sum(mass,dtype=np.float64))
            if p==0 and ("InternalEnergy" not in g or g["InternalEnergy"].shape!=(n,) or np.any(g["InternalEnergy"][:]<=0)):
                raise ValueError("Gas needs positive specific internal energy")
            if "Metallicity" in g:
                z=g["Metallicity"][:]
                if z.ndim not in (1,2) or np.any(z<0) or np.any(z>1):raise ValueError(f"{key}: invalid metal fractions")
                if z.ndim==2 and z.shape[1]==11 and np.any(z[:,0]+z[:,1]>1+1e-6):raise ValueError("Z+He exceeds unity")
        flat=np.concatenate(ids) if ids else np.empty(0,dtype=np.uint64)
        result["raw_ids_unique"]=len(np.unique(flat))==len(flat)
        result["composite_ids_unique"]=len(set(identities))==len(identities)
        if require_unique_ids and not result["raw_ids_unique"]:raise ValueError("Global ParticleIDs collide")
        if not result["composite_ids_unique"]:raise ValueError("Global (ParticleID,child,generation) identities collide")
    return result
