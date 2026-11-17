#!/usr/bin/env python3
"""Set the jet specific-angular-momentum vector on an explicit IC copy."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import h5py
import numpy as np

FIELDS = {"c": "BH_Specific_AngMom", "cpp": "Sink_Specific_AngMom"}

def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()

def set_bh_angular_momentum(filename, angmom_vector=None, target_format="auto"):
    """Edit the explicitly selected file; callers must pass their owned copy."""
    path=Path(filename).expanduser().resolve(strict=True)
    axis=np.asarray([0.,0.,1.] if angmom_vector is None else angmom_vector,dtype=np.float64)
    if axis.shape!=(3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis)==0:
        raise ValueError("axis must be a finite, nonzero vector with three components")
    if target_format not in ("auto","c","cpp"):
        raise ValueError("target format must be auto, c, or cpp")
    with h5py.File(path,"r+") as f:
        if "PartType5" not in f:
            raise ValueError("PartType5 is missing; no black-hole/sink particle to initialize")
        group=f["PartType5"]
        if "Coordinates" not in group or group["Coordinates"].ndim!=2 or group["Coordinates"].shape[1]!=3:
            raise ValueError("PartType5/Coordinates must have shape (N,3)")
        count=len(group["Coordinates"])
        if count<1:raise ValueError("PartType5 is empty")
        for name in ("Masses","ParticleIDs"):
            if name in group and group[name].shape!=(count,):
                raise ValueError("PartType5/"+name+" has an inconsistent shape")
        if "Header" in f and "NumPart_ThisFile" in f["Header"].attrs:
            if int(f["Header"].attrs["NumPart_ThisFile"][5])!=count:
                raise ValueError("Header PartType5 count differs from its datasets")
        if target_format=="auto":
            version=float(f["Header"].attrs.get("GIZMO_version",0)) if "Header" in f else 0
            target_format="cpp" if version>=2026 or FIELDS["cpp"] in group else "c"
        field=FIELDS[target_format]
        other=FIELDS["c" if target_format=="cpp" else "cpp"]
        for name in FIELDS.values():
            if name in group and (group[name].shape!=(count,3) or not np.isfinite(group[name][...]).all()):
                raise ValueError(name+" must be a finite (N,3) dataset")
        if field in group and other in group and not np.array_equal(group[field][...],group[other][...]):
            raise ValueError("Conflicting C/C++ specific-angular-momentum aliases")
        target=np.tile(axis,(count,1))
        old={name:group[name][...].tolist() for name in FIELDS.values() if name in group}
        # These changes apply only to the file explicitly selected for editing.
        if field not in group and other in group:
            group.move(other,field)
        elif other in group:
            del group[other]
        if field in group and group[field].dtype.kind=="f":
            group[field][...]=target
        else:
            if field in group:del group[field]
            group.create_dataset(field,data=target,dtype=np.float64)
        if not np.allclose(group[field][...],target,rtol=1e-7,atol=0):
            raise RuntimeError("Written angular momentum does not match requested vector")
        report=dict(path=str(path),target_format=target_format,field=field,count=count,
                    specific_angmom_vector=axis.tolist(),previous_values=old,
                    note="Only the axis field was initialized; this does not enable AGN physics or initialize other sink state.")
        f.flush()
    return report

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",default="./snapshot_011.hdf5")
    destination=parser.add_mutually_exclusive_group()
    destination.add_argument("--output",help="New HDF5 copy; existing destination is refused")
    destination.add_argument("--in-place",action="store_true",help="Explicitly edit a previously copied, owned jet input")
    parser.add_argument("--target-format",choices=("auto","c","cpp"),default="auto")
    parser.add_argument("--axis",nargs=3,type=float,default=(0.,0.,1.))
    parser.add_argument("--report",help="Write a new JSON audit file; existing path is refused")
    args=parser.parse_args(argv)
    source=Path(args.input).expanduser().resolve(strict=True)
    report_path=Path(args.report).expanduser().resolve() if args.report else None
    if report_path is not None and report_path.exists():raise FileExistsError(report_path)
    if args.in_place:
        before=_sha(source)
        report=set_bh_angular_momentum(source,args.axis,args.target_format)
        report.update(input_sha256_before=before,output_sha256=_sha(source),in_place=True)
    else:
        output=Path(args.output).expanduser().resolve() if args.output else source.with_name(source.stem+"_jetaxis"+source.suffix)
        if output==source:raise ValueError("Use --in-place only for a deliberately copied input")
        if output.exists():raise FileExistsError(output)
        output.parent.mkdir(parents=True,exist_ok=True)
        tmp=output.with_name(output.name+".partial")
        before=_sha(source)
        with source.open("rb") as inp,tmp.open("xb") as out:shutil.copyfileobj(inp,out,8*1024*1024)
        try:
            report=set_bh_angular_momentum(tmp,args.axis,args.target_format)
            if _sha(source)!=before:raise RuntimeError("Source changed while copying")
            os.link(tmp,output);tmp.unlink()
            report.update(path=str(output),source=str(source),input_sha256_before=before,output_sha256=_sha(output),in_place=False)
        except BaseException:
            if tmp.exists():tmp.unlink()
            raise
    if report_path is not None:
        report_path.parent.mkdir(parents=True,exist_ok=True)
        with report_path.open("x") as handle:json.dump(report,handle,ensure_ascii=False,indent=2);handle.write("\n")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report

if __name__=="__main__":
    main()
