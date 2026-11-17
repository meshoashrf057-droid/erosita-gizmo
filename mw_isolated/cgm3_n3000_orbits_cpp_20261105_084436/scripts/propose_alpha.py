"""Prepare a coarse orbit correction after a verified failure in this CGM3 campaign."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import subprocess
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MW = ROOT.parent
sys.path.insert(0, str(MW))
sys.path.insert(0, str(MW.parent))
import gate_endpoint as gate
from stage_safety import expected_parameters
import merge_relaxed_mw_lmc_smc as merger
from shooting_local import forward_state, propose_local_step
from satellite_relax_common import plain
from gizmo_units import resolve_units

FAMILY_KEY = "cpp-shooting-family:bb9b2bbcbcef2fddb4e66c4084fe218a9ecadb983331db587f7ed2fa0fef854f"
FAMILY_FILE = ROOT / "provenance/cgm3_family_inputs.json"
FAMILY_SHA = "90e187bfe2c3f27d270aee247eb27029f88450d006a2e7a7f2aea12f07feaeef"
SETTINGS = MW / "mw_lmc_smc_relaxed_trial13_000.orbit.json"
SETTINGS_SHA = "46debe8151f28c6de0f16272f6c39741f472d5a932cc551a0dc201edaeae5802"
DEPENDENCIES = {
    MW/"shooting_local.py":"4d76cbddba7f1691c544904e7b83722832d0b436fc27f7fcb6605d78edc5bf2e",
    MW/"tg19_backend.py":"6bad4ae89405b3095092fb65f472272c92a6f9900277a19ca03f71f64ab4d37a",
    MW/"merge_relaxed_mw_lmc_smc.py":"ff71bd1dc6ab4a37a5843a051fbb7c26231d96ee23eb6236eb20417c1f77ad6c",
    MW/"satellite_relax_common.py":"a7ff947aa0d5e2fb23025759d942b378345624fad026d75928c54097b4d3f0d7",
    MW/"shooting_cpp_20261031_082656/apply_orbit_step.py":"21c0f6a88030910f9c5326592a7d7d0a8bef866a22f481af150358e4bc0ffe6f",
}
require = gate.require
read = gate.read
sha = gate.sha

def paths(base, target):
    base=Path(base).resolve()
    target=Path(target).resolve()
    require(base.parent==ROOT and base.is_dir(), "Base must be a trial of THIS CGM3 host")
    require(target.parent==ROOT and re.fullmatch(r'alpha[0-9]{2,}',target.name),
            "Target must be a direct, new alphaNN child of this campaign")
    require(not target.exists(), "Target trial already exists")
    return base,target

def family_of(run):
    require(sha(FAMILY_FILE)==FAMILY_SHA,"CGM3 family changed")
    frozen=read(FAMILY_FILE)
    audit=read(run/'orbit_000.orbit.json')
    require(audit.get('model_key')==audit.get('family_key')==FAMILY_KEY and audit.get('model_key_inputs')==frozen,
            "Base IC is outside the exact CGM3 family")
    require(audit['source_sha256']==frozen['source_sha256'],"Wrong source realization")
    return frozen,audit

def awaiting_failure(base):
    campaign=read(ROOT/'campaign.json')
    require(not campaign.get('jet_authorized') and not campaign.get('accepted_trial') and
            not str(campaign.get('stage','')).startswith('orbit_passed'),"Campaign no longer eligible for orbit corrections")
    for status_path in ROOT.glob('*/production/status.json'):
        run=status_path.parents[1]
        state=read(status_path).get('state')
        if run!=base and state in {'starting','running','validating','exited','completed','simulation_completed'}:
            require((run/'endpoint_review/gate_result.json').is_file(), 'Another trial still needs its endpoint review')
    results={}
    for path in ROOT.glob('*/endpoint_review/gate_result.json'):
        run=path.parents[1]; decision=read(path)
        require(decision['model_key']==FAMILY_KEY and Path(decision['run_directory']).resolve()==run,
                "Published endpoint review does not bind the CGM3 trial")
        require(decision['state']=='orbit_failed' and decision['passed_components']<12,
                "A trial already passed or its gate is invalid")
        results[run]=decision
    require(base in results,"A verified unsuccessful CGM3 endpoint gate is required")
    if len(results)>1 or campaign.get('best_actual_run') is not None:
        selected=campaign.get('best_actual_run')
        require(isinstance(selected,str) and (ROOT/selected).resolve()==base,
                "Audit and designate campaign.best_actual_run before using multiple endpoints")
    require(float(results[base]['max_abs_sigma'])>4.0,"Use current-family live differences near the target (max ≤ 4)")
    return results[base]

def verify_evidence(base):
    frozen,audit=family_of(base)
    g=awaiting_failure(base)
    status_path=base/'production/status.json'
    status=read(status_path)
    require(status.get('state') in ('completed','simulation_completed') and status.get('simulation_returncode')==0,
            "Simulation did not finish successfully")
    if status['state']=='completed':
        require(status.get('returncode')==0 and status.get('wrapper_returncode',0)==0 and
                status.get('orbit_validation_returncode',0)==0,"Completion status unsuccessful")
    ic_sha,side,reference=gate.bind_run(base,status)
    require(side==audit and ic_sha==g['reference_ic_sha256'],"IC bound to previous gate changed")
    expected_parameters(frozen,base)
    snapshots=gate.snapshot_evidence(base,status)
    folder=Path(g['residual_csv']).resolve().parent
    require(folder in (base/'orbit_validation',base/'endpoint_review/validation'),"Unbound validation directory")
    paths_by_name,decision,d12=gate.verify_products(base,folder,reference,ic_sha,audit,snapshots,gate.semantic_functions())
    _,end,_=snapshots[-1]
    file_hash={str(p):sha(p) for p in paths_by_name.values()}
    require(file_hash==g['product_hashes'] and sha(status_path)==g['production_status_sha256'] and
            sha(end)==g['endpoint_snapshot']['sha256'] and str(end)==g['endpoint_snapshot']['path'],
            "Run, endpoint or validator artifacts changed")
    require(not decision['orbit_pass'] and decision['max_abs_sigma']>4 and np.isclose(d12,g['actual_D12'])
            and np.isclose(decision['max_abs_sigma'],g['max_abs_sigma']),"Actual gate and published scores differ")
    require(sha(gate.VALIDATOR)==g['validator_sha256']==gate.VALIDATOR_SHA,"Validator changed")
    require(sha(SETTINGS)==SETTINGS_SHA and all(sha(p)==h for p,h in DEPENDENCIES.items()),"Frozen static method changed")
    src=[Path(p).resolve(strict=True) for p in audit['source_files']]
    require([sha(p) for p in src]==frozen['source_sha256'],"Source snapshot hash mismatch")
    require(sha(Path(audit['gizmo']['path']))==frozen['gizmo_sha256'],"GIZMO changed")
    with h5py.File(base/'orbit_000.hdf5','r') as f:
        embedded=json.loads(f['MergeProvenance/AuditJSON'].asstr()[()])
        require(embedded=={k:v for k,v in audit.items() if k!='output_ic_sha256'},"Embedded audit differs")
        orbit=f['MagellanicOrbitReference']
        actual=np.r_[tuple(orbit[s+'_AppliedInitial'+q][:] for s in ('LMC','SMC') for q in ('Position_kpc','Velocity_kms'))]
        require(np.array_equal(actual,audit['applied_initial_12d']),"Embedded parent state differs")
        require(len(orbit['TimeFromPresent_Gyr'])==501,"Static ODE sample count changed")
    require(np.array_equal(g['residual_12d'],np.concatenate([decision['values'][sat]['residual'] for sat in ('lmc','smc')]))
            and np.array_equal(g['normalized_residual_12d'],decision['normalized_residual_12d'])
            and g['measurement_signature']==frozen['measurement_signature'] and
            g['residual_csv_sha256']==sha(Path(g['residual_csv'])),'Gate residual vector/CSV or measurement changed')
    return frozen,audit,g,decision

@contextlib.contextmanager
def exclusive_compute():
    # The same lock as the production/smoke driver; it exists after a real baseline run.
    with (ROOT/'status/compute.flock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(proc.read_text().strip()=='GIZMO' for proc in Path('/proc').glob('[0-9]*/comm') if proc.exists()):
            raise RuntimeError('GIZMO is running; postpone orbit correction')
        yield

def visible_only():
    require(os.isatty(0) and os.environ.get('TMUX_PANE'),"Execute the actual ODE/IC operation in an attached Ubuntu terminal")
    camp=read(ROOT/'campaign.json')
    attrs=subprocess.check_output(['tmux','-L',camp['terminal_server'],'display-message','-p','-t',
        os.environ['TMUX_PANE'],'#{session_name} #{session_attached} #{pane_tty}'],text=True).strip().split()
    require(len(attrs)==3 and attrs[0]==camp['terminal_session'] and int(attrs[1])>0 and
            os.path.realpath(attrs[2])==os.ttyname(0),"Wrong or detached terminal")

def make_proposal(base,target,alpha,pos_cap,vel_cap):
    base,target=paths(base,target)
    frozen,audit,g,decision=verify_evidence(base)
    require(np.isfinite(alpha) and 0<alpha<=1 and 0<pos_cap<=20 and 0<vel_cap<=30,"Invalid correction or caps")
    visible_only()
    cfg=read(SETTINGS)['model_key_inputs']
    require(cfg['backend_sha256']==DEPENDENCIES[MW/'tg19_backend.py'] and cfg['df'] is True and
            cfg['recoil'] is True and cfg['coulomb_log']==3,"Wrong static switches")
    centers=np.asarray(audit['source_centers_kpc'],dtype=float)
    models=[]
    for i,(path,center) in enumerate(zip(audit['source_files'],centers,strict=True)):
        pieces={}
        with h5py.File(path,'r') as h:
            units,_=resolve_units(h)
            require(units==frozen['raw_units'] and h['Header'].attrs['Time']==audit['source_times_code'][i],"Individually frozen source units/time changed")
            for p in range(6):
                if 'PartType'+str(p) not in h: continue
                grp=h['PartType'+str(p)]
                positions=np.asarray(grp['Coordinates'],dtype=float)
                masses=(np.asarray(grp['Masses'],dtype=float) if 'Masses' in grp else
                        np.full(len(positions),float(h['Header'].attrs['MassTable'][p])))
                require(positions.shape==(len(masses),3) and np.isfinite(positions).all() and
                        np.isfinite(masses).all() and (masses>0).all(),"Invalid enclosed-mass inputs")
                pieces[p]={'Coordinates':positions,'Masses':masses}
        build=merger.host_profile if i==0 else merger.MeasuredSatellite
        models.append(build({'parts':pieces},center,audit['units']))
    host,lmc,smc=models
    kwargs=dict(profile=host,lmc=lmc,smc=smc,lookback_gyr=audit['lookback_gyr'],use_dynamical_friction=cfg['df'],
                include_host_recoil=cfg['recoil'],coulomb_log=cfg['coulomb_log'],orbit_samples=501)
    def inverse(end):
        rp,rv,sp,sv=np.asarray(end,dtype=float).reshape(4,3)
        state=merger.b.integrate_magellanic_orbit(**kwargs,lmc_target_position=rp,lmc_target_velocity=rv,
                                                  smc_target_position=sp,smc_target_velocity=sv)[0].copy()
        state[[3,4,5,9,10,11]]/=merger.b.KMS_TO_KPC_GYR
        return state
    measured=np.concatenate([decision['values'][s]['residual'] for s in ('lmc','smc')])
    sigma=np.concatenate([decision['values'][s]['sigma'] for s in ('lmc','smc')])
    initial=np.asarray(audit['applied_initial_12d'],dtype=float)
    delta,diag=propose_local_step(initial,measured,inverse,lambda x:forward_state(x,kwargs),sigma,alpha,
                                 max_position_step_kpc=pos_cap,max_velocity_step_kms=vel_cap)
    require(diag['status']=='accepted_static_prediction' and np.any(delta),"Static update rejected")
    radii=np.array([5.,20.,50.,100.,200.])
    return plain(dict(version='cgm3-bounded-alpha-v1',family_key=FAMILY_KEY,status=diag['status'],target_run=str(target),
        base_run=str(base),base_ic=str(base/'orbit_000.hdf5'),base_ic_sha256=g['reference_ic_sha256'],
        residual_csv=g['residual_csv'],residual_csv_sha256=g['residual_csv_sha256'],binary_sha256=frozen['gizmo_sha256'],
        measurement_signature=g['measurement_signature'],target_time_code=gate.TARGET,base_initial_12d=initial,
        step_12d=delta,proposed_initial_12d=initial+delta,diagnostics=diag,alpha=alpha,max_position_step_kpc=pos_cap,
        max_velocity_step_kms=vel_cap,residual_12d=measured,sigma_12d=sigma,current_D12=g['actual_D12'],
        current_max_abs_sigma=g['max_abs_sigma'],endpoint_snapshot=g['endpoint_snapshot'],gate_sha256=sha(base/'endpoint_review/gate_result.json'),
        evidence_sha256=g['product_hashes'],source_sha256=frozen['source_sha256'],source_files=audit['source_files'],
        profile={'radii_kpc':radii,'enclosed_MW_msun':host.menc(radii),'friction_dm_density_msun_kpc3':host.dm_density(radii),
                 'satellite_masses_msun':[lmc.live_mass,smc.live_mass],'prescription':'actual spherical particle mass profiles; current host halo density for friction'},
        code_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'scripts/apply_orbit_step.py',Path(gate.__file__),
             ROOT/'scripts/stage_safety.py',*DEPENDENCIES]},static_switches_source_sha256=SETTINGS_SHA,live_convergence_verified=False))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run',required=True,type=Path)
    parser.add_argument('--trial-dir',required=True,type=Path)
    parser.add_argument('--alpha',type=float,default=.3)
    parser.add_argument('--max-position-step-kpc',type=float,default=20.)
    parser.add_argument('--max-velocity-step-kms',type=float,default=30.)
    args=parser.parse_args()
    base,target=paths(args.base_run,args.trial_dir)
    out=ROOT/(target.name+'_proposal.json')
    require(not out.exists(),'Proposal already exists')
    family_of(base)
    awaiting_failure(base)
    visible_only()
    with exclusive_compute():
        q=make_proposal(base,target,args.alpha,args.max_position_step_kpc,args.max_velocity_step_kms)
        awaiting_failure(base)
    with tempfile.NamedTemporaryFile(mode='w',dir=ROOT,prefix='.'+out.name+'.',delete=False) as f:
        tmp=Path(f.name)
        json.dump(q,f,indent=2,allow_nan=False)
        f.write('\n')
    try:
        os.link(tmp,out)
    finally:
        tmp.unlink()
    print(out,flush=True)

if __name__=='__main__':
    main()
