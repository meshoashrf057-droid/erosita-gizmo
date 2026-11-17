#!/usr/bin/env python3
"""Source-only adapter for one gas19 alpha=.3 correction; no live Jacobian/probes.
Only invoke after the true complete new-family endpoint was independently gated.
"""
from __future__ import annotations
import argparse,contextlib,fcntl,hashlib,json,math,os,re,subprocess,sys
from pathlib import Path
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1];MW=ROOT.parent;PROJECT=MW.parent
sys.path[:0]=[str(ROOT/'scripts'),str(MW),str(PROJECT),str(Path(__file__).parent)]
import gate_endpoint as gate
import stage_safety as safety
import merge_relaxed_mw_lmc_smc as merger
from satellite_relax_common import plain
from gizmo_units import resolve_units
from shooting_local import forward_state,propose_local_step
FAMILY='cpp-shooting-family:1877384654334dc2bb7735d26e625cb03f7bb15e5a48738c03f53a10281a1040'
FAMILY_SHA='0a7bda9b2de9f4118f1b18ba9adf405102a55aa361fc38b82b4b909851d8bd51'
BASE_IC_SHA='81c53a9ea40bdd7fb42b7b154a99fc6c18a46ba4820de0411c7e90f61262aa72'
BASE_PARAM_SHA='3a325a991983dfeef2cfb148414541a44a9d7b5e393584af3f640dbc9d5b95c3'
SETTINGS=MW/'mw_lmc_smc_relaxed_trial13_000.orbit.json'
SETTINGS_SHA='46debe8151f28c6de0f16272f6c39741f472d5a932cc551a0dc201edaeae5802'
PINS={
 MW/'shooting_local.py':'4d76cbddba7f1691c544904e7b83722832d0b436fc27f7fcb6605d78edc5bf2e',
 MW/'tg19_backend.py':'6bad4ae89405b3095092fb65f472272c92a6f9900277a19ca03f71f64ab4d37a',
 MW/'merge_relaxed_mw_lmc_smc.py':'ff71bd1dc6ab4a37a5843a051fbb7c26231d96ee23eb6236eb20417c1f77ad6c',
 MW/'satellite_relax_common.py':'a7ff947aa0d5e2fb23025759d942b378345624fad026d75928c54097b4d3f0d7',
 PROJECT/'gizmo_units.py':'215145d94e30b945eb61739975b73d6bd780c07731728e8e17ec05aeb1517fe2',
 ROOT/'scripts/gate_endpoint.py':'12c2fb75fb80dc5acbae408b223cdb84c921da1e9acf15cda1e3f984a3d1ca4d'}
HOST_ARCHIVE=Path('/mnt/i/erosita数据/mw/11.5/cgm3_n3000_isolated_20261104/snapshots/snapshot_013.hdf5')
GLOBAL_LEASE=MW/'cgm3_n3000_orbits_cpp_20261105_084436/status/compute.flock'
require=gate.require;read=gate.read;sha=gate.sha
def save_new(p,value):
    with Path(p).open('x') as f:json.dump(plain(value),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def sources_and_pins():
    require(sha(ROOT/'provenance/cgm3_family_inputs.json')==FAMILY_SHA,'Gas19 family changed')
    for p,h in PINS.items():require(sha(p)==h,'Static code changed: '+str(p))
    require(sha(SETTINGS)==SETTINGS_SHA,'Static switches source changed')
    launch=ROOT/'provenance/launch_hashes.json'
    for p,h in read(launch)['sha256'].items():require(sha(Path(p))==h,'Launch dependency changed: '+p)
    own=[Path(__file__),Path(__file__).with_name('gas19_rigid_writer_review01.py'),
         Path(__file__).with_name('gas19_alpha_runtime_review01.py'),
         Path(__file__).with_name('gas19_alpha_production_review01.py')]
    return {str(p):sha(p) for p in [*PINS,SETTINGS,launch,*own]}
def visible_only():
    require(os.isatty(0) and os.isatty(1) and os.environ.get('TMUX_PANE'),'Attached visible terminal required')
    c=read(ROOT/'campaign.json')
    s=subprocess.check_output(['tmux','-L',c['terminal_server'],'display-message','-p','-t',os.environ['TMUX_PANE'],
                               '#{session_name} #{session_attached} #{pane_pid}'],text=True).split()
    require(len(s)==3 and s[0]==c['terminal_session'] and int(s[1])>0,'Wrong/detached tmux session')
    pid=os.getpid();chain=[]
    while pid>1:
        chain.append(pid);text=(Path('/proc')/str(pid)/'stat').read_text()
        pid=int(text[text.rfind(')')+2:].split()[1])
    require(int(s[2]) in chain,'Process is outside the original attached pane')
@contextlib.contextmanager
def exclusive_compute():
    visible_only()
    # Existing resume02 global lease first, then this campaign's direct-driver lease.
    with GLOBAL_LEASE.open('r+') as global_lock:
        fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (ROOT/'status/compute.flock').open('r+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            require(not safety.compute_processes(),'GIZMO/MPI already running')
            yield
def paths(base,target,output_absent=True):
    base=Path(base).resolve(strict=True);target=Path(target).resolve()
    require(base==ROOT/'calibration01','Bound only to this gas19 calibration01')
    require(target==ROOT/'alpha01','Bound only to the first independent gas19 alpha01')
    if output_absent:require(not target.exists(),'Trial exists; inspect instead of repeating')
    return base,target
def read_family(base):
    c=read(ROOT/'campaign.json');f=read(ROOT/'provenance/cgm3_family_inputs.json')
    require(c['family_key']==FAMILY and c['family_inputs_sha256']==FAMILY_SHA,'Wrong campaign family')
    require('cpp-shooting-family:'+hashlib.sha256(json.dumps(f,sort_keys=True,allow_nan=False).encode()).hexdigest()==FAMILY,'Family fingerprint changed')
    require(c.get('smoke_required') is False and c.get('jet_authorized') is False and c.get('agn_authorized') is False,'Wrong user scope')
    require(not c.get('accepted_trial') and not (ROOT/'accepted_orbit.json').exists(),'A real orbit passed; stop')
    a=read(base/'orbit_000.orbit.json')
    require(a['family_key']==a['model_key']==FAMILY and a['model_key_inputs']==f,'Wrong IC family')
    require(a['source_sha256']==f['source_sha256'] and a['rotation_matrices']==f['rotation_matrices'],'Wrong sources/rotation')
    require(a['output_ic_sha256']==BASE_IC_SHA and sha(base/'orbit_000.hdf5')==BASE_IC_SHA,'Original gas19 IC changed')
    require(a['lookback_gyr']==1. and a['stage2_start_code']==gate.START and a['stage2_target_code']==float(gate.TARGET),'Wrong full physical-Gyr interval')
    require(c['no_previous_jacobian_reuse'] is True,'Old response reuse forbidden')
    return c,f,a
def verify_status(report,base,status_file,gate_status_file):
    p=Path(status_file).resolve(strict=True);before=Path(gate_status_file).resolve(strict=True)
    st=read(p);old=read(before)
    require(p.parent.parent==base and Path(st['cwd']).resolve()==p.parent,'Status outside base runtime')
    require(sha(before)==report['production_status_sha256'],'No exact immutable pre-gate status evidence')
    require(old.get('state') in ('completed','simulation_completed') and old.get('simulation_returncode')==0,'Gate did not bind successful simulation')
    mutable={'state','ended_utc','returncode','wrapper_returncode','orbit_validation_returncode',
             'validation_command','validation_pid','validation_started_utc','validation_ended_utc','orbit_validation_note'}
    require({k:v for k,v in old.items() if k not in mutable}=={k:v for k,v in st.items() if k not in mutable},'Scientific/run fields changed after gate')
    require(st.get('state') in ('completed','simulation_completed') and st.get('simulation_returncode')==0,'Run incomplete')
    if st['state']=='completed':
        require(st.get('returncode')==0 and st.get('wrapper_returncode',0)==0 and st.get('orbit_validation_returncode')==0,'Wrapper/validator failed')
    for key,wanted in [('ic_sha256',BASE_IC_SHA),('binary_sha256',gate.BINARY_SHA),('param_sha256',BASE_PARAM_SHA)]:
        require(old[key]==st[key]==wanted,'Runtime binding differs: '+key)
    require(st['sidecar_sha256']==sha(base/'orbit_000.orbit.json'),'Sidecar changed')
    require(sha(p.parent/'stage2.param')==BASE_PARAM_SHA,'Runtime parameters changed')
    require(st['command'][-3:] in ([str(PROJECT.parent/'gizmo-public-gizmo-cpp/GIZMO'),'stage2.param','1'],
                                 [str(PROJECT.parent/'gizmo-public-gizmo-cpp/GIZMO'),'stage2.param','2']),'Wrong native invocation')
    require(st.get('final_snapshot_time') is not None and gate.same_time(st['final_snapshot_time'],gate.TARGET),'True endpoint not reached')
    return p,st,before
def verify_evidence(base,gate_result,status_file,gate_status_file,near_authorization=None):
    base=Path(base).resolve(strict=True)
    pins=sources_and_pins();c,f,a=read_family(base);gp=Path(gate_result).resolve(strict=True)
    require(gp==base/'endpoint_review/gate_result.json','Not canonical new-family gate')
    g=read(gp)
    require(g['state']=='orbit_failed' and g['total_components']==12 and g['passed_components']<12,'No valid failed gate')
    require(g['model_key']==FAMILY and Path(g['run_directory']).resolve()==base,'Wrong gate family/trial')
    require(g['reference_ic_sha256']==BASE_IC_SHA and g['family_input_sha256']==f['source_sha256'],'Wrong gate IC/sources')
    require(g['measurement_signature']==f['measurement_signature'],'Measurement changed')
    p,st,before=verify_status(g,base,status_file,gate_status_file);output=p.parent/'output';vc=g['validation_command']
    require(Path(vc[vc.index('--snapshot-dir')+1]).resolve()==output and Path(vc[vc.index('--reference-ic')+1]).resolve()==base/'orbit_000.hdf5','Validator input differs')
    snapshots=[]
    for s in output.glob('snapshot_*.hdf5'):
        require(re.fullmatch(r'snapshot_[0-9]+\.hdf5',s.name) and s.resolve().parent==output,'Redirected snapshot')
        with h5py.File(s,'r') as h:
            require(int(h['Header'].attrs.get('NumFilesPerSnapshot',1))==1,'Incomplete snapshot')
            t=float(h['Header'].attrs['Time']);require(np.isfinite(t) and t>=gate.START-1e-10,'Invalid frame time')
        snapshots.append((t,s,gate.stamp(s)))
    snapshots.sort()
    require(snapshots and all(x[0]<y[0] for x,y in zip(snapshots,snapshots[1:])),'Incomplete/duplicate history')
    t,end,_=snapshots[-1]
    require(gate.same_time(t,gate.TARGET) and gate.same_time(g['endpoint_snapshot']['time_code'],gate.TARGET)
            and str(end)==g['endpoint_snapshot']['path']==st['final_snapshot'] and sha(end)==g['endpoint_snapshot']['sha256'],'Not true bound final frame')
    reference={'target_time_code':float(gate.TARGET)}
    with h5py.File(base/'orbit_000.hdf5','r') as h:
        embedded=json.loads(h['MergeProvenance/AuditJSON'].asstr()[()])
        require(embedded=={k:v for k,v in a.items() if k!='output_ic_sha256'},'Embedded IC audit differs')
        orbit=h['MagellanicOrbitReference'];reference['unit_time_gyr']=float(orbit.attrs['UnitTime_Gyr'])
        require(math.isclose((float(gate.TARGET)-gate.START)*reference['unit_time_gyr'],1.,rel_tol=0,abs_tol=1e-14),'Not exactly 1 physical Gyr')
        initial=np.concatenate([orbit[s+'_AppliedInitial'+v][:] for s in ('LMC','SMC') for v in ('Position_kpc','Velocity_kms')])
        require(np.array_equal(initial,a['applied_initial_12d']),'Base initial metadata differs')
        for sat in ('lmc','smc'):
            cap=sat.upper();obs=np.r_[orbit[cap+'_Position_kpc'][-1],orbit[cap+'_Velocity_kms'][-1]]
            sigma=np.r_[orbit[cap+'_Current_PositionSigma_kpc'][:],orbit[cap+'_Current_VelocitySigma_kms'][:]]
            require(np.allclose(obs,gate.OBS[sat],rtol=0,atol=1e-12) and np.allclose(sigma,gate.SIGMA[sat],rtol=0,atol=1e-12),'Observation/sigma changed')
            reference[sat]={'observed':obs.tolist(),'sigma':sigma.tolist()}
    folder=Path(g['residual_csv']).resolve().parent
    require(folder in (base/'orbit_validation',base/'endpoint_review/validation'),'Unbound validation folder')
    products,d,d12=gate.verify_products(base,folder,reference,BASE_IC_SHA,a,snapshots,gate.semantic_functions())
    require({str(p):sha(p) for p in products.values()}==g['product_hashes'],'Validation products changed')
    require(not d['orbit_pass'] and np.array_equal(d['normalized_residual_12d'],g['normalized_residual_12d'])
            and math.isclose(d12,g['actual_D12'],rel_tol=1e-12) and d['max_abs_sigma']==g['max_abs_sigma'],'Published gate differs')
    residual=np.concatenate([d['values'][s]['residual'] for s in ('lmc','smc')])
    require(np.array_equal(residual,g['residual_12d']) and sha(Path(g['residual_csv']))==g['residual_csv_sha256'],'Residual/CSV differs')
    require(sha(gate.VALIDATOR)==g['validator_sha256']==gate.VALIDATOR_SHA and sha(gate.SEMANTICS)==g['semantic_functions_sha256']==gate.SEMANTICS_SHA,'Validator changed')
    route={'default_coarse_max_gt_4':True,'authorization':None}
    if g['max_abs_sigma']<=4:
        require(near_authorization is not None,'Near-target default retained; explicit current single-alpha routing required')
        auth=Path(near_authorization).resolve(strict=True);r=read(auth)
        require(auth.parent==ROOT/'provenance' and r.get('schema')=='gas19-single-alpha-user-routing/v1'
                and r.get('authorized') is True and r.get('allow_near_alpha') is True and r.get('family_key')==FAMILY
                and r.get('base_ic_sha256')==BASE_IC_SHA and r.get('gate_sha256')==sha(gp) and r.get('alpha')==.3
                and r.get('scope')=='single gas19 alpha01 after this exact failed endpoint','Unbound near-alpha authorization')
        route={'default_coarse_max_gt_4':False,'authorization':{'path':str(auth),'sha256':sha(auth)}}
    selected=c.get('best_actual_run')
    require(selected is None or Path(selected if str(selected).startswith('/') else ROOT/selected).resolve()==base,'Another endpoint is best')
    require(all(gate.stamp(s)==stamp for _,s,stamp in snapshots),'History changed during review')
    proof={'gate_result':str(gp),'gate_sha256':sha(gp),'status_file':str(p),'status_sha256':sha(p),
           'gate_status_file':str(before),'gate_status_sha256':sha(before),'routing':route,'code_sha256':pins,'endpoint_snapshot':g['endpoint_snapshot']}
    return f,a,g,d,proof
def make_proposal(base,target,gate_result,status_file,gate_status_file,near_authorization=None):
    base,target=paths(base,target)
    f,a,g,d,proof=verify_evidence(base,gate_result,status_file,gate_status_file,near_authorization)
    cfg=read(SETTINGS)['model_key_inputs']
    require(cfg['df'] is True and cfg['recoil'] is True and cfg['coulomb_log']==3 and cfg['backend_sha256']==PINS[MW/'tg19_backend.py'],'Static switches changed')
    models=[];resolved=[]
    for i,(original,expected,center) in enumerate(zip(a['source_files'],f['source_sha256'],a['source_centers_kpc'],strict=True)):
        p=Path(original)
        if not p.is_file():
            require(i==0 and expected==gate.HOST_SHA,'Only same-SHA archived host replaces unavailable path');p=HOST_ARCHIVE
        require(sha(p)==expected,'Profile source changed');pieces={}
        with h5py.File(p,'r') as h:
            units,_=resolve_units(h)
            require(units==f['raw_units'] and h['Header'].attrs['Time']==a['source_times_code'][i],'Source epoch/units changed')
            for j in range(6):
                if 'PartType'+str(j) not in h:continue
                grp=h['PartType'+str(j)];positions=np.asarray(grp['Coordinates'],dtype=float)
                masses=(np.asarray(grp['Masses'],dtype=float) if 'Masses' in grp else np.full(len(positions),float(h['Header'].attrs['MassTable'][j])))
                require(positions.shape==(len(masses),3) and np.isfinite(positions).all() and np.isfinite(masses).all() and (masses>0).all(),'Invalid profile inputs')
                pieces[j]={'Coordinates':positions,'Masses':masses}
        models.append((merger.host_profile if i==0 else merger.MeasuredSatellite)({'parts':pieces},np.asarray(center),a['units']))
        require(sha(p)==expected,'Profile source changed while read')
        resolved.append({'original':original,'actual_read_path':str(p),'sha256':expected})
    host,lmc,smc=models
    kwargs=dict(profile=host,lmc=lmc,smc=smc,lookback_gyr=a['lookback_gyr'],use_dynamical_friction=cfg['df'],include_host_recoil=cfg['recoil'],coulomb_log=cfg['coulomb_log'],orbit_samples=501)
    def inverse(end):
        rp,rv,sp,sv=np.asarray(end,dtype=float).reshape(4,3)
        state=merger.b.integrate_magellanic_orbit(**kwargs,lmc_target_position=rp,lmc_target_velocity=rv,smc_target_position=sp,smc_target_velocity=sv)[0].copy()
        state[[3,4,5,9,10,11]]/=merger.b.KMS_TO_KPC_GYR
        return state
    residual=np.concatenate([d['values'][s]['residual'] for s in ('lmc','smc')])
    sigma=np.concatenate([d['values'][s]['sigma'] for s in ('lmc','smc')])
    initial=np.asarray(a['applied_initial_12d'],dtype=float)
    delta,diag=propose_local_step(initial,residual,inverse,lambda x:forward_state(x,kwargs),sigma,.3,max_position_step_kpc=20.,max_velocity_step_kms=30.)
    require(diag['status']=='accepted_static_prediction' and np.any(delta),'Original bounded static update rejected')
    radii=np.array([5.,20.,50.,100.,200.])
    return plain(dict(version='gas19-bounded-alpha-v1',family_key=FAMILY,status=diag['status'],target_run=str(target),base_run=str(base),
        base_ic=str(base/'orbit_000.hdf5'),base_ic_sha256=BASE_IC_SHA,residual_csv=g['residual_csv'],residual_csv_sha256=g['residual_csv_sha256'],
        binary_sha256=f['gizmo_sha256'],measurement_signature=g['measurement_signature'],target_time_code=float(gate.TARGET),base_initial_12d=initial,
        step_12d=delta,proposed_initial_12d=initial+delta,diagnostics=diag,alpha=.3,max_position_step_kpc=20.,max_velocity_step_kms=30.,
        residual_12d=residual,sigma_12d=sigma,current_D12=g['actual_D12'],current_max_abs_sigma=g['max_abs_sigma'],endpoint_snapshot=g['endpoint_snapshot'],
        gate_sha256=proof['gate_sha256'],evidence_sha256=g['product_hashes'],source_sha256=f['source_sha256'],source_files=a['source_files'],
        resolved_profile_sources=resolved,profile={'radii_kpc':radii,'enclosed_MW_msun':host.menc(radii),'friction_dm_density_msun_kpc3':host.dm_density(radii),
        'satellite_masses_msun':[lmc.live_mass,smc.live_mass],'prescription':'fresh actual gas19 spherical particle mass profiles'},evidence=proof,
        live_convergence_verified=False,smoke_required=False,old_live_jacobian_reused=False))
def proposal_arguments(q):
    e=q['evidence'];a=e['routing']['authorization']
    return(q['base_run'],q['target_run'],e['gate_result'],e['status_file'],e['gate_status_file'],a['path'] if a else None)
def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True);p=sub.add_parser('propose')
    for name in ('base-run','trial-dir','gate-result','status-file','gate-status-file'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--near-authorization',type=Path)
    p=sub.add_parser('apply');p.add_argument('--proposal',required=True,type=Path);p.add_argument('--trial-dir',required=True,type=Path);a=ap.parse_args()
    with exclusive_compute():
        if a.command=='propose':
            base,target=paths(a.base_run,a.trial_dir);out=ROOT/(target.name+'_proposal.json');require(not out.exists(),'Proposal exists')
            q=make_proposal(base,target,a.gate_result,a.status_file,a.gate_status_file,a.near_authorization);save_new(out,q);print(out,flush=True)
        else:
            p=a.proposal.resolve(strict=True);q=read(p);base,target=paths(q['base_run'],a.trial_dir)
            require(p==ROOT/(target.name+'_proposal.json') and q['target_run']==str(target),'Wrong proposal target')
            before=sha(p);fresh=make_proposal(*proposal_arguments(q));require(fresh==q and sha(p)==before,'Fresh recomputation differs')
            from gas19_rigid_writer_review01 import apply
            apply(p,target,ROOT/'campaign.json',target.name)
if __name__=='__main__':main()
