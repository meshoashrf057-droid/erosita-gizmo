#!/usr/bin/env python3
"""Exhaustive gas19 alpha IC check and direct formal runtime preparation.
No subcommand starts a simulation; no smoke run or smoke PASS is produced.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,os,shlex,shutil,sys
from pathlib import Path
import h5py
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'scripts')]
def require(x,m):
    if not x:raise ValueError(m)
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def equal(a,b):
    x=np.asarray(a);y=np.asarray(b)
    if x.shape!=y.shape or x.dtype!=y.dtype:return False
    return np.array_equal(x,y,equal_nan=True) if x.dtype.kind in 'fc' else np.array_equal(x,y)
def attrs_equal(a,b):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
def verify_rigid_copy(base,candidate,q):
    """All original fields/attrs/Header; only native-unit exact satellite phase translations."""
    before=sha(base);require(before==q['base_ic_sha256'],'Parent IC changed')
    counts={'phase_fields':0,'unchanged_internal_fields':0,'original_star_counts':{}}
    step=np.asarray(q['step_12d'],dtype=float);require(step.shape==(12,) and np.isfinite(step).all(),'Invalid step')
    with h5py.File(base,'r') as a,h5py.File(candidate,'r') as b:
        require(set(a)==set(b) and attrs_equal(a.attrs,b.attrs),'Top-level groups/attrs changed')
        require(attrs_equal(a['Header'].attrs,b['Header'].attrs),'Header changed')
        audit=json.loads(a['MergeProvenance/AuditJSON'].asstr()[()])
        units=[float(audit['units']['length_kpc']),float(audit['units']['velocity_kms'])]
        require(all(np.isfinite(units)) and min(units)>0,'Invalid conversion')
        for name in a:
            if not name.startswith('PartType'):continue
            g=a[name];h=b[name];require(set(g)==set(h) and attrs_equal(g.attrs,h.attrs),'Particle group changed')
            origins=a['MergeParticleOrigins'][name][...]
            require(equal(origins,b['MergeParticleOrigins'][name][...]),'Body identity map changed')
            for field in g:
                x=g[field][...];y=h[field][...]
                require(x.dtype==y.dtype and x.shape==y.shape and attrs_equal(g[field].attrs,h[field].attrs),'Field schema/attrs changed')
                if x.dtype.kind in 'fc':require(np.isfinite(x).all() and np.isfinite(y).all(),'Nonfinite particle field')
                if field in ('Coordinates','Velocities'):
                    off=0 if field=='Coordinates' else 3;u=units[off//3];wanted=x.copy()
                    for origin in (1,2):wanted[origins==origin]+=step[6*(origin-1)+off:6*(origin-1)+off+3]/u
                    require(equal(wanted,y) and equal(x[origins==0],y[origins==0]),'Incorrect rigid step/MW change')
                    counts['phase_fields']+=1
                else:
                    require(equal(x,y),'Internal field, ID, metal or stellar age changed '+name+'/'+field)
                    counts['unchanged_internal_fields']+=1
            if name=='PartType4':
                ids=h['ParticleIDs'][...]
                for origin,label,wanted,lo,hi in [(1,'LMC',5000,120000000,129999999),(2,'SMC',1600,210000000,219999999)]:
                    mask=(origins==origin)&(ids>=lo)&(ids<=hi)
                    require(int(mask.sum())==wanted,'Original-star population changed')
                    keys=np.column_stack([h[k][...][mask] for k in ('ParticleIDs','ParticleChildIDsNumber','ParticleIDGenerationNumber')])
                    require(len(np.unique(keys,axis=0))==wanted,'Composite original-star IDs not unique')
                    counts['original_star_counts'][label]=wanted
        for name in ('MagellanicOrbitReference','MergeParticleOrigins'):
            require(set(a[name])==set(b[name]),'Reference/origin dataset schema changed')
        for sat in ('LMC','SMC'):
            for suffix in ('_Position_kpc','_Velocity_kms','_Current_PositionSigma_kpc','_Current_VelocitySigma_kms'):
                k=sat+suffix
                require(equal(a['MagellanicOrbitReference'][k][...],b['MagellanicOrbitReference'][k][...]),'Observation/reference/sigma changed')
        actual=np.concatenate([b['MagellanicOrbitReference'][s+'_AppliedInitial'+v][...] for s in ('LMC','SMC') for v in ('Position_kpc','Velocity_kms')])
        require(np.array_equal(actual,q['proposed_initial_12d']),'Applied state differs from proposal')
    require(sha(base)==before,'Original IC changed while checking')
    return dict(status='pass',base_ic_sha256=before,**counts)
def prepare(trial,proposal):
    import gas19_alpha_review01 as model
    trial=Path(trial).resolve(strict=True);p=Path(proposal).resolve(strict=True);q=read(p)
    require(trial==ROOT/'alpha01' and p==ROOT/'alpha01_proposal.json' and q['target_run']==str(trial),'Wrong target')
    require(q['version']=='gas19-bounded-alpha-v1' and q['alpha']==.3,'Wrong single alpha correction')
    for n in ('production','run_configuration.json','production_ready.json','production_launch_clearance.json','initial_review.json','start_production.sh'):
        require(not (trial/n).exists(),'Prior preparation exists; do not repeat '+n)
    model.verify_evidence(q['base_run'],*model.proposal_arguments(q)[2:])
    require(q['evidence']['code_sha256']==model.sources_and_pins(),'Proposal code changed')
    base=Path(q['base_run']);cfg0=read(base/'run_configuration.json');cfg=copy.deepcopy(cfg0)
    ic=trial/'orbit_000.hdf5';side=trial/'orbit_000.orbit.json'
    require(sha(ic)==read(side)['output_ic_sha256'] and read(side)['shooting_proposal_sha256']==sha(p),'New IC/proposal differs')
    report=verify_rigid_copy(Path(q['base_ic']),ic,q)
    report.update(proposal=str(p),proposal_sha256=sha(p),candidate=str(ic),candidate_sha256=sha(ic),sidecar_sha256=sha(side),
                  family_key=model.FAMILY,no_smoke='Explicit user instruction; no smoke run or PASS')
    prod=trial/'production';prod.mkdir();(prod/'output').mkdir()
    source_param=base/'production/stage2.param';require(sha(source_param)==model.BASE_PARAM_SHA,'Parent params changed')
    with source_param.open('rb') as a,(prod/'stage2.param').open('xb') as b:shutil.copyfileobj(a,b)
    require(sha(prod/'stage2.param')==model.BASE_PARAM_SHA,'Parameter byte-copy failed')
    deps=[]
    for item in cfg0['runtime_dependencies']:
        require(item['mode']=='production','Unexpected runtime phase');rel=Path(item['relative_path'])
        require(not rel.is_absolute() and rel.parts[0]=='production' and '..' not in rel.parts,'Unsafe dependency path')
        src=base/rel;dst=trial/rel;require(sha(src)==item['sha256'],'Parent dependency changed')
        dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b)
        require(sha(src)==sha(dst)==item['sha256'] and dst.stat().st_nlink==1,'Dependency copy/source changed')
        deps.append(dict(item,source=str(src)))
    require(len(deps)==len(cfg0['runtime_dependencies'])==50,'Incomplete dependency copies')
    cfg.update(schema='gas19-alpha-direct-formal-runtime-v1',trial_dir=str(trial),ic_sha256=sha(ic),sidecar_sha256=sha(side),
        param_sha256={'production':sha(prod/'stage2.param')},runtime_dependencies=deps,smoke_required=False,
        alpha_parent=str(base),alpha_proposal=str(p),alpha_proposal_sha256=sha(p))
    require(cfg['restart_flag']==2 and cfg['mpi_tasks']==6 and cfg['omp_threads']==1 and cfg['nice']==19,'Execution settings changed')
    policy=ROOT/'provenance/user_no_smoke_policy.json';require(read(policy)['smoke_required'] is False,'No-smoke policy missing')
    save(trial/'initial_review.json',report);save(trial/'run_configuration.json',cfg)
    ready=dict(status='pass',ic_sha256=cfg['ic_sha256'],sidecar_sha256=cfg['sidecar_sha256'],
        production_param_sha256=cfg['param_sha256']['production'],binary_sha256=cfg['binary']['sha256'],
        smoke_required=False,user_no_smoke_policy_sha256=sha(policy),initial_review_sha256=sha(trial/'initial_review.json'),
        alpha_proposal_sha256=sha(p),alpha_code_sha256=model.sources_and_pins(),
        release_basis='Actual exhaustive static input checks and user no-smoke policy; no smoke PASS')
    save(trial/'production_ready.json',ready)
    save(trial/'production_launch_clearance.json',dict(status='pass',ready_sha=sha(trial/'production_ready.json'),
        ic_sha256=cfg['ic_sha256'],user_no_smoke_policy_sha256=sha(policy),smoke_required=False))
    python='/home/xingdi/gizmo-runs/.venv/bin/python'
    guardcmd=[python,'-B','-u',str(Path(__file__).resolve()),'guard','--trial-dir',str(trial)]
    drive=[python,'-B','-u',str(ROOT/'scripts/drive_stage.py'),'--trial-dir',str(trial),'--mode','production']
    shell='#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(trial))+'\n'
    shell+=shlex.join(guardcmd)+'\nexport GIZMO_VISIBLE_TTY=$(tty)\nexec script -q -e -f -c '
    shell+=shlex.quote(shlex.join(drive))+' '+shlex.quote(str(trial/'production_terminal_session.log'))+'\n'
    with (trial/'start_production.sh').open('x') as f:f.write(shell)
    (trial/'start_production.sh').chmod(0o755)
    import stage_safety as safety
    safety.verify_before_run(trial,'production');print(json.dumps(ready,indent=2),flush=True)
def guard(trial):
    import gas19_alpha_review01 as model
    import run_trial,stage_safety
    trial=Path(trial).resolve(strict=True);require(trial==ROOT/'alpha01','Wrong alpha trial');model.visible_only()
    ready=read(trial/'production_ready.json');q=read(ROOT/'alpha01_proposal.json')
    require(ready['alpha_code_sha256']==model.sources_and_pins(),'Alpha source changed')
    require(ready['alpha_proposal_sha256']==sha(ROOT/'alpha01_proposal.json'),'Proposal changed')
    require(ready['initial_review_sha256']==sha(trial/'initial_review.json'),'Initial review changed')
    for p,h in q['evidence_sha256'].items():require(sha(p)==h,'Endpoint evidence changed')
    require(sha(q['evidence']['gate_result'])==q['gate_sha256'],'Original gate changed')
    run_trial.preflight(trial,'production');stage_safety.verify_before_run(trial,'production')
def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    x=s.add_parser('prepare');x.add_argument('--trial-dir',required=True,type=Path);x.add_argument('--proposal',required=True,type=Path)
    x=s.add_parser('guard');x.add_argument('--trial-dir',required=True,type=Path);a=p.parse_args()
    if a.command=='prepare':
        import gas19_alpha_review01 as model
        with model.exclusive_compute():prepare(a.trial_dir,a.proposal)
    else:guard(a.trial_dir)
if __name__=='__main__':main()
