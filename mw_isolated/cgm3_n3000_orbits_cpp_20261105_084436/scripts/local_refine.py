"""Prepare and audit small same-anchor CGM3 live orbital differences, one complete trial at a time."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import re
import tempfile
import h5py
import numpy as np

import propose_alpha as coarse
from live_endpoint_evidence import verify_evidence
from gate_endpoint import read, require, sha
from satellite_relax_common import plain

ROOT=coarse.ROOT
MATH=coarse.MW/'shooting_local_live_v1/local_live_math.py'
MATH_SHA='498cd8f17b9f886428cdaaa5913bb4296b629c8690929768059ca7de5a56aba0'
ORDER=[s+'_'+c for s in ('lmc','smc') for c in ('x','y','z','vx','vy','vz')]
VERSION='cgm3-live-difference-v1'

def exclusive_json(path,record):
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.'+path.name+'.',delete=False) as f:
        tmp=Path(f.name)
        json.dump(record,f,indent=2,allow_nan=False)
        f.write('\n')
    try:
        os.link(tmp,path)
    finally:
        tmp.unlink()

def stop_if_done():
    camp=read(ROOT/'campaign.json')
    require(camp.get('family_key')==coarse.FAMILY_KEY and not camp['jet_authorized']
            and not camp.get('accepted_trial') and not str(camp['stage']).startswith('orbit_passed'),
            'Campaign has stopped or no longer matches CGM3')
    for path in ROOT.glob('*/endpoint_review/gate_result.json'):
        data=read(path)
        require(data['state']=='orbit_failed' and data['passed_components']<12 and
                data['model_key']==coarse.FAMILY_KEY,'A trial has passed or gate is inconsistent: '+str(path))
    for path in ROOT.glob('*/production/status.json'):
        state=read(path).get('state')
        if state in {'starting','running','validating','exited','simulation_completed','completed'}:
            require((path.parents[1]/'endpoint_review/gate_result.json').is_file(),'Awaiting endpoint review: '+str(path))
    return camp

def code_hashes():
    require(sha(MATH)==MATH_SHA,'Frozen pure minimax implementation changed')
    files=[Path(__file__),ROOT/'scripts/local_rigid_writer.py',ROOT/'scripts/live_endpoint_evidence.py',
           ROOT/'scripts/propose_alpha.py',ROOT/'scripts/gate_endpoint.py',ROOT/'scripts/stage_safety.py',MATH]
    return {str(p):sha(p) for p in files}

def spec(group, axis, value):
    require(re.fullmatch(r'local[0-9]{2,}',group),'Use localNN for a group')
    require(axis in ORDER,'Unsupported coordinate')
    h=float(value)
    require(np.isfinite(h) and 0<abs(h)<=.5,'Explicit signed difference must be nonzero, at most 0.5 in its native unit')
    unit='km/s' if axis.split('_')[1].startswith('v') else 'kpc'
    return dict(coordinate=axis,index=ORDER.index(axis),signed_step=h,unit=unit,target=str(ROOT/(group+'_'+axis)))

def plan_path(group):
    require(re.fullmatch(r'local[0-9]{2,}',group),'Use localNN for a group')
    return ROOT/'provenance'/(group+'_plan.json')

def freeze_plan(group, anchor, axes, rationale):
    destination=plan_path(group)
    require(not destination.exists(),'Group already frozen')
    camp=stop_if_done()
    anchor=Path(anchor).resolve(strict=True)
    require(anchor.parent==ROOT and (ROOT/camp['best_actual_run']).resolve()==anchor,'Select the reviewed best actual CGM3 anchor')
    _,audit,g,_=verify_evidence(anchor)
    require(3<g['max_abs_sigma']<=4,'Anchor must be close to target with at least one real failed component')
    probes=[spec(group,*pair.split('=',1)) for pair in axes]
    require(len(set(p['coordinate'] for p in probes))==len(probes)==3,'Choose three distinct explicit axes for this group')
    require(all(not Path(p['target']).exists() for p in probes) and not (ROOT/(group+'_joint')).exists(),'Trial name in use')
    require(bool(rationale.strip()),'Record the new-family scientific reason for this choice of axes')
    return dict(version=VERSION,family_key=coarse.FAMILY_KEY,group=group,anchor=str(anchor),
                anchor_ic_sha256=g['reference_ic_sha256'],anchor_gate_sha256=sha(anchor/'endpoint_review/gate_result.json'),
                anchor_applied_12d=audit['applied_initial_12d'],axes_reason=rationale,probes=probes,
                joint_target=str(ROOT/(group+'_joint')),code_sha256=code_hashes())

def load_plan(group):
    stop_if_done()
    path=plan_path(group)
    plan=read(path)
    require(plan['version']==VERSION and plan['group']==group and plan['family_key']==coarse.FAMILY_KEY and
            plan['code_sha256']==code_hashes() and len(plan['probes'])==3,'Plan implementation or binding differs')
    probes=[spec(group,p['coordinate'],p['signed_step']) for p in plan['probes']]
    require(probes==plan['probes'] and len({p['coordinate'] for p in probes})==3 and
            plan['joint_target']==str(ROOT/(group+'_joint')),'Bad plan geometry/path')
    anchor=Path(plan['anchor']).resolve(strict=True)
    require(anchor.parent==ROOT and sha(anchor/'endpoint_review/gate_result.json')==plan['anchor_gate_sha256'],
            'Actual CGM3 anchor gate differs')
    _,audit,g,decision=verify_evidence(anchor)
    require(3<decision['max_abs_sigma']<=4 and plan['anchor_ic_sha256']==g['reference_ic_sha256'] and
            plan['anchor_applied_12d']==audit['applied_initial_12d'],'Frozen anchor differs')
    return plan,audit,g

def one_axis(p):
    v=np.zeros(12)
    v[p['index']]=p['signed_step']
    return v

def check_initial_delta(anchor,probe,step,units):
    # Check the phase-space change and every internal/identity field directly against the common anchor.
    with h5py.File(anchor,'r') as fa,h5py.File(probe,'r') as fp:
        groups=[name for name in fa if name.startswith('PartType')]
        require(set(groups)=={name for name in fp if name.startswith('PartType')},'Changed particle types')
        for name in groups:
            a,b=fa[name],fp[name]
            origin=fa['MergeParticleOrigins'][name][:]
            require(set(a)==set(b) and np.array_equal(origin,fp['MergeParticleOrigins'][name][:]),'Changed field/origin schema')
            for field in a:
                require(a[field].shape==b[field].shape and a[field].dtype==b[field].dtype,'Changed field shape/dtype')
                for i in range(0,len(origin),150000):
                    sub=slice(i,i+150000)
                    expected=a[field][sub]
                    if field in ('Coordinates','Velocities'):
                        k=0 if field=='Coordinates' else 3
                        unit=units['length_kpc'] if k==0 else units['velocity_kms']
                        for tag in (1,2):
                            j=6*(tag-1)+k
                            expected[origin[sub]==tag]+=step[j:j+3]/unit
                    require(np.array_equal(expected,b[field][sub]),'Probe differs from one rigid single-axis increment: '+name+'/'+field)

def completed_probe(plan,p,anchor_audit,full_check):
    trial=Path(p['target']);anchor=Path(plan['anchor'])
    gatefile=trial/'endpoint_review/gate_result.json'
    require(gatefile.is_file(),'Need actual probe review: '+str(gatefile))
    _,audit,g,decision=verify_evidence(trial)
    step=one_axis(p)
    source_proposal=ROOT/(trial.name+'_proposal.json')
    verification=read(trial/'ic_verification.json')
    require(verification['status']=='pass' and verification['output_ic_sha256']==g['reference_ic_sha256'] and
            audit['parent_ic']==str(anchor/'orbit_000.hdf5') and audit['parent_ic_sha256']==plan['anchor_ic_sha256'] and
            audit['anchor_gate_sha256']==plan['anchor_gate_sha256'] and audit['alpha_applied'] is False and
            audit['local_role']=='probe' and audit['local_group_id']==plan['group'] and audit['plan_sha256']==sha(plan_path(plan['group'])) and
            audit['shooting_proposal']==str(source_proposal) and audit['shooting_proposal_sha256']==sha(source_proposal) and
            np.array_equal(audit['shooting_increment_12d'],step) and
            np.array_equal(audit['applied_initial_12d'],np.asarray(plan['anchor_applied_12d'])+step),'Probe linkage/step differs')
    if full_check: check_initial_delta(anchor/'orbit_000.hdf5',trial/'orbit_000.hdf5',step,anchor_audit['units'])
    return dict(run=str(trial),gate_sha256=sha(gatefile),ic_sha256=g['reference_ic_sha256'],endpoint=g['endpoint_snapshot'],
                csv_sha256=g['residual_csv_sha256'],axis=p['coordinate'],signed_step=p['signed_step'],unit=p['unit'],
                normalized_residual_12d=decision['normalized_residual_12d'])

def response(anchor,columns,axes):
    z=np.asarray(anchor,dtype=float)
    measured=np.column_stack([v['normalized_residual_12d'] for v in columns])
    h=np.asarray([p['signed_step'] for p in axes])
    require(z.shape==(12,) and measured.shape==(12,3) and np.isfinite(measured).all(),'Need three finite new live response columns')
    jac=(measured-z[:,None])/h[None,:]
    require(np.isfinite(jac).all(),'Finite real differences required in all three columns')
    return jac

def generate(group,choice):
    plan,audit,g=load_plan(group)
    axes=plan['probes']; anchor=Path(plan['anchor'])
    if choice=='joint':
        columns=[completed_probe(plan,p,audit,True) for p in axes]
        J=response(g['normalized_residual_12d'],columns,axes)
        module_spec=importlib.util.spec_from_file_location('pure_local_minimax',MATH)
        math=importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(math)
        scales=np.array([abs(p['signed_step']) for p in axes])
        z=np.asarray(g['normalized_residual_12d'])
        # Even perfect independent sign choices cannot move a row more than this much within the trust box.
        row_budget=np.sum(np.abs(J*scales),axis=1)
        try:
            fit=math.propose_step(z,J,scales,target_max=2.8)
        except RuntimeError as error:
            scaled=J*scales
            singular=np.linalg.svd(scaled,compute_uv=False)
            rank=int(np.linalg.matrix_rank(scaled));condition=float(singular[0]/singular[-1]) if rank==3 else None
            fit=dict(ready_for_live_validation=False,status='diagnostic_only',reasons=['optimizer_failure'],error=str(error),
                     rank=rank,condition_number=condition,singular_values=singular.tolist(),delta=[0.]*3)
        fit['trust_box_row_change_upper_bounds_sigma']=row_budget.tolist()
        fit['coordinates_cannot_reach_2p8_even_alone']=[ORDER[i] for i in np.flatnonzero(abs(z)-row_budget>=2.8)]
        delta=np.zeros(12)
        for p,d in zip(axes,fit['delta'],strict=True): delta[p['index']]=d
        ready=fit['ready_for_live_validation'] and fit['rank']==3 and fit['condition_number']<=1000 and fit['projected_max_abs']<2.8
        status='ready_for_live_validation' if ready else 'diagnostic_only'
        diag=dict(anchor_z=g['normalized_residual_12d'],response_per_physical_unit=J.tolist(),axes=axes,probes=columns,fit=fit,alpha=None)
        target=Path(plan['joint_target']);role='joint'
        coordinate=unit=h=None
    else:
        k=next((i for i,p in enumerate(axes) if p['coordinate']==choice),None)
        require(k is not None,'Coordinate not in frozen group')
        for earlier in axes[:k]: completed_probe(plan,earlier,audit,False)
        p=axes[k];delta=one_axis(p);role='probe';status='ready_for_probe'
        target=Path(p['target']);coordinate=choice;unit=p['unit'];h=p['signed_step']
        diag=dict(independent_difference=True,coordinate=coordinate,physical_step=h,unit=unit,alpha=None)
    require(not target.exists(),'Target trial already exists')
    base=np.asarray(plan['anchor_applied_12d'])
    return plain(dict(version=VERSION,status=status,local_role=role,family_key=coarse.FAMILY_KEY,local_group_id=group,
        code_sha256=plan['code_sha256'],plan_sha256=sha(plan_path(group)),anchor_gate_sha256=plan['anchor_gate_sha256'],
        anchor_run=str(anchor),base_run=str(anchor),anchor_ic_sha256=g['reference_ic_sha256'],
        base_ic=str(anchor/'orbit_000.hdf5'),base_ic_sha256=g['reference_ic_sha256'],base_initial_12d=base,
        step_12d=delta,proposed_initial_12d=base+delta,target_run=str(target),target_time_code=coarse.gate.TARGET,
        residual_csv=g['residual_csv'],residual_csv_sha256=g['residual_csv_sha256'],diagnostics=diag,
        measurement_signature=g['measurement_signature'],binary_sha256=coarse.read(coarse.FAMILY_FILE)['gizmo_sha256'],
        alpha=None,alpha_applied=False,probe_coordinate=coordinate,probe_step_value=h,probe_step_unit=unit,
        live_convergence_verified=False))

def verify_proposal(q,path,target):
    require(q['version']==VERSION and q['family_key']==coarse.FAMILY_KEY,'Foreign proposal')
    require(q['status'] in {'ready_for_probe','ready_for_live_validation'} and q['alpha'] is None and q['alpha_applied'] is False,
            'Proposal is diagnostic-only or was rescaled')
    target=Path(target).resolve()
    require(target.parent==ROOT and q['target_run']==str(target) and path==ROOT/(target.name+'_proposal.json'),'Wrong target trial/path')
    choice=q['probe_coordinate'] if q['local_role']=='probe' else 'joint'
    expected=generate(q['local_group_id'],choice)
    require(expected==q,'Real evidence, group, response, candidate or implementation changed')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest='command',required=True)
    plan=subs.add_parser('plan');plan.add_argument('--group',required=True);plan.add_argument('--anchor',type=Path,required=True)
    plan.add_argument('--axes',nargs=3,required=True,metavar='COORD=SIGNED_STEP');plan.add_argument('--reason',required=True)
    gen=subs.add_parser('probe');gen.add_argument('--group',required=True);gen.add_argument('--coordinate',required=True)
    joint=subs.add_parser('joint');joint.add_argument('--group',required=True)
    ap=subs.add_parser('apply');ap.add_argument('--proposal',type=Path,required=True);ap.add_argument('--trial-dir',type=Path,required=True)
    a=p.parse_args()
    stop_if_done(); coarse.visible_only()
    with coarse.exclusive_compute():
        if a.command=='plan':
            exclusive_json(plan_path(a.group),freeze_plan(a.group,a.anchor,a.axes,a.reason))
        elif a.command in ('probe','joint'):
            q=generate(a.group,a.coordinate if a.command=='probe' else 'joint')
            if q['status']=='diagnostic_only':
                out=ROOT/'provenance'/(a.group+'_diagnostic_only.json')
            else:
                out=ROOT/(Path(q['target_run']).name+'_proposal.json')
            stop_if_done();exclusive_json(out,q)
            print(out,q['status'])
        else:
            from local_rigid_writer import apply
            path=a.proposal.resolve(strict=True)
            apply(path,a.trial_dir.resolve(),ROOT/'campaign.json',a.trial_dir.resolve().name)

if __name__=='__main__': main()
