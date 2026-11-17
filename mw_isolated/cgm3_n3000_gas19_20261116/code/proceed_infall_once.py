
from pathlib import Path
import os,json,hashlib,copy,subprocess,datetime,fcntl,re,shutil,sys
import numpy as np,h5py
P=Path('/home/xingdi/gizmo-runs');ROOT=P/'mw_isolated/cgm3_n3000_gas19_20261116'
OLD=P/'mw_isolated/cgm3_n3000_orbits_cpp_20261105_084436';S=ROOT/'scripts';PY=P/'.venv/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def update(stage,extra=None):
 d=json.loads((ROOT/'campaign.json').read_text());d.update(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());d.update(extra or {});save(ROOT/'campaign.json',d)
def params(text,changes):
 lines=[];seen=set()
 for l in text.splitlines():
  a=re.split('[%#]',l,1)[0].split()
  if a and a[0] in changes:lines.append(f'{a[0]:<40} {changes[a[0]]}');seen.add(a[0])
  else:lines.append(l)
 assert seen==set(changes),('Missing params',set(changes)-seen)
 return '\n'.join(lines)+'\n'
def main():
 assert os.isatty(0) and os.isatty(1) and os.environ.get('TMUX_PANE')
 assert subprocess.check_output(['tmux','-L','gizmo-orbits','list-clients'],text=True).strip()
 assert json.loads((ROOT/'campaign.json').read_text())['smoke_required'] is False
 with (ROOT/'status/proceed_infall_once.json').open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'scope':'New-family direct formal infall; no smoke, no jet'},f)
 with (OLD/'status/compute.flock').open('r+') as globallease:
  fcntl.flock(globallease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  sys.path.insert(0,str(S));import stage_safety as safety
  assert not safety.compute_processes()
  cfg=json.loads((ROOT/'campaign.json').read_text());host=Path(cfg['host_endpoint']);assert sha(host)==cfg['host_sha256']
  sats=[];manifests=[]
  for name in ['lmc','smc']:
   stat=json.loads((ROOT/'relaxation'/name/'production/status.json').read_text());assert stat['state']=='complete' and stat['exit_code']==0 and abs(stat['final_time']-.6136)<1e-12
   last=Path(stat['final_snapshot']);assert sha(last)==stat['final_snapshot_sha256'];sats.append(last)
   manifests.append(json.loads((ROOT/'relaxation'/name/'relax_manifest.json').read_text()))
  ref=Path(cfg['previous_accepted_ic_archive']);assert sha(ref)=='6702f70859a4878669fd51216547715cb1457cd07edf974a59d9e10ea1067be8'
  oldaudit=json.loads(ref.with_suffix('.orbit.json').read_text())
  rotations=oldaudit['rotation_matrices'];key=copy.deepcopy(oldaudit['model_key_inputs']);key['production_parameters']['CpuTimeBetRestartFile']=900.0
  newpaths=[host,*sats];newhashes=[sha(p) for p in newpaths]
  key.update(source_sha256=newhashes,gas_mass_normalization_Msun=cfg['gas_mass_initial_Msun'],gas_revision_role='Literature lower mass normalization only; same collisionless TG19 profiles')
  packet={'source_files':[str(x) for x in newpaths],'source_sha256':newhashes,'source_times_code':[.6136272158015191,.6136,.6136],
   'source_models':[None,*[x['model'] for x in manifests]],'model_key_inputs':key,'reference_source_sha256':oldaudit['source_sha256']}
  save(ROOT/'provenance/new_family_sources.json',packet)
  save(ROOT/'provenance/phase_space_reference.json',{'initial_ic':str(ref),'initial_ic_sha256':sha(ref),'run':str(OLD/'local02_lmc_y_p05'),'role':'Old accepted actual initial12D and rotations only; new gas family needs new measured endpoint; no old Jacobian'})
  trial=ROOT/'calibration01';trial.mkdir(exist_ok=False)
  update('merging_new_relaxed_satellites',{'new_relaxation_snapshots':[str(x) for x in sats]})
  subprocess.run([str(PY),'-B','-u',str(S/'build_calibration_ic.py')],cwd=ROOT,check=True)
  audit=json.loads((trial/'orbit_000.orbit.json').read_text());ic=trial/'orbit_000.hdf5'
  assert audit['source_sha256']==newhashes and audit['output_ic_sha256']==sha(ic)
  assert np.array_equal(np.array(audit['applied_initial_12d']),np.array(oldaudit['applied_initial_12d']))
  assert np.array_equal(np.array(audit['rotation_matrices']),np.array(rotations))
  assert audit['family_key']!=oldaudit['model_key']
  with h5py.File(ic,'r') as f:
   counts=f['Header'].attrs['NumPart_Total'].tolist()
   assert counts==audit['counts'] and abs(float(f['Header'].attrs['Time'])-.6136272158015191)<1e-12
   for pt in [0,1,4,5]:
    g=f[f'PartType{pt}']
    for name in g:assert np.isfinite(g[name][:]).all(),(pt,name)
   expected={'LMC':(120000000,129999999,5000),'SMC':(210000000,219999999,1600)}
   ids=f['PartType4/ParticleIDs'][:]
   for name,(lo,hi,n) in expected.items():assert int(np.sum((ids>=lo)&(ids<=hi)))==n
  save(trial/'initial_review.json',{'status':'pass','source_snapshots':packet['source_files'],'source_sha256':newhashes,'same_actual_initial12D_and_rotations':True,'counts':counts,'finite_fields':True,'original_star_counts':{'LMC':5000,'SMC':1600},'new_family_key':audit['family_key'],'ic_sha256':sha(ic),'no_smoke':'Explicit user instruction'})
  update('new_ic_ready_for_direct_formal_runtime',{'family_key':audit['family_key'],'family_inputs_sha256':sha(ROOT/'provenance/cgm3_family_inputs.json'),'initial_ic_sha256':sha(ic),'active_ic_sha256':sha(ic)})
  prod=trial/'production';prod.mkdir();(prod/'output').mkdir();(prod/'spcool_tables').mkdir()
  template=OLD/'local02_lmc_y_p05';base=ROOT/'provenance/infall_base.param';text=base.read_text()
  text=params(text,{'InitCondFile':'../orbit_000','OutputDir':'./output/','TimeBegin':'.61362721580151913','TimeMax':'1.6363392421373844','TimeOfFirstSnapshot':'.61362721580151913','MaxSizeTimestep':'.001','MinSizeTimestep':'1e-12','MaxMemSize':'1000','CpuTimeBetRestartFile':'900','CritPhysDensity':'3000','MaxKernelRadius':'1e56'})
  (prod/'stage2.param').write_text(text);deps=[]
  for src in [template/'production/TREECOOL',*sorted((template/'production/spcool_tables').glob('spcool_*'))]:
   if not src.is_file() or (src.name!='TREECOOL' and not re.fullmatch(r'spcool_\d+',src.name)):continue
   dst=prod/src.relative_to(template/'production');shutil.copy2(src,dst);assert sha(dst)==sha(src)
   deps.append({'mode':'production','relative_path':str(dst.relative_to(trial)),'source':str(src),'sha256':sha(src)})
  config=json.loads((template/'run_configuration.json').read_text())
  for keyname in ['smoke_only','smoke_timeout_seconds']:config.pop(keyname,None)
  config.update(trial_dir=str(trial),schema='gas19-direct-formal-runtime-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
   parameter_source=str(base),parameter_source_sha256=sha(base),ic_sha256=sha(ic),sidecar_sha256=sha(trial/'orbit_000.orbit.json'),
   param_sha256={'production':sha(prod/'stage2.param')},runtime_dependencies=deps,smoke_required=False,
   readiness_required=['current IC, sidecar, production params, pinned binary, explicit user no-smoke policy'],
   physical_inheritance='Same n3000 host and preAGN parameters; new independently relaxed lower-gas satellites; no smoke simulation')
  save(trial/'run_configuration.json',config)
  policy=ROOT/'provenance/user_no_smoke_policy.json'
  ready={'status':'pass','ic_sha256':config['ic_sha256'],'sidecar_sha256':config['sidecar_sha256'],'production_param_sha256':config['param_sha256']['production'],'binary_sha256':config['binary']['sha256'],'smoke_required':False,'user_no_smoke_policy_sha256':sha(policy),'release_basis':'User explicitly requires direct formal simulation; input/run binding verified; no smoke PASS claimed'}
  save(trial/'production_ready.json',ready)
  save(trial/'production_launch_clearance.json',{'status':'pass','ready_sha':sha(trial/'production_ready.json'),'ic_sha256':sha(ic),'user_no_smoke_policy_sha256':sha(policy),'smoke_required':False})
  cmd=f"{PY} -B -u {S/'drive_stage.py'} --trial-dir {trial} --mode production"
  shell="#!/usr/bin/env bash\nset -euo pipefail\ncd "+str(trial)+"\nexport GIZMO_VISIBLE_TTY=$(tty)\nexec script -q -e -f -c '"+cmd+"' "+str(trial/'production_terminal_session.log')+"\n"
  (trial/'start_production.sh').write_text(shell);(trial/'start_production.sh').chmod(0o755)
  pins=json.loads((OLD/'provenance/launch_hashes.json').read_text())['sha256']
  for path,expectedhash in pins.items():
   if Path(path).is_file():assert sha(Path(path))==expectedhash,'Historical core dependency changed '+path
  pins={p:h for p,h in pins.items() if Path(p).is_file()}
  for p in [*S.glob('*.py'),ROOT/'code/proceed_infall_once.py',policy,ROOT/'provenance/cgm3_family_inputs.json',ROOT/'provenance/new_family_sources.json']:
   pins[str(p)]=sha(p)
  save(ROOT/'provenance/launch_hashes.json',{'scope':'gas19 new-family direct formal infall','sha256':pins})
  # Input/parameter guard only; no smoke simulation or synthetic smoke result.
  safety.verify_before_run(trial,'production')
  update('direct_formal_infall_starting',{'active_run':str(trial),'trial':'calibration01','status_path':str(prod/'status.json')})
  rc=subprocess.run([str(PY),'-B','-u',str(S/'production_and_gate.py'),'--trial-dir',str(trial)],cwd=ROOT).returncode
  assert rc in [0,3],f'Formal infall/validator returned {rc}'
if __name__=='__main__':
 try:main()
 except BaseException as e:update('new_family_infall_requires_review',{'error':repr(e)});raise
