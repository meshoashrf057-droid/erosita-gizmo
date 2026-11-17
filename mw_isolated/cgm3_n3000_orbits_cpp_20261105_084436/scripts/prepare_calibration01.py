from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys
N=Path(__file__).resolve().parents[1]
P='/home/xingdi/gizmo-runs/.venv/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def update(stage,extra=None):
 p=N/'campaign.json';c=json.loads(p.read_text());c.update(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());c.update(extra or {})
 t=p.with_suffix('.json.partial');t.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def main():
 assert os.isatty(0) and os.isatty(1)
 pane=os.environ['TMUX_PANE']
 session=subprocess.run(['tmux','-L','gizmo-orbits','display-message','-p','-t',pane,'#{session_name}'],capture_output=True,text=True,check=True).stdout.strip()
 clients=subprocess.run(['tmux','-L','gizmo-orbits','list-clients','-F','#{session_name}'],capture_output=True,text=True,check=True).stdout.splitlines()
 assert session=='cgm3_n3000_orbit' and session in clients
 assert not list((N/'calibration01').iterdir()),'Prior preparation artifacts exist'
 with (N/'status/prepare_once.lock').open('x') as f:json.dump({'pid':os.getpid(),'tty':os.ttyname(0),'session':session},f)
 steps=[('build_calibration_ic.py',[]),('verify_calibration_ic.py',[]),('make_trial_runtime.py',['--trial-dir',str(N/'calibration01')])]
 for script,args in steps:
  update('preparing_'+script[:-3])
  print('RUNNING',script,flush=True)
  subprocess.run([P,'-B','-u',str(N/'scripts'/script),*args],cwd=N,check=True)
  if script=='build_calibration_ic.py':
   manifest=json.loads((N/'provenance/adapter/adaptation_manifest.json').read_text())
   family_path=N/'provenance/cgm3_family_inputs.json';family=json.loads(family_path.read_text());a=json.loads((N/'calibration01/orbit_000.orbit.json').read_text())
   assert family==manifest['expected_family_inputs']
   assert sha(family_path)==manifest['expected_family_file_sha256']
   assert a['model_key']==a['family_key']==manifest['family_key']
   update('initial_merged_pending_static_check',{'family_key':a['family_key'],'family_inputs':family,'initial_ic_sha256':a['output_ic_sha256'],'baseline_ic':str(N/'calibration01/orbit_000.hdf5'),'baseline_ic_sha256':a['output_ic_sha256'],'smoke_status_path':str(N/'calibration01/smoke/status.json'),'production_status_path':str(N/'calibration01/production/status.json'),'status_path':str(N/'status/preparation_result.json')})
 result={'state':'prepared_static_verified','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ready_for_smoke_only':True,'production_authorized_after_real_smoke_and_launch_clearance':True}
 with (N/'status/preparation_result.json').open('x') as f:json.dump(result,f,indent=2)
 update('static_passed_runtime_prepared')
 print('PREPARATION_COMPLETE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  update('preparation_requires_review',{'error':str(e)})
  raise
