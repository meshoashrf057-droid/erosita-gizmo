#!/usr/bin/env python3
"""Single visible isolated build; never modifies either reference source tree."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,datetime,fcntl,re
P=Path('/home/xingdi/gizmo-runs')
A=P/'compat_validation/20261115_fixed_supply_cpp_port_review01'
S=A/'source';N=P/'mw_isolated/cgm3_n3000_orbits_cpp_20261105_084436'
T=N/'local02_lmc_y_p05';J=P/'erosita_xray/cgm3_n3000_cpp_fixed_supply_20261115'
B=A/'build_review01';SELF=Path(__file__).resolve()
def require(v,m):
 if not v:raise RuntimeError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
 with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False);f.write('\n')
def identity(p):return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def visible():
 require(os.isatty(0) and os.isatty(1) and os.ttyname(0)==os.ttyname(1),'Build requires attached script PTY')
 clients=subprocess.check_output(['tmux','-L','gizmo-orbits','list-clients','-F','#{client_pid} #{client_tty} #{client_session}'],text=True)
 panes=subprocess.check_output(['tmux','-L','gizmo-orbits','list-panes','-t','cgm3_n3000_orbit','-F','#{pane_id} #{pane_pid} #{pane_tty}'],text=True)
 require('882 /dev/pts/0 cgm3_n3000_orbit' in clients and '%0 847 /dev/pts/2' in panes,'Existing visible client/pane changed')
 pid=os.getpid();chain=[]
 for _ in range(20):
  chain.append(pid)
  if pid==1:break
  status=Path(f'/proc/{pid}/status').read_text();pid=int(re.search(r'^PPid:\s*(\d+)',status,re.M)[1])
 require(847 in chain and 846 in chain,'Build is not in existing visible pane ancestry')
 for d in Path('/proc').iterdir():
  if not d.name.isdigit():continue
  try:args=(d/'cmdline').read_bytes().decode().split('\0')
  except (FileNotFoundError,PermissionError):continue
  require(not any(Path(a).name.startswith('GIZMO') or Path(a).name=='render_accepted_all_v10log.py' for a in args if a),'Another simulation/render workload exists')
 return dict(pid=os.getpid(),tty=os.ttyname(0),chain=chain,clients=clients,panes=panes)
def fence():
 port=read(A/'source_port_manifest.json');inp=read(A/'source_input_manifest.json')
 require(sha(A/'source_input_manifest.json')==port['source_input_manifest_sha256'],'Original source input manifest changed')
 require(sha(A/'source_port_manifest.json')=='09621c0f0d1220c08e3fb78d9c5e3abe18890f529d15de2a8b67b82367f545c4','Port manifest changed')
 require(sha(A/'independent_source_review.json')=='914bcb86dd7232b0dc07ca308817e8bfe87a2726e04029d784a46122864ff202','Independent source review changed')
 require(sha(A/'static_checks.json')=='5814e4a30193675f74239e9f39fb6f7ca6b353edbe51b6502df5cc0f6b8eb207','Static checks changed')
 for root,manifest in [(S,port),(Path(inp['source_root']),inp)]:
  for r in manifest['files']:
   p=root/r['path'];require(p.is_file() and not p.is_symlink() and sha(p)==r['sha256'],'Source changed: '+str(p))
 for name,digest in inp['protected_reference_sha256'].items():require(sha(Path(name))==digest,'Protected reference changed: '+name)
 return dict(port=identity(A/'source_port_manifest.json'),independent_review=identity(A/'independent_source_review.json'))
def main():
 require(S.resolve()==S and J.resolve()==J,'Redirected isolated directory')
 v=visible()
 with (N/'status/compute.flock').open('r+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  require(not B.exists(),'Build attempt exists; do not repeat')
  summaries=[p for p in (T/'visualizations/manifests').glob('render_run_*.json') if read(p).get('all_requested_plots_and_videos_complete') is True]
  require(len(summaries)==1,'All requested plots/videos not yet complete')
  summary=summaries[0];d=read(summary)
  require(d['verified_png_count']==66 and d['verified_video_count']==6,'Incomplete visualization counts')
  for row in d['manifests']+d['video_manifests']:
   require(sha(Path(row['path']))==row['sha256'],'Visualization manifest changed')
  axis=J/'preparation/azaxis_report.json';ic=J/'preparation/ic_initialization.json'
  require(axis.is_file() and ic.is_file(),'Owned final-axis and IC preparation must precede build')
  require(read(axis)['specific_angmom_vector']==[0.,0.,1.] and read(axis)['target_format']=='cpp','Unexpected own-axis setup')
  baseline=fence();require(not (S/'GIZMO_fixed_supply_cpp').exists(),'Untracked preexisting binary')
  B.mkdir();command=['/usr/bin/nice','-n','19','/usr/bin/make','-j2','CONFIG=Config.sh','EXEC=GIZMO_fixed_supply_cpp']
  start=dict(started_utc=utc(),visible=v,command=command,cwd=str(S),source_fence=baseline,axis=identity(axis),
   ic_initialization=identity(ic),visualization_complete=identity(summary),self=identity(SELF),
   note='This is isolated build only. No original source, Config, binary, accepted snapshot or scientific contract is edited.')
  write(B/'started.json',start);print('BUILD_STARTED',json.dumps(start),flush=True)
  try:
   env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
   rc=subprocess.run(command,cwd=S,env=env).returncode
   require(rc==0,'Compiler/linker failed with exit '+str(rc))
   header=(S/'GIZMO_config.h').read_text()
   for macro in ['SINK_PARTICLES','SINK_REPOSITION_ON_POTMIN','SINK_WIND_SPAWN','SINK_FOLLOW_ACCRETED_ANGMOM',
    'SINK_FOLLOW_ACCRETED_MOMENTUM','SINK_FOLLOW_ACCRETED_COM','SINK_FB_COLLIMATED','SINK_WAKEUP_GAS',
    'SINK_INTERACT_ON_GAS_TIMESTEP','SINK_OUTPUT_MOREINFO','OUTPUT_UNSPAWNED_SINKMASS']:
    require(re.search(r'^#define\s+'+macro+r'\b',header,re.M),'Missing generated macro '+macro)
   require(re.search(r'^#define\s+SINK_DEBUG_FIX_MDOT_MBH\s+\(?0\.00124\)?\s*$',header,re.M),'Fixed supply differs')
   require(re.search(r'^#define\s+SINK_WIND_SPAWN\s+2\s*$',header,re.M),'Wrong spawn mode')
   binary=S/'GIZMO_fixed_supply_cpp';require(binary.is_file() and binary.stat().st_size>100000,'No linked executable')
   ldd=subprocess.run(['/usr/bin/ldd',str(binary)],capture_output=True,text=True)
   require(ldd.returncode==0 and 'not found' not in ldd.stdout,'Runtime shared dependency unresolved')
   require(fence()==baseline,'Source/reference changed during isolated compile')
   result=dict(status='build_pass_only',completed_utc=utc(),exit_code=rc,binary=identity(binary),
    config=identity(S/'Config.sh'),generated_config=identity(S/'GIZMO_config.h'),
    generated_config_text=header,ldd=ldd.stdout,source_fence=baseline,started=identity(B/'started.json'),
    runtime_validated=False,smoke_validated=False)
   write(B/'complete.json',result);print('BUILD_COMPLETE',json.dumps(result),flush=True)
  except BaseException as e:
   write(B/'failure.json',dict(failed_utc=utc(),error=repr(e),source_preservation_note='Preserve failed build products/log; never overwrite or rerun an existing launch blindly.'))
   raise
if __name__=='__main__':main()
