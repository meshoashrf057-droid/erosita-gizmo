from pathlib import Path
import os,sys,json,hashlib,datetime,shutil,fcntl,subprocess,re,shlex
sys.dont_write_bytecode=True
P=Path('/home/xingdi/gizmo-runs');R=P/'mw_isolated/cgm3_n3000_gas19_20261116';T=R/'calibration01';OLD=T/'production_resume01';NEW=T/'production_resume02';A=R/'recovery_20261117_02'
sys.path.insert(0,str(R/'scripts'))
import run_trial as rt
from stage_safety import compute_processes
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def load(p):return json.loads(p.read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):rt.write_status(p,d)
def exclusive(p,d):
 with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def stage(name,**extra):
 c=load(R/'campaign.json');c.update(stage=name,updated_utc=utc(),**extra);save(R/'campaign.json',c)
def cp(src,dst):
 assert src.is_file() and not src.is_symlink()
 before=sha(src);dst.parent.mkdir(parents=True,exist_ok=True)
 with src.open('rb') as f,dst.open('xb') as g:shutil.copyfileobj(f,g,8*1024*1024);g.flush();os.fsync(g.fileno())
 assert sha(src)==sha(dst)==before
 return {'source':str(src),'path':str(dst),'bytes':dst.stat().st_size,'sha256':before}
def tail_text(path):
 with path.open('rb') as f:f.seek(max(0,path.stat().st_size-16000));return f.read().decode(errors='replace')
def main():
 assert os.isatty(0) and os.isatty(1) and os.environ.get('TMUX_PANE')
 assert subprocess.check_output(['tmux','-L','gizmo-orbits','list-clients'],text=True).strip()
 assert not compute_processes(),'Existing compute job'
 with (P/'mw_isolated/cgm3_n3000_orbits_cpp_20261105_084436/status/compute.flock').open('r+') as lease:
  fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  exclusive(A/'resume_claim.json',{'utc':utc(),'pid':os.getpid(),'instruction':'用户授权GitHub上传完成后续跑至终点、检查12D再alpha；main commit9190fc626ebd69791796c44da5ab4e38d30576b5已上传核验','smoke_required':False})
  assert shutil.disk_usage(R).free>12*1024**3
  subprocess.run([str(P/'.venv/bin/python'),'-B','-u',str(A/'checkpoint_audit.py')],check=True)
  cfg=load(T/'run_configuration.json');oldstatus=load(OLD/'status.json');audit=load(A/'checkpoint_audit.json')
  assert audit['status']=='structural_and_consistency_checks_passed' and len(audit['rows'])==6
  checkpoint=audit['rows'][0]['All']['Time']
  assert checkpoint==1.6273583648919576 and audit['rows'][0]['All']['SnapshotFileCount']==11
  assert not NEW.exists() and checkpoint<cfg['production_target_code']
  assert load(R/'provenance/user_no_smoke_policy.json')['smoke_required'] is False
  assert sha(cfg['binary']['path'])==cfg['binary']['sha256']==audit['binary_sha256']
  assert sha('/home/xingdi/gizmo-public-gizmo-cpp/Config.sh')==audit['config_sha256']
  for p,h in load(R/'provenance/launch_hashes.json')['sha256'].items():assert sha(p)==h,('changed dependency',p)
  assert sha(T/'orbit_000.hdf5')==cfg['ic_sha256'] and sha(T/'orbit_000.orbit.json')==cfg['sidecar_sha256']
  assert sha(OLD/'stage2.param')==cfg['param_sha256']['production']
  assert not any((OLD/'output'/x).exists() for x in ['stop','cont'])
  exclusive(A/'interruption_evidence.json',{'utc':utc(),'old_status':oldstatus,'actual_compute_processes':[],'log_last_sync':re.findall(r'Sync-Point\s+(\d+), Time:\s*([^,\s]+)',tail_text(OLD/'gizmo_run.log'))[-1],'checkpoint_time':checkpoint,'uptime':Path('/proc/uptime').read_text(),'cause':'User reports intentional interruption; prior status recorded ProcessInterrupted(2); resume requested after GitHub upload','original_status_preserved':True})
  NEW.mkdir();(NEW/'output/restartfiles').mkdir(parents=True)
  copied=[cp(OLD/'stage2.param',NEW/'stage2.param')]
  for row in audit['rows']:
   src=Path(row['path']);assert sha(src)==row['sha256'] and row['exact_EOF']
   copied.append(cp(src,NEW/'output/restartfiles'/src.name))
  for src in sorted((OLD/'output').rglob('*')):
   if not src.is_file() or 'restartfiles' in src.parts:continue
   dst=NEW/'output'/src.relative_to(OLD/'output')
   if re.fullmatch(r'snapshot_[0-9]+\.hdf5',src.name):
    assert src.parent==OLD/'output' and int(src.stem.split('_')[-1])<audit['rows'][0]['All']['SnapshotFileCount']
    before=sha(src);dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
    assert sha(dst)==before and src.stat().st_ino==dst.stat().st_ino
    copied.append({'source':str(src),'path':str(dst),'bytes':dst.stat().st_size,'sha256':before,'storage':'hardlink_read_only_previous_snapshot','native_next_snapshot_index':11})
   else:copied.append(cp(src,dst))
  for d in cfg['runtime_dependencies']:
   src=OLD/Path(d['relative_path']).relative_to('production');assert sha(src)==d['sha256']
   copied.append(cp(src,NEW/src.relative_to(OLD)))
  exclusive(A/'prepared_manifest.json',{'utc':utc(),'source':str(OLD),'destination':str(NEW),'checkpoint_time':checkpoint,'checkpoint_audit_sha256':sha(A/'checkpoint_audit.json'),'copied_files':copied,'binary':cfg['binary'],'config_sha256':audit['config_sha256'],'param_sha256':sha(NEW/'stage2.param'),'changed_physical_parameters':[],'restart_flag':1,'smoke_required':False,'target_unchanged':cfg['production_target_code'],'endpoint_adapter_sha256':sha(R/'scripts/gate_endpoint_resume02.py')})
  native=['/usr/bin/mpirun','-np','6','/usr/bin/nice','-n','19',cfg['binary']['path'],'stage2.param','1']
  state=dict(schema='gas19-native-checkpoint-resume-v1',state='starting',started_utc=utc(),wrapper_pid=os.getpid(),mpi_pid=None,command=native,cwd=str(NEW),visible_terminal=os.ttyname(0),resume_time=checkpoint,original_directory=str(OLD),restart_flag=1,mpi_ranks=6,omp_threads=1,nice=19,ic_sha256=cfg['ic_sha256'],sidecar_sha256=cfg['sidecar_sha256'],param_sha256=sha(NEW/'stage2.param'),binary_sha256=cfg['binary']['sha256'],simulation_returncode=None,wrapper_returncode=None,returncode=None,target_time=cfg['production_target_code'],smoke_required=False)
  exclusive(NEW/'send_once.json',dict(state));save(NEW/'status.json',state)
  stage('checkpoint_resume_starting',active_run=str(T),active_production=str(NEW),active_status=str(NEW/'status.json'),status_path=str(NEW/'status.json'),recovery_checkpoint_time=checkpoint)
  env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LD_LIBRARY_PATH=cfg['dynamic_library_path'],MPLCONFIGDIR=str(T/'mpl'),TMPDIR=str(T/'tmp'))
  def started(pid):
   state.update(state='running',mpi_pid=pid,process_group=pid);save(NEW/'status.json',state)
   stage('checkpoint_resume_running',active_mpi_pid=pid)
  try:
   rc=rt.stream_process(native,NEW,env,NEW/'gizmo_run.log',None,started)
   state.update(simulation_returncode=rc,returncode=rc,simulation_ended_utc=utc())
   if rc!=0:
    state.update(state='simulation_failed',wrapper_returncode=rc,ended_utc=utc());save(NEW/'status.json',state);stage('checkpoint_resume_requires_review',error='Native exit '+str(rc));return rc
   final,time=rt.last_snapshot(NEW/'output')
   state.update(final_snapshot=str(final),final_snapshot_time=time)
   assert abs(time-cfg['production_target_code'])<1e-10,'Stopped before true target'
   state.update(state='simulation_completed',wrapper_returncode=0);save(NEW/'status.json',state)
   stage('endpoint_gate_running')
   gate=[str(P/'.venv/bin/python'),'-B','-u',str(R/'scripts/gate_endpoint_resume02.py'),'--run-dir',str(T)]
   gate_log=A/'endpoint_gate_terminal.log'
   assert not gate_log.exists()
   grc=subprocess.run(['script','-q','-f','-e','-c',shlex.join(gate),str(gate_log)],cwd=T,env=env).returncode
   for entry in copied:
    if entry.get('storage')=='hardlink_read_only_previous_snapshot':assert sha(entry['source'])==sha(entry['path'])==entry['sha256']
   assert grc in [0,3],'Endpoint evidence failure '+str(grc)
   result=load(T/'endpoint_review/gate_result.json')
   state.update(state='completed',wrapper_returncode=0,orbit_validation_returncode=0,validation_command=result['validation_command'],ended_utc=utc());save(NEW/'status.json',state)
   exclusive(T/'orbit_outcome.json',result)
   extra={'latest_gate':str(T/'endpoint_review/gate_result.json'),'active_run':None,'best_actual_run':str(T),'best_D12':result['D12'],'best_max_abs_sigma':result['max_abs_sigma']}
   if grc==0:
    exclusive(R/'accepted_orbit.json',result);extra.update(accepted_trial=str(T),next_stage='Stop for user review; no automatic jet')
   else:extra.update(next_stage='Review new-family measured residuals; no old Jacobian reuse or automatic jet')
   stage('orbit_passed_awaiting_user_review' if grc==0 else 'measured_orbit_needs_correction',**extra)
   return grc
  except BaseException as e:
   state.update(error=repr(e),state='resume_requires_review',ended_utc=utc());save(NEW/'status.json',state);stage('checkpoint_resume_requires_review',error=repr(e));raise
if __name__=='__main__':sys.exit(main())
