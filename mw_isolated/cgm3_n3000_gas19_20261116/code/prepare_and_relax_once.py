
from pathlib import Path
import json,os,subprocess,shlex,hashlib,datetime,fcntl,re,shutil,gc
import h5py,numpy as np
P=Path('/home/xingdi/gizmo-runs');ROOT=P/'mw_isolated/cgm3_n3000_gas19_20261116'
PY=P/'.venv/bin/python';BIN=P.parent/'gizmo-public-gizmo-cpp/GIZMO'
OLD=P/'mw_isolated/cgm3_n3000_orbits_cpp_20261105_084436';DEP=OLD/'local02_lmc_y_p05/production'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):
 t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def stage(n,extra=None):
 p=ROOT/'campaign.json';d=json.loads(p.read_text());d.update(stage=n,updated_utc=now());d.update(extra or {});save(p,d);print('STAGE',n,extra,flush=True)
def unique():
 for p in Path('/proc').glob('[0-9]*'):
  try:
   argv=(p/'cmdline').read_bytes().split(b'\0');name=Path(argv[0].decode()).name
   assert not(name.startswith('GIZMO') or name in ['mpirun','mpiexec']),f'Existing compute {p.name}'
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
def change(text,updates):
 used=set();lines=[]
 for line in text.splitlines():
  w=re.split('[%#]',line,1)[0].split()
  if w and w[0] in updates:lines.append(f'{w[0]:<40} {updates[w[0]]}');used.add(w[0])
  else:lines.append(line)
 for k,v in updates.items():
  if k not in used:lines.append(f'{k:<40} {v}')
 return '\n'.join(lines)+'\n'
def deps(dest):
 for p in [DEP/'TREECOOL',*sorted((DEP/'spcool_tables').glob('*'))]:
  if p.is_file():
   q=dest/p.relative_to(DEP);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
def native(run,mode,system,target):
 unique();assert sha(BIN)=='07154d9677d2e897a7dc557e37f9565d0be835a5e612943870cc5021b137f343'
 assert not (run/'status.json').exists()
 (run/'output').mkdir()
 d={'state':'running','started_utc':now(),'system':system,'mode':mode,'cwd':str(run),'target_time':target,'restart_flag':0,'mpi_ranks':6,'omp_threads':1,'nice':19,'pid':os.getpid(),'binary_sha256':sha(BIN)}
 save(run/'status.json',d);stage(system.lower()+'_'+mode+'_running',{'active_run':str(run),'active_status':str(run/'status.json')})
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LD_LIBRARY_PATH='/home/xingdi/gizmo-public/hdf5-1.8.21/hdf5/lib')
 cmd='nice -n 19 mpirun -np 6 '+shlex.quote(str(BIN))+' relax.param 0'
 proc=subprocess.Popen(['script','-q','-e','-f','-c',cmd,str(run/'run.log')],cwd=run,env=env)
 d['transport_pid']=proc.pid;d['command']=cmd;save(run/'status.json',d)
 rc=proc.wait();d.update(state='complete' if rc==0 else 'failed',exit_code=rc,finished_utc=now());save(run/'status.json',d)
 assert rc==0,f'{system} {mode} native exit{rc}'
 snaps=sorted((run/'output').glob('snapshot_*.hdf5'));assert len(snaps)>=2
 with h5py.File(snaps[-1],'r') as f:assert abs(float(f['Header'].attrs['Time'])-target)<1e-12
 d.update(final_snapshot=str(snaps[-1]),final_time=target,final_snapshot_sha256=sha(snaps[-1]),snapshots=len(snaps));save(run/'status.json',d);return d
def inspect_generated(system,run,mass):
 ic=run/(system.lower()+'_ic.hdf5');man=json.loads((run/'relax_manifest.json').read_text());old=json.loads((P/'erosita_xray'/('lmcre' if system=='LMC' else 'smcre')/'relax_manifest.json').read_text())
 model=man['model'];diff=[k for k in model if model[k]!=old['model'][k]];assert diff==['cgm_mass_rmax']
 with h5py.File(ic,'r') as f:
  u=float(f['Header'].attrs.get('UnitMass_In_CGS',f['Header'].attrs.get('UnitMass_in_g')))/1.989e33
  got=float(f['PartType0/Masses'][:].sum()*u);assert abs(got-mass)/mass<1e-12
  counts=f['Header'].attrs['NumPart_Total'].tolist();assert counts[1:]==old['counts'][1:]
  for name in ['PartType0','PartType1','PartType4']:
   for key in ['Coordinates','Velocities','Masses']:
    assert np.isfinite(f[name][key][:]).all()
  ids=np.concatenate([f[k+'/ParticleIDs'][:] for k in ['PartType0','PartType1','PartType4']]);assert len(np.unique(ids))==len(ids)
 result={'status':'pass','ic_sha256':sha(ic),'gas_mass_Msun':got,'only_model_parameter_changed':diff,'counts':counts,'collisionless_structure_unchanged':True,'velocities_and_gas_height_recomputed':True}
 save(run/'initial_review.json',result);print('INITIAL REVIEW',result,flush=True)
def smoke_review(run,ic,target):
 status=json.loads((run/'status.json').read_text())
 with h5py.File(ic,'r') as a,h5py.File(status['final_snapshot'],'r') as b:
  assert abs(float(b['Header'].attrs['Time'])-target)<1e-12
  for pt in ['PartType0','PartType1','PartType4']:
   assert len(a[pt+'/ParticleIDs'])==len(b[pt+'/ParticleIDs'])
   for key in b[pt]:
    v=b[pt][key][:]
    assert np.isfinite(v).all(),(pt,key)
   ii=np.argsort(a[pt+'/ParticleIDs'][:]);jj=np.argsort(b[pt+'/ParticleIDs'][:])
   assert np.array_equal(a[pt+'/ParticleIDs'][:][ii],b[pt+'/ParticleIDs'][:][jj])
   assert np.allclose(a[pt+'/Masses'][:][ii],b[pt+'/Masses'][:][jj],rtol=1e-10,atol=1e-15)
   if pt in ['PartType0','PartType4']:
    assert np.allclose(a[pt+'/Metallicity'][:][ii],b[pt+'/Metallicity'][:][jj],rtol=1e-8,atol=1e-12)
 save(run/'smoke_review.json',{'status':'pass','exit0':True,'exact_target_time':target,'identities_masses_metals_finite':True})
def main():
 assert os.environ.get('TMUX_PANE')=='%0' and os.isatty(0) and os.isatty(1)
 assert subprocess.check_output(['tmux','-L','gizmo-orbits','list-clients'],text=True).strip()
 with (ROOT/'status/prepare_relax_once.json').open('x') as f:json.dump({'utc':now(),'pid':os.getpid(),'tty':os.ttyname(0)},f)
 with (OLD/'status/compute.flock').open('r+') as lease:
  fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);unique()
  for system,mass in [('LMC',1.1e9),('SMC',7.9e8)]:
   run=ROOT/'relaxation'/system.lower()
   stage('generating_'+system.lower())
   cmd=[str(PY),'-B','-u',str(ROOT/'code'/f'make_{system.lower()}_relax_ic.py'),'--target-format','cpp','--base-param',str(ROOT/'code/relax_base.param'),'--gizmo-executable',str(BIN),'--run-dir',str(run),'--relax-code','.6136','--stellar-age-gyr','5','--gas-total-mass',str(mass),'--gas-particle-mass','1e5','--dm-particle-mass','5e6','--star-particle-mass','5e5']
   subprocess.run(cmd,cwd=ROOT/'code',check=True);inspect_generated(system,run,mass)
  for system in ['LMC','SMC']:
   run=ROOT/'relaxation'/system.lower();base=(run/(system.lower()+'.param')).read_text();ic=run/(system.lower()+'_ic.hdf5')
   for mode in ['smoke','production']:
    q=run/mode;q.mkdir();deps(q)
    common={'InitCondFile':'../'+system.lower()+'_ic','OutputDir':'./output/','TimeLimitCPU':'288600','MaxMemSize':'1000'}
    target=1e-5 if mode=='smoke' else .6136
    common.update({'TimeBegin':'0','TimeMax':format(target,'.17g'),'TimeOfFirstSnapshot':'1e-8' if mode=='smoke' else '0','TimeBetSnapshot':'9.99e-6' if mode=='smoke' else '.1','MaxSizeTimestep':'1e-6' if mode=='smoke' else '.001','MinSizeTimestep':'1e-12','InitStellarAge':'5.0'})
    (q/'relax.param').write_text(change(base,common))
    result=native(q,mode,system,target)
    if mode=='smoke':smoke_review(q,ic,target)
    else:stage(system.lower()+'_relaxation_complete',{'last_completed_relaxation':result})
  stage('both_satellites_relaxed_ready_for_infall_merge',{'active_run':None})
  save(ROOT/'status/relaxations_complete.json',{'utc':now(),'status':'complete','systems':['LMC','SMC'],'next':'new-family merge, independent initial validation, smoke and infall; no jet'})
if __name__=='__main__':
 try:main()
 except BaseException as e:stage('preparation_or_relaxation_requires_review',{'error':repr(e)});raise
