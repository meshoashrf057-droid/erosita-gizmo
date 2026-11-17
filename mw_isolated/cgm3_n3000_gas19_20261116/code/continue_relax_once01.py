
from pathlib import Path
import importlib.util,fcntl,os,json
import numpy as np,h5py
P=Path('/home/xingdi/gizmo-runs');ROOT=P/'mw_isolated/cgm3_n3000_gas19_20261116'
sp=importlib.util.spec_from_file_location('base_prepare',ROOT/'code/prepare_and_relax_once.py');b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
def feedback_smoke_review(run,ic,target):
 d=json.loads((run/'status.json').read_text());assert d['state']=='complete' and d['exit_code']==0
 checks={};before=0.;after=0.
 with h5py.File(ic,'r') as a,h5py.File(d['final_snapshot'],'r') as z:
  assert abs(float(z['Header'].attrs['Time'])-target)<1e-12
  for pt in ['PartType0','PartType1','PartType4']:
   x,y=a[pt],z[pt];ii=np.argsort(x['ParticleIDs'][:]);jj=np.argsort(y['ParticleIDs'][:])
   assert len(ii)==len(jj) and np.array_equal(x['ParticleIDs'][:][ii],y['ParticleIDs'][:][jj])
   for name in ['ParticleChildIDsNumber','ParticleIDGenerationNumber']:assert np.array_equal(x[name][:][ii],y[name][:][jj])
   for name in y:assert np.isfinite(y[name][:]).all(),(pt,name)
   aa=x['Masses'][:][ii];bb=y['Masses'][:][jj];before+=aa.sum();after+=bb.sum()
   checks[pt]={'initial_code_mass':float(aa.sum()),'final_code_mass':float(bb.sum()),'mass_change_Msun':float((bb.sum()-aa.sum())*1e10)}
   if pt=='PartType1':assert np.array_equal(aa,bb)
   if pt=='PartType4':
    assert np.array_equal(x['Metallicity'][:][ii],y['Metallicity'][:][jj])
    assert np.all(bb<=aa*(1+1e-12))
   if pt=='PartType0':
    assert np.all(bb>=aa*(1-1e-12))
    assert np.all(y['Metallicity'][:,0]>=x['Metallicity'][:,0].min()-1e-12)
  assert abs(after-before)<=max(abs(before),1)*1e-12
 result={'status':'pass','exit0':True,'exact_target':target,'identity_triplets_and_finite_fields':True,'mass_budget':checks,'total_mass_conserved':True,'original_failed_assumption':'Per-particle gas/star mass and gas metallicity need not stay fixed during active stellar feedback; original failed assertion preserved in first terminal log. This is real mass return/enrichment, not storage rounding.','checks':'Collisionless DM mass and stellar metallicity unchanged; gas gain balances star loss; gas metal enrichment permitted; no validator rerun or output edits'}
 b.save(run/'smoke_feedback_review01.json',result);print('FEEDBACK MASS BUDGET PASS',result,flush=True)
def main():
 assert os.environ.get('TMUX_PANE')=='%0' and os.isatty(0) and os.isatty(1)
 with (ROOT/'status/continue_relax_once01.json').open('x') as f:json.dump({'utc':b.now(),'pid':os.getpid(),'no_IC_or_LMC_smoke_repeat':True},f)
 with (b.OLD/'status/compute.flock').open('r+') as lease:
  fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);b.unique()
  for system,mass in [('LMC',1.1e9),('SMC',7.9e8)]:
   run=ROOT/'relaxation'/system.lower();b.inspect_generated(system,run,mass)
   base=(run/(system.lower()+'.param')).read_text();ic=run/(system.lower()+'_ic.hdf5')
   if system=='LMC':feedback_smoke_review(run/'smoke',ic,1e-5)
   for mode in (['production'] if system=='LMC' else ['smoke','production']):
    q=run/mode;q.mkdir();b.deps(q)
    target=1e-5 if mode=='smoke' else .6136
    updates={'InitCondFile':'../'+system.lower()+'_ic','OutputDir':'./output/','TimeLimitCPU':'288600','MaxMemSize':'1000','TimeBegin':'0','TimeMax':format(target,'.17g'),'TimeOfFirstSnapshot':'1e-8' if mode=='smoke' else '0','TimeBetSnapshot':'9.99e-6' if mode=='smoke' else '.1','MaxSizeTimestep':'1e-6' if mode=='smoke' else '.001','MinSizeTimestep':'1e-12','InitStellarAge':'5.0'}
    (q/'relax.param').write_text(b.change(base,updates))
    result=b.native(q,mode,system,target)
    if mode=='smoke':feedback_smoke_review(q,ic,target)
    else:b.stage(system.lower()+'_relaxation_complete',{'last_completed_relaxation':result})
  b.stage('both_satellites_relaxed_ready_for_infall_merge',{'active_run':None})
  b.save(ROOT/'status/relaxations_complete.json',{'utc':b.now(),'status':'complete','systems':['LMC','SMC'],'next':'New-family merge/static/smoke/infall; no jet'})
if __name__=='__main__':
 try:main()
 except BaseException as e:b.stage('relaxation_continuation_requires_review',{'error':repr(e)});raise
