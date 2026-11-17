
from pathlib import Path
import os,time,json,fcntl,importlib.util
P=Path('/home/xingdi/gizmo-runs');ROOT=P/'mw_isolated/cgm3_n3000_gas19_20261116'
sp=importlib.util.spec_from_file_location('base_run',ROOT/'code/prepare_and_relax_once.py');b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
assert os.isatty(0) and os.isatty(1)
with (ROOT/'status/no_smoke_followthrough_once.json').open('x') as f:json.dump({'utc':b.now(),'pid':os.getpid(),'policy':'No future smoke; preserve LMC formal run'},f)
old=Path('/proc/25882')
print('Waiting for existing LMC formal relaxation; pending SMC smoke has been cancelled.',flush=True)
while old.exists():
 try:
  cmd=(old/'cmdline').read_bytes()
  if b'continue_relax_once01.py' not in cmd:break
 except FileNotFoundError:break
 time.sleep(3)
p=ROOT/'relaxation/lmc/production/status.json';s=json.loads(p.read_text())
assert s['state']=='complete' and s['exit_code']==0 and abs(s['final_time']-.6136)<1e-12,'LMC not complete; preserve evidence'
assert not (ROOT/'relaxation/smc/smoke/status.json').exists(),'Cancelled smoke unexpectedly started'
with (b.OLD/'status/compute.flock').open('r+') as lease:
 fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);b.unique()
 b.save(ROOT/'status/legacy_pending_smoke_cancelled.json',{'state':'expected_workflow_stop_after_user_policy_change','LMC_formal_exit':0,'pending_SMC_smoke':'cancelled with no launch','legacy_terminal_evidence':str(ROOT/'status/relaxation_continue_terminal01.log')})
 d=json.loads((ROOT/'campaign.json').read_text());d.pop('error',None);d.update(stage='smc_direct_production_preparing',smoke_required=False,active_run=None);b.save(ROOT/'campaign.json',d)
 run=ROOT/'relaxation/smc';q=run/'production';q.mkdir();b.deps(q)
 base=(run/'smc.param').read_text()
 (q/'relax.param').write_text(b.change(base,{'InitCondFile':'../smc_ic','OutputDir':'./output/','TimeBegin':'0','TimeMax':'.6136','TimeOfFirstSnapshot':'0','TimeBetSnapshot':'.1','MaxSizeTimestep':'.001','MinSizeTimestep':'1e-12','MaxMemSize':'1000','InitStellarAge':'5.0'}))
 result=b.native(q,'production','SMC',.6136)
 b.stage('both_satellites_relaxed_ready_for_direct_infall',{'active_run':None,'last_completed_relaxation':result,'smoke_required':False})
 b.save(ROOT/'status/relaxations_complete.json',{'utc':b.now(),'status':'complete','systems':['LMC','SMC'],'smoke_policy':'No future smoke; historic LMC smoke retained; SMC formal directly','next':'new-family merged IC -> direct formal infall; no jet'})
entry=ROOT/'code/proceed_infall_once.py'
if entry.exists():
 import subprocess
 subprocess.run([str(P/'.venv/bin/python'),'-B','-u',str(entry)],cwd=ROOT,check=True)
