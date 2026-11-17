#!/home/xingdi/gizmo-runs/.venv/bin/python
"""Read-only structural audit of existing six-rank GIZMO checkpoints.
Print JSON to stdout. Does not write checkpoints or launch GIZMO.
Layout comes from the exact frozen binary's DWARF via non-running GDB.
"""
from pathlib import Path
from datetime import datetime,timezone
import ctypes,ctypes.util,hashlib,json,re,struct,subprocess
R=Path('/home/xingdi/gizmo-runs/mw_isolated/cgm3_n3000_gas19_20261116/calibration01')
SRC=Path('/home/xingdi/gizmo-public-gizmo-cpp')
B=SRC/'GIZMO'
EXPECTED_BINARY='07154d9677d2e897a7dc557e37f9565d0be835a5e612943870cc5021b137f343'
EXPECTED_CONFIG='62490ddc581aa9648afd7338ecacdc4d0385b2dd76544bba5e097918d6d16180'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(B)==EXPECTED_BINARY
assert sha(SRC/'Config.sh')==EXPECTED_CONFIG
size_expressions={'All':'sizeof(struct global_data_all_processes)','particle':'sizeof(struct particle_data)','gas_cell':'sizeof(struct gas_cell_data)','NODE':'sizeof(struct NODE)','extNODE':'sizeof(struct extNODE)','topnode':'sizeof(struct topnode_data)','SelRnd':'sizeof(SelRnd)','TimeBinActive':'sizeof(TimeBinActive)','OutputDir':'sizeof(All.OutputDir)','RestartFile':'sizeof(All.RestartFile)'}
fields={'Time':'<d','TimeBegin':'<d','TimeMax':'<d','Ti_Current':'<q','Ti_nextoutput':'<q','SnapshotFileCount':'<i','TotNumPart':'<q','TotN_gas':'<q','MaxPart':'<i','MaxPartGas':'<i','OutputDir':'str','RestartFile':'str'}
expressions={**{'size_'+k:v for k,v in size_expressions.items()},**{'offset_'+k:'(long)&((struct global_data_all_processes*)0)->'+k for k in fields}}
cmd=['gdb','--batch','-nx','-q',str(B)]
for expr in expressions.values():cmd+=['-ex','p '+expr]
gdb=subprocess.run(cmd,capture_output=True,text=True,check=True)
values=[int(v) for v in re.findall(r'^\$\d+ = (\d+)\s*$',gdb.stdout,re.M)]
assert len(values)==len(expressions),(gdb.stdout,gdb.stderr)
layout=dict(zip(expressions,values))
g=ctypes.CDLL(ctypes.util.find_library('gsl'));g.gsl_rng_alloc.argtypes=[ctypes.c_void_p];g.gsl_rng_alloc.restype=ctypes.c_void_p;g.gsl_rng_size.argtypes=[ctypes.c_void_p];g.gsl_rng_size.restype=ctypes.c_size_t;g.gsl_rng_free.argtypes=[ctypes.c_void_p]
rng=g.gsl_rng_alloc(ctypes.c_void_p.in_dll(g,'gsl_rng_ranlxd1').value);rng_size=g.gsl_rng_size(rng);g.gsl_rng_free(rng)
config=(SRC/'GIZMO_config.h').read_text()
assert '#define GALSF ' in config
assert '#define CHIMES' not in config and '#define TURB_DRIVING' not in config
rows=[]
for rank in range(6):
 p=R/'production_resume01/output/restartfiles'/('restart.'+str(rank));before=p.stat()
 with p.open('rb') as f:
  header=f.read(layout['size_All'])
  npart=struct.unpack('<i',f.read(4))[0];f.seek(npart*layout['size_particle'],1)
  ngas=struct.unpack('<i',f.read(4))[0]
  f.seek(ngas*layout['size_gas_cell']+rng_size+layout['size_SelRnd']+layout['size_TimeBinActive'],1)
  gas_split,stars_conv,nmulti,ntopleaves,ntopnodes,nnodes=struct.unpack('<6i',f.read(24))
  assert all(v>=0 for v in [npart,ngas,gas_split,stars_conv,nmulti,ntopleaves,ntopnodes,nnodes])
  remaining=nnodes*(layout['size_NODE']+layout['size_extNODE'])+npart*8+ntopnodes*4+(6*nmulti*4)*2+ntopnodes*layout['size_topnode']+ntopnodes*4+ntopleaves*4+64
  calculated_bytes=f.tell()+remaining
 common={}
 for k,typ in fields.items():
  off=layout['offset_'+k]
  common[k]=header[off:off+layout['size_'+k]].split(b'\0')[0].decode() if typ=='str' else struct.unpack_from(typ,header,off)[0]
 row={'path':str(p),'rank':rank,'sha256':sha(p),'bytes':before.st_size,'mtime_utc':datetime.fromtimestamp(before.st_mtime,timezone.utc).isoformat(),'calculated_bytes':calculated_bytes,'exact_EOF':calculated_bytes==before.st_size,'All':common,'NumPart':npart,'N_gas':ngas,'Gas_split':gas_split,'Stars_converted':stars_conv,'MULTIPLEDOMAINS':nmulti,'NTopleaves':ntopleaves,'NTopnodes':ntopnodes,'Numnodestree':nnodes}
 after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 assert row['exact_EOF'];rows.append(row)
for k in fields:
 assert all(row['All'][k]==rows[0]['All'][k] for row in rows),k
assert sum(row['NumPart'] for row in rows)==rows[0]['All']['TotNumPart']
assert sum(row['N_gas']-row['Stars_converted'] for row in rows)==rows[0]['All']['TotN_gas']
assert .6136272158015191<rows[0]['All']['Time']<1.6363392421373844
assert rows[0]['All']['TimeBegin']==.6136272158015191 and rows[0]['All']['TimeMax']==1.6363392421373844
result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'structural_and_consistency_checks_passed','binary_sha256':sha(B),'config_sha256':sha(SRC/'Config.sh'),'generated_config_sha256':sha(SRC/'GIZMO_config.h'),'source_sha256':{str(SRC/p):sha(SRC/p) for p in ['file_io/restart.cc','core/begrun.cc']},'layout_source':'DWARF of the exact frozen executable, queried with GDB without run/start/attach; gsl_rng_size from installed ranlxd1 implementation','layout':layout,'gdb_stdout':gdb.stdout,'gdb_stderr':gdb.stderr,'gsl_rng_name':'gsl_rng_ranlxd1','gsl_rng_state_bytes':rng_size,'rows':rows,'sum_NumPart':sum(x['NumPart'] for x in rows),'sum_N_gas_slots':sum(x['N_gas'] for x in rows),'sum_Stars_converted':sum(x['Stars_converted'] for x in rows),'gas_count_note':'Excess stored gas slots are verified equal to pending Stars_converted; this is not a particle-loss claim.','restart_semantics':{'reads':'All, particle/gas state, RNG, active bins, tree/domain state; no IC reinitialization for flag1','path':'restart.cc reads supplied parameter OutputDir/restartfiles; begrun.cc418 resets OutputDir from supplied parameters after loading All','safe_plan':'Copy six regular shards into fresh resume output/restartfiles and run same frozen executable with -np6 and flag1; preserve original shards and outputs; use copies, not symlinks/hardlinks. Preserve preceding complete snapshot001..010 for diagnostic coverage; current SnapshotFileCount must be11.','timer':'begrun.cc503 resets TimeLastRestartFile after restoring All; checkpoint wall interval900 remains effective on resumed run.'},'limitations':['Structural exact-EOF and common-state checks do not replace actual same-binary flag1 load.','No dynamical test, particle-field finite test, or original write-time checksum comparison performed.','Original files opened read-only and not modified.']}
with (Path(__file__).parent/'checkpoint_audit.json').open('x') as f: json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
