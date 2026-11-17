
from pathlib import Path
import os,json,hashlib,subprocess,re
ROOT=Path(__file__).resolve().parent.parent
def require(x,m):
 if not x:raise ValueError(m)
def new_json(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def memory():
 return {x.split(':')[0]:int(x.split()[1]) for x in Path('/proc/meminfo').read_text().splitlines() if x.split(':')[0] in ['MemAvailable','SwapTotal','SwapFree']}
def compute_processes():
 out=[]
 for p in Path('/proc').glob('[0-9]*'):
  try:
   argv=(p/'cmdline').read_bytes().split(b'\0');name=Path(argv[0].decode()).name
   if name.startswith('GIZMO') or name in ['mpirun','mpiexec']:out.append({'pid':int(p.name),'command':[x.decode() for x in argv if x]})
  except (FileNotFoundError,PermissionError):pass
 return out
def verify_before_run(trial,mode):
 require(mode=='production' and trial.parent==ROOT,'Only direct formal production is enabled')
 require(os.isatty(0) and os.isatty(1) and os.environ.get('TMUX_PANE'),'Visible terminal required')
 require(subprocess.check_output(['tmux','-L','gizmo-orbits','list-clients'],text=True).strip(),'Attached terminal required')
 require(not compute_processes(),'Another GIZMO/MPI job exists')
 c=json.loads((ROOT/'campaign.json').read_text());a=json.loads((trial/'orbit_000.orbit.json').read_text())
 require(c['jet_authorized'] is False and c['smoke_required'] is False,'Unexpected scope')
 expected=json.loads((ROOT/'provenance/new_family_sources.json').read_text())
 require(a['source_sha256']==expected['source_sha256'] and a['source_files']==expected['source_files'],'Wrong new source family')
 require(a['model_key']==c['family_key'] and sha(ROOT/'provenance/cgm3_family_inputs.json')==c['family_inputs_sha256'],'Wrong new family binding')
 for p,h in json.loads((ROOT/'provenance/launch_hashes.json').read_text())['sha256'].items():require(sha(Path(p))==h,'Changed launch dependency '+p)
 params={}
 for line in (trial/'production/stage2.param').read_text().splitlines():
  w=re.split('[%#]',line,1)[0].split()
  if w:params[w[0]]=w[1]
 for k,v in {'CritPhysDensity':3000,'MaxKernelRadius':1e56,'MinSizeTimestep':1e-12,'MaxSizeTimestep':.001,'TimeBegin':.6136272158015191,'TimeMax':1.6363392421373844}.items():
  require(float(params[k])==v,'Wrong parameter '+k)
 require(float(params['BlackHoleAccretionFactor'])==float(params['BlackHoleEddingtonFactor'])==float(params['BlackHoleFeedbackFactor'])==0,'AGN not off')
