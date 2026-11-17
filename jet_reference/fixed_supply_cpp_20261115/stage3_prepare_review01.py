#!/home/xingdi/gizmo-runs/.venv/bin/python
"""Static preparation interfaces; default plan is read-only. No stage starts automatically."""
from __future__ import annotations
import argparse,contextlib,datetime,fcntl,hashlib,json,math,os,re,shutil,subprocess,sys
from pathlib import Path
AJET=Path('/home/xingdi/gizmo-runs/compat_validation/20261115_fixed_supply_cpp_port_review01')
CAMPAIGN=Path('/home/xingdi/gizmo-runs/mw_isolated/cgm3_n3000_orbits_cpp_20261105_084436')
RUNS=Path('/home/xingdi/gizmo-runs/erosita_xray')
PYTHON=Path('/home/xingdi/gizmo-runs/.venv/bin/python')
AZAXIS=RUNS/'azaxis.py'
JET_REFERENCE=RUNS/'erosita.param'
DEFAULT_STAGE=RUNS/'cgm3_n3000_cpp_fixed_supply_20261115'
FROZEN_INPUTS={'/home/xingdi/gizmo-runs/compat_validation/20261115_fixed_supply_cpp_port_review01/stage3_contract.json': '020450d2dc9a4241a223df7c55a75e78cac7af0da7ea6ac73f68e95d320e55f2', '/home/xingdi/gizmo-runs/compat_validation/20261115_fixed_supply_cpp_port_review01/sink_input_schema.json': '1b44f897736b824d6e66b2f591d231f461715624a5c21a91450f7a8e555819b2', '/home/xingdi/gizmo-runs/compat_validation/20261115_fixed_supply_cpp_port_review01/source_port_manifest.json': '09621c0f0d1220c08e3fb78d9c5e3abe18890f529d15de2a8b67b82367f545c4', '/home/xingdi/gizmo-runs/erosita_xray/azaxis.py': 'a801866391d225b042fecbec5b0e2d0e1cf03159e5a09629dc951f1cdeff0459', '/home/xingdi/gizmo-runs/erosita_xray/erosita.param': 'df200fec6feeac7182abdd5332fd38a647e081aa4a5b36f7827b00c1c7ebd7f0', '/home/xingdi/gizmo-runs/compat_validation/20261115_fixed_supply_cpp_port_review01/stage3_runtime_inputs_contract.json': '0c81dcde6b035d7ca9f7f06d153f3280ccc080af02314e1a7ac4fd4e51bdc24a'}
SMOKE_MYR=.3
SMOKE_OUTPUT_CODE=.00005
def require(ok,message):
    if not ok:raise RuntimeError(message)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def regular(path,exists=True):
    p=Path(path).expanduser();require(p.is_absolute(),'absolute path required: '+str(p))
    require(not p.is_symlink(),'symlink rejected: '+str(p));q=p.resolve(strict=exists)
    require(p==q,'resolved path differs: '+str(p))
    if exists:require(q.is_file(),'regular file required: '+str(q))
    return q
def stage_path(path):
    p=Path(path).expanduser();require(p.is_absolute(),'stage root must be absolute');q=p.resolve()
    require(q==p and q.parent==RUNS and re.fullmatch(r'cgm3_n3000_cpp_fixed_supply_[A-Za-z0-9_]+',q.name),
            'stage root must be a cgm3_n3000_cpp_fixed_supply_* direct child of erosita_xray')
    return q
def load(path):return json.loads(regular(path).read_text())
def artifact(path):return {'path':str(regular(path)),'sha256':sha(path)}
def verify_binding(record):
    require(set(record)>={'path','sha256'},'invalid artifact binding')
    p=regular(record['path']);require(sha(p)==record['sha256'],'artifact SHA changed: '+str(p));return p
def contract_and_schema():
    for fn,digest in FROZEN_INPUTS.items():require(sha(regular(fn))==digest,'preparation input drift: '+fn)
    c=load(AJET/'stage3_contract.json');s=load(AJET/'sink_input_schema.json')
    require(len(c['parameter_translations'])==13 and len(s['additional_fields'])==6,'contract size mismatch')
    return c,s
def exclusive_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    require(p.resolve()==p and not p.is_symlink(),'unsafe JSON target')
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
def ancestors(pid):
    chain=[]
    while pid>1 and pid not in chain:
        chain.append(pid)
        try:pid=int(next(x.split()[1] for x in Path(f'/proc/{pid}/status').read_text().splitlines() if x.startswith('PPid:')))
        except (OSError,StopIteration):break
    return chain
def visible_terminal():
    require(os.isatty(0),'must run in attached visible Ubuntu terminal')
    pane=os.environ.get('TMUX_PANE');require(pane and re.fullmatch(r'%[0-9]+',pane),'dedicated tmux pane required')
    record=subprocess.check_output(['tmux','-L','gizmo-orbits','display-message','-p','-t',pane,
                                    '#{session_name}|#{pane_pid}|#{pane_tty}'],text=True).strip().split('|')
    require(len(record)==3 and record[0]=='cgm3_n3000_orbit','wrong dedicated tmux session')
    require(int(record[1]) in ancestors(os.getpid()),'not a descendant of visible pane')
    clients=subprocess.check_output(['tmux','-L','gizmo-orbits','list-clients','-F',
                                     '#{client_pid}|#{session_name}|#{client_tty}'],text=True).splitlines()
    attached=[x for x in clients if len(x.split('|'))==3 and x.split('|')[1]==record[0]]
    require(attached,'no attached visible terminal client')
    for x in attached:require(Path('/proc/'+x.split('|')[0]).exists(),'attached client PID disappeared')
    return {'pane':pane,'session':record[0],'pane_pid':int(record[1]),'pane_tty':record[2],
            'stdin_tty':os.ttyname(0),'attached_clients':attached,'ancestry':ancestors(os.getpid())}
def active_compute():
    found=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=p.read_bytes().split(b'\0');exe=Path(os.fsdecode(args[0])).name if args and args[0] else ''
            if exe.startswith('GIZMO') or exe in ('mpirun','mpiexec','orterun'):
                found.append({'pid':int(p.parent.name),'argv':[os.fsdecode(x) for x in args if x]})
        except (OSError,ProcessLookupError):pass
    return found
@contextlib.contextmanager
def compute_lease():
    with regular(CAMPAIGN/'status/compute.flock').open('r+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not active_compute(),'another GIZMO/MPI job is active')
        yield
def parameter_map(content):
    out={}
    for line in content.splitlines():
        fields=line.split('%',1)[0].split('#',1)[0].split()
        if not fields:continue
        require(len(fields)==2,'parameter needs exactly key and value: '+line)
        require(fields[0] not in out,'duplicate parameter: '+fields[0]);out[fields[0]]=fields[1]
    return out
def inventory(h5):
    import h5py
    items=[]
    h5.visititems(lambda name,obj:items.append({'path':name,'shape':list(obj.shape),'dtype':str(obj.dtype)})
                 if isinstance(obj,h5py.Dataset) and name.startswith('PartType') else None)
    return sorted(items,key=lambda x:x['path'])
def phase_time(c,phase):
    begin=c['timeline']['TimeBegin'];duration=SMOKE_MYR if phase=='smoke' else 30.
    return begin,begin+duration/1000./c['timeline']['unit_time_Gyr'],SMOKE_OUTPUT_CODE if phase=='smoke' else .001
def write_parameters(root):
    c,_=contract_and_schema();root=stage_path(root);base=regular(c['current_nonjet_param_base'])
    require(sha(base)==c['current_nonjet_param_base_sha256'],'parameter base changed')
    before=parameter_map(base.read_text());names={x['legacy']:x for x in c['parameter_translations']}
    require(set(names)<=set(before),'missing legacy jet parameters')
    require(not any(x['canonical'] in before for x in names.values()),'canonical/legacy collision')
    for k,v in c['explicit_preserved_controls'].items():
        require(k in before and math.isclose(float(before[k]),float(v),rel_tol=1e-14,abs_tol=0),'baseline drift: '+k)
    jet_reference=parameter_map(regular(JET_REFERENCE).read_text())
    require(set(names)<=set(jet_reference),'missing historical jet reference parameter')
    for x in names.values():
        require(math.isclose(float(jet_reference[x['legacy']]),x['value'],rel_tol=1e-14,abs_tol=0),
                'jet contract differs from frozen historical jet reference: '+x['legacy'])
    transitions=[{'legacy':x['legacy'],'before':before[x['legacy']],'canonical':x['canonical'],
                  'after':format(x['value'],'.17g'),'jet_reference_value':jet_reference[x['legacy']]} for x in names.values()]
    report_path=root/'preparation/parameter_translation.json'
    require(not any(p.exists() or p.is_symlink() for p in [report_path]+[root/m/'stage3.param' for m in ('smoke','production')]),'parameters already prepared')
    runtime=load(AJET/'stage3_runtime_inputs_contract.json')
    for row in runtime['files']:
        require(sha(regular(row['source']))==row['sha256'],'accepted cooling input drift: '+row['source'])
        for phase in ('smoke','production'):
            destination=root/phase/row['relative_path']
            require(destination.is_relative_to(root/phase) and not destination.exists() and not destination.is_symlink(),
                    'runtime destination already exists or escapes phase')
    details={}
    for phase in ('smoke','production'):
        begin,end,spacing=phase_time(c,phase)
        updates={'InitCondFile':str(root/'inputs/jet_000'),'OutputDir':'./output/','TimeBegin':format(begin,'.17g'),
                 'TimeMax':format(end,'.17g'),'TimeOfFirstSnapshot':format(begin,'.17g'),'TimeBetSnapshot':format(spacing,'.17g')}
        require(set(updates)<=set(before),'missing timeline/path key');lines=[]
        for line in base.read_text().splitlines():
            words=line.split('%',1)[0].split('#',1)[0].split()
            if not words:lines.append(line);continue
            key=words[0]
            if key in names:
                row=names[key];lines.append(f"{row['canonical']:<38} {row['value']:.17g}")
            elif key in updates:lines.append(f'{key:<38} {updates[key]}')
            else:lines.append(line)
        payload='\n'.join(lines)+'\n';after=parameter_map(payload)
        require(not (set(names)&set(after)),'old parameter remains');unchanged=set(before)-set(names)-set(updates)
        require(all(before[k]==after[k] for k in unchanged),'nonjet parameter drift')
        path=root/phase/'stage3.param';path.parent.mkdir(parents=True,exist_ok=True)
        require(path.parent.resolve()==path.parent,'symlinked phase directory')
        with path.open('x') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        runtime_records=[]
        for row in runtime['files']:
            destination=root/phase/row['relative_path'];destination.parent.mkdir(parents=True,exist_ok=True)
            require(destination.parent.resolve()==destination.parent,'symlinked runtime directory')
            with regular(row['source']).open('rb') as inp,destination.open('xb') as out:
                shutil.copyfileobj(inp,out,8*1024*1024);out.flush();os.fsync(out.fileno())
            require(sha(destination)==row['sha256'] and sha(regular(row['source']))==row['sha256'],'runtime copy/source SHA mismatch')
            runtime_records.append(dict(path=str(destination),sha256=row['sha256'],source=row['source'],bytes=row['bytes']))
        details[phase]={'params':artifact(path),'runtime_inputs':runtime_records,'time_begin':begin,'target_time':end,
                        'duration_physical_Myr':SMOKE_MYR if phase=='smoke' else 30.,
                        'output_interval_code':spacing,'restart_flag':2,'changed_keys':sorted(set(names)|set(updates)),
                        'unchanged_parameter_count':len(unchanged)}
    report={'schema':'fixed-supply-jet-parameter-translation/v1','status':'prepared_not_launched','created_utc':now(),
            'stage_root':str(root),'base':artifact(base),'contract':artifact(AJET/'stage3_contract.json'),
            'preparation_script':artifact(Path(__file__)),'jet_reference':artifact(JET_REFERENCE),
            'runtime_inputs_contract':artifact(AJET/'stage3_runtime_inputs_contract.json'),
            'translations':c['parameter_translations'],'transitions':transitions,'phases':details}
    exclusive_json(report_path,report);print(json.dumps(report,indent=2))
def initialize_ic(root,clearance_path):
    import h5py,numpy as np
    c,s=contract_and_schema();root=stage_path(root);visible=visible_terminal();clearance=load(clearance_path)
    require(clearance.get('schema')=='fixed-supply-jet-initialization-clearance/v1' and
            clearance.get('status')=='approved_for_jet_ic_initialization' and clearance.get('stage_root')==str(root),'initialization clearance mismatch')
    require(set(clearance.get('bindings',{}))=={'plot_review','accepted_review','source_review'},'initialization evidence roles mismatch')
    for binding in clearance['bindings'].values():verify_binding(binding)
    source=regular(c['input_endpoint']);expected=c['input_endpoint_sha256_from_gate'];output=root/'inputs/jet_000.hdf5'
    prep=root/'preparation';axis_report=prep/'azaxis_report.json';manifest=prep/'ic_initialization.json';receipt=prep/'ic_initialization_once.json'
    require(not any(p.exists() or p.is_symlink() for p in [output,output.with_suffix('.partial'),axis_report,manifest,receipt]),'initialization already attempted')
    with compute_lease():
        before=sha(source);require(before==expected,'accepted endpoint SHA mismatch')
        with h5py.File(source,'r') as f:
            inv=inventory(f);require(len(inv)==56,'actual endpoint inventory changed')
            require(float(f['Header'].attrs['Time'])==c['timeline']['TimeBegin'],'endpoint Time mismatch')
            require(np.array_equal(f['PartType5/ParticleIDs'][...],s['existing_BH_ParticleIDs']),'BH identity mismatch')
            require(np.array_equal(f['PartType5/Masses'][...],s['existing_BH_Masses_code']),'BH mass mismatch')
            for item in s['additional_fields']:require('PartType5/'+item['name'] not in f,'existing sink state must not be reset')
            require('BH_Specific_AngMom' not in f['PartType5'],'legacy axis alias present')
        prep.mkdir(parents=True,exist_ok=True);output.parent.mkdir(parents=True,exist_ok=True)
        require(prep.resolve()==prep and output.parent.resolve()==output.parent,'symlinked prep/inputs directory')
        exclusive_json(receipt,{'schema':'fixed-supply-jet-initialization-once/v1','created_utc':now(),'pid':os.getpid(),
                               'stage_root':str(root),'source':str(source),'output':str(output),'clearance':artifact(clearance_path),'visible_terminal':visible})
        partial=output.with_suffix('.partial')
        with source.open('rb') as src,partial.open('xb') as dst:
            shutil.copyfileobj(src,dst,8*1024*1024);dst.flush();os.fsync(dst.fileno())
        require(sha(partial)==before,'full-copy SHA mismatch')
        os.link(partial,output);partial.unlink() # only publish a complete new owned copy, no overwrite
        require(output.stat().st_nlink==1,'owned IC must have one hard link')
        subprocess.run([str(PYTHON),'-B',str(AZAXIS),'--input',str(output),'--in-place','--target-format','cpp',
                        '--axis','0','0','1','--report',str(axis_report)],check=True)
        with h5py.File(output,'r+') as f:
            group=f['PartType5'];count=len(group['ParticleIDs'])
            for item in s['additional_fields']:
                if item['name']=='Sink_Specific_AngMom':continue
                shape=tuple(count if v=='N_BH' else v for v in item['shape']);value=item.get('initial_value_code',item.get('initial_value'))
                group.create_dataset(item['name'],data=np.full(shape,value,dtype=item['dtype']),dtype=item['dtype'])
            f.flush();new_inv=inventory(f)
        after=sha(source);require(after==before,'source changed during initialization');require(len(new_inv)==62,'six-field inventory mismatch')
        result={'schema':'fixed-supply-jet-ic-initialization/v1','status':'initialized_pending_independent_review','created_utc':now(),
                'stage_root':str(root),'source':str(source),'source_sha256':before,'source_sha256_before':before,'source_sha256_after':after,
                'input_endpoint_time':c['timeline']['TimeBegin'],'output':str(output),'output_sha256':sha(output),
                'source_dataset_count':len(inv),'output_dataset_count':len(new_inv),
                'added_fields':sorted('PartType5/'+x['name'] for x in s['additional_fields']),
                'source_inventory':inv,'output_inventory':new_inv,'azaxis_tool':str(AZAXIS),'azaxis_tool_sha256':sha(AZAXIS),
                'azaxis_report':str(axis_report),'azaxis_report_sha256':sha(axis_report),
                'sink_schema_sha256':sha(AJET/'sink_input_schema.json'),'contract_sha256':sha(AJET/'stage3_contract.json'),
                'preparation_script':artifact(Path(__file__)),'clearance':artifact(clearance_path),'evidence_bindings':clearance['bindings'],
                'full_field_values_verified':False,'independent_review_required':True}
        exclusive_json(manifest,result);print(json.dumps(result,indent=2))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','translate-params','initialize-ic'],nargs='?',default='plan')
    parser.add_argument('--stage-root',type=Path,default=DEFAULT_STAGE);parser.add_argument('--clearance',type=Path)
    args=parser.parse_args();root=stage_path(args.stage_root)
    if args.action=='translate-params':write_parameters(root)
    elif args.action=='initialize-ic':
        require(args.clearance is not None,'--clearance required');initialize_ic(root,args.clearance)
    else:
        c,s=contract_and_schema()
        print(json.dumps({'state':'plan_only','stage_root':str(root),'ic':str(root/'inputs/jet_000.hdf5'),
         'azaxis_report':str(root/'preparation/azaxis_report.json'),'ic_manifest':str(root/'preparation/ic_initialization.json'),
         'parameter_manifest':str(root/'preparation/parameter_translation.json'),
         'phases':{p:dict(zip(['time_begin','target_time','snapshot_interval_code'],phase_time(c,p))) for p in ['smoke','production']},
         'additional_sink_fields':s['additional_fields'],'no_stage_executed':True},indent=2))
if __name__=='__main__':main()
