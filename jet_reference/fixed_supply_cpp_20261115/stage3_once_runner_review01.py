#!/home/xingdi/gizmo-runs/.venv/bin/python
"""Single-attempt flag2 stage runner. Requires independently issued artifact-bound clearance."""
from __future__ import annotations
import argparse,json,math,os,shlex,subprocess,sys,threading,time
from pathlib import Path
import stage3_prepare_review01 as prep
require=prep.require
COMMON_ROLES={'binary','config','generated_config','source_manifest','source_review','ic','ic_manifest','ic_review',
              'params','param_manifest','contract','sink_schema','preparation_script','runner_script','plot_review'}
def update_status(path,data):
    tmp=path.with_name('.status-update-'+str(os.getpid())+'.json')
    with tmp.open('x') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path) # only this claimed stage's status; never its single-attempt receipt or scientific outputs
def verify_launch(root,phase,clearance_path):
    c,s=prep.contract_and_schema();root=prep.stage_path(root);clearance=prep.load(clearance_path)
    require(clearance.get('schema')=='fixed-supply-jet-launch-clearance/v1' and
            clearance.get('status')=='approved_for_flag2_'+phase and clearance.get('phase')==phase and
            clearance.get('stage_root')==str(root),'launch clearance does not match this phase/root')
    require(clearance.get('mpi_ranks')==6 and clearance.get('omp_threads')==1 and clearance.get('nice')==19 and
            clearance.get('restart_flag')==2,'resource/restart clearance mismatch')
    bindings=clearance.get('bindings',{});roles=COMMON_ROLES|({'smoke_review','smoke_status'} if phase=='production' else set())
    require(set(bindings)==roles,'launch artifact roles mismatch')
    paths={k:prep.verify_binding(v) for k,v in bindings.items()}
    expected={'binary':prep.AJET/'source/GIZMO_fixed_supply_cpp','config':prep.AJET/'source/Config.sh',
      'generated_config':prep.AJET/'source/GIZMO_config.h','source_manifest':prep.AJET/'source_port_manifest.json',
      'source_review':prep.AJET/'independent_source_review.json','ic':root/'inputs/jet_000.hdf5',
      'ic_manifest':root/'preparation/ic_initialization.json','ic_review':root/'preparation/ic_independent_review01.json',
      'params':root/phase/'stage3.param','param_manifest':root/'preparation/parameter_translation.json',
      'contract':prep.AJET/'stage3_contract.json','sink_schema':prep.AJET/'sink_input_schema.json',
      'preparation_script':prep.AJET/'stage3_prepare_review01.py','runner_script':Path(__file__).resolve()}
    if phase=='production':expected.update(smoke_review=root/'smoke/smoke_review.json',smoke_status=root/'smoke/status.json')
    require(all(paths[k]==v for k,v in expected.items()),'artifact role resolved to wrong path')
    require(os.access(paths['binary'],os.X_OK),'compiled binary is not executable')
    source_manifest=prep.load(paths['source_manifest'])
    for row in source_manifest['files']:
        require(prep.sha(prep.AJET/'source'/row['path'])==row['sha256'],'prepared source drift: '+row['path'])
    source_review=prep.load(paths['source_review']);require(source_review.get('source_level_pass') is True,'source review did not pass')
    manifest=prep.load(paths['ic_manifest']);ic_review=prep.load(paths['ic_review'])
    require(manifest.get('output')==str(paths['ic']) and manifest.get('output_sha256')==bindings['ic']['sha256'] and
            manifest.get('source_sha256')==c['input_endpoint_sha256_from_gate'],'IC manifest mismatch')
    require(ic_review.get('status')=='pass' and ic_review.get('candidate')==str(paths['ic']) and
            ic_review.get('candidate_sha256')==bindings['ic']['sha256'] and
            ic_review.get('manifest_sha256')==bindings['ic_manifest']['sha256'],'independent IC review mismatch')
    compiled=ic_review.get('compiled_config_schema_review',{})
    require(compiled.get('status')=='pass_static_generated_config_and_native_source_binding','IC compiled-schema review is missing')
    for role in ('binary','config','generated_config'):
        product=compiled.get('products',{}).get(role,{})
        require(product.get('path')==str(paths[role]) and product.get('sha256')==bindings[role]['sha256'],
                'current build differs from IC schema review: '+role)
    build_path=prep.verify_binding(compiled['build_manifest'])
    require(build_path==prep.AJET/'build_review01/complete.json','unexpected isolated build receipt')
    build=prep.load(build_path)
    require(build.get('status')=='build_pass_only' and build.get('exit_code')==0,'isolated recompilation did not complete')
    for role in ('binary','config','generated_config'):
        require(build[role]['path']==str(paths[role]) and build[role]['sha256']==bindings[role]['sha256'],'build product binding drift: '+role)
    require(manifest.get('preparation_script')==bindings['preparation_script'],'IC initialization code binding changed')
    pm=prep.load(paths['param_manifest'])
    require(pm.get('preparation_script')==bindings['preparation_script'],'parameter preparation code binding changed')
    require(pm.get('stage_root')==str(root) and pm['phases'][phase]['params']==bindings['params'],'parameter translation record mismatch')
    runtime_contract=prep.load(prep.AJET/'stage3_runtime_inputs_contract.json')
    require(pm.get('runtime_inputs_contract')==prep.artifact(prep.AJET/'stage3_runtime_inputs_contract.json'),'runtime input contract changed')
    expected_runtime={str(root/phase/row['relative_path']):row for row in runtime_contract['files']}
    actual_runtime=pm['phases'][phase].get('runtime_inputs',[])
    require(len(actual_runtime)==len(expected_runtime) and {r['path'] for r in actual_runtime}==set(expected_runtime),'runtime input inventory differs')
    for row in actual_runtime:
        reference=expected_runtime[row['path']]
        require(row['sha256']==reference['sha256'] and prep.sha(prep.regular(row['path']))==reference['sha256'],
                'cooling runtime input changed: '+row['path'])
    params=prep.parameter_map(paths['params'].read_text());begin,end,spacing=prep.phase_time(c,phase)
    require(float(params['TimeBegin'])==begin and float(params['TimeMax'])==end and
            float(params['TimeOfFirstSnapshot'])==begin and float(params['TimeBetSnapshot'])==spacing,'exact physical timeline mismatch')
    require(params['InitCondFile']==str(root/'inputs/jet_000') and params['OutputDir']=='./output/','stage input/output path mismatch')
    for item in c['parameter_translations']:
        require(item['legacy'] not in params and math.isclose(float(params[item['canonical']]),item['value'],rel_tol=1e-14,abs_tol=0),
                'jet parameter mismatch: '+item['canonical'])
    for key,value in c['explicit_preserved_controls'].items():
        require(math.isclose(float(params[key]),value,rel_tol=1e-14,abs_tol=0),'nonjet control drift: '+key)
    require(clearance.get('time_begin')==begin and clearance.get('target_time')==end,'clearance time mismatch')
    if phase=='production':
        smoke_status=prep.load(paths['smoke_status']);smoke_review=prep.load(paths['smoke_review'])
        require(smoke_status.get('state')=='complete' and smoke_status.get('exit_code')==0 and smoke_review.get('status')=='pass',
                'independent actual jet smoke is not complete/pass')
        for role in COMMON_ROLES-{'params'}:
            require(smoke_status['bindings'][role]==bindings[role],'production differs from smoke: '+role)
        require(smoke_review.get('status_sha256')==bindings['smoke_status']['sha256'],'smoke review/status binding mismatch')
    return c,clearance,paths,begin,end
def process_record(pid):
    p=Path('/proc')/str(pid)
    status={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in (p/'status').read_text().splitlines() if ':' in line}
    stat=(p/'stat').read_text().rsplit(')',1)[1].split();environ={}
    for raw in (p/'environ').read_bytes().split(b'\0'):
        if raw.startswith(b'OMP_NUM_THREADS='):environ['OMP_NUM_THREADS']=os.fsdecode(raw.split(b'=',1)[1])
    fds={}
    for fd in (0,1,2):
        try:fds[str(fd)]=os.readlink(p/'fd'/str(fd))
        except OSError:fds[str(fd)]=None
    return {'pid':pid,'ppid':int(status['PPid']),'name':status['Name'],'comm':(p/'comm').read_text().strip(),
            'cwd':os.readlink(p/'cwd'),'command':[os.fsdecode(x) for x in (p/'cmdline').read_bytes().split(b'\0') if x],
            'nice':int(stat[16]),'ni':int(stat[16]),'rss_kib':int(status.get('VmRSS','0 kB').split()[0]),
            'threads':int(status['Threads']),'OMP_NUM_THREADS':environ.get('OMP_NUM_THREADS'),'environ':environ,
            'ancestry':prep.ancestors(pid),'fd_targets':fds,'fd_links':fds,'tty':fds.get('0'),'tty_nr':int(stat[4])}
def sample_processes(transport_pid,known_mpi_pid=None):
    records=[]
    for p in Path('/proc').glob('[0-9]*'):
        pid=int(p.name)
        try:
            if transport_pid in prep.ancestors(pid):records.append(process_record(pid))
        except (OSError,ProcessLookupError,ValueError,KeyError):pass
    mpi=[r for r in records if r['command'] and Path(r['command'][0]).name in ('mpirun','mpiexec','orterun','prterun')]
    return {'utc':prep.now(),'transport_pid':transport_pid,'mpi_pid':mpi[0]['pid'] if len(mpi)==1 else known_mpi_pid,
            'mpi_candidates':[r['pid'] for r in mpi],'processes':sorted(records,key=lambda x:x['pid'])}
def resource_monitor(transport_pid,stop,path,info):
    try:
        with path.open('x') as f:
            count=0
            while True:
                sample=sample_processes(transport_pid,info.get('mpi_pid'))
                if sample['mpi_candidates']:
                    require(len(sample['mpi_candidates'])==1,'multiple actual MPI descendants under one transport')
                    info['mpi_pid']=sample['mpi_candidates'][0]
                    mpi=next(r for r in sample['processes'] if r['pid']==info['mpi_pid'])
                    info['mpi_pty_evidence']={'pid':mpi['pid'],'cwd':mpi['cwd'],'command':mpi['command'],
                                             'fd_targets':mpi['fd_targets'],'fd_links':mpi['fd_links'],'tty_nr':mpi['tty_nr'],'ancestry':mpi['ancestry']}
                f.write(json.dumps(sample)+'\n');f.flush();count+=1
                if stop.wait(.25 if count<20 else 5.):break
            f.write(json.dumps(sample_processes(transport_pid,info.get('mpi_pid')))+'\n');f.flush()
    except BaseException as exc:info['monitor_error']=repr(exc)
def restart_inventory(stage):
    files=[]
    for p in sorted((stage/'output').rglob('*')) if (stage/'output').exists() else []:
        if p.is_file() and 'restart' in p.name.lower():
            require(not p.is_symlink() and p.resolve().is_relative_to(stage),'redirected restart file')
            files.append({'path':str(p),'bytes':p.stat().st_size,'sha256':prep.sha(p)})
    return files
def run(root,phase,clearance_path):
    root=prep.stage_path(root);stage=root/phase;visible=prep.visible_terminal()
    require(stage.is_dir() and stage.resolve()==stage,'phase directory is not prepared')
    claims=['send_once.json','status.json','run.log','resource_samples.jsonl']
    require(not any((stage/x).exists() or (stage/x).is_symlink() for x in claims),'stage already attempted; never relaunch')
    output=stage/'output'
    require(not output.exists() or (output.is_dir() and not any(output.iterdir()) and not output.is_symlink()),'output is not new/empty')
    with prep.compute_lease():
        c,clearance,paths,begin,end=verify_launch(root,phase,clearance_path)
        native=['nice','-n','19','mpirun','-np','6',str(paths['binary']),str(paths['params']),'2']
        transport=['script','-q','-e','-f','-c',shlex.join(native),str(stage/'run.log')]
        claim={'schema':'fixed-supply-jet-stage-once/v1','phase':phase,'stage_root':str(root),'created_utc':prep.now(),
               'pid':os.getpid(),'command':native,'transport_command':transport,'cwd':str(stage),
               'clearance':prep.artifact(clearance_path),'visible_terminal':visible}
        prep.exclusive_json(stage/'send_once.json',claim)
        state={'schema':'fixed-supply-jet-stage-status/v1','phase':phase,'stage_root':str(root),'started_utc':prep.now(),
               'finished_utc':None,'pid':os.getpid(),'transport_pid':None,'mpi_pid':None,'command':native,'transport_command':transport,
               'cwd':str(stage),'state':'starting','exit_code':None,'bindings':clearance['bindings'],'clearance':prep.artifact(clearance_path),
               'time_begin':begin,'target_time':end,'mpi_ranks':6,'omp_threads':1,'nice':19,'restart_flag':2,
               'visible_terminal':visible,'inner_pty_required':True,'runtime_science_validated':False}
        prep.exclusive_json(stage/'status.json',state)
        env=os.environ.copy();env['OMP_NUM_THREADS']='1'
        process=None;stop=threading.Event();monitor=None;info={}
        try:
            # Only the outer script transport is piped. It assigns a real inner PTY
            # to the native MPI command and is the sole writer of stage/run.log.
            process=subprocess.Popen(transport,cwd=stage,env=env,stdin=None,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            state.update(transport_pid=process.pid,state='running');update_status(stage/'status.json',state)
            monitor=threading.Thread(target=resource_monitor,args=(process.pid,stop,stage/'resource_samples.jsonl',info),daemon=True);monitor.start()
            for line in process.stdout:
                sys.stdout.buffer.write(line);sys.stdout.buffer.flush()
                if info.get('mpi_pid') is not None and state['mpi_pid'] is None:
                    state.update(mpi_pid=info['mpi_pid'],mpi_pty_evidence=info.get('mpi_pty_evidence'))
                    update_status(stage/'status.json',state)
            code=process.wait();stop.set();monitor.join()
            state.update(mpi_pid=info.get('mpi_pid'),mpi_pty_evidence=info.get('mpi_pty_evidence'))
            require('monitor_error' not in info,'resource monitor failed: '+str(info.get('monitor_error')))
            if code==0:
                require(info.get('mpi_pid') is not None,'no actual MPI PID observed')
                fds=info['mpi_pty_evidence']['fd_targets']
                require(all(isinstance(fds.get(str(i)),str) and fds[str(i)].startswith('/dev/pts/') for i in (1,2)),
                        'actual MPI stdout/stderr were not an inner PTY')
            # script -e returns the native command exit; its run.log is closed now.
            state.update(state='complete' if code==0 else 'failed',exit_code=code,finished_utc=prep.now(),
                         run_log=prep.artifact(stage/'run.log'),resource_samples=prep.artifact(stage/'resource_samples.jsonl'),
                         restart_files=restart_inventory(stage))
            update_status(stage/'status.json',state)
            print(json.dumps({'phase':phase,'state':state['state'],'exit_code':code,'target_time':end,
                              'runtime_science_validated':False,'independent_review_required':True}),flush=True)
            return code
        except BaseException as exc:
            stop.set()
            if monitor is not None:monitor.join(timeout=10)
            state.update(state='supervision_error_process_may_still_be_running' if process is not None and process.poll() is None else 'failed',
                         error=repr(exc),finished_utc=prep.now(),exit_code=None if process is None else process.poll(),
                         mpi_pid=info.get('mpi_pid'),mpi_pty_evidence=info.get('mpi_pty_evidence'))
            update_status(stage/'status.json',state)
            raise # retain evidence and possible live processes; no automatic restart/cleanup
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-root',type=Path,default=prep.DEFAULT_STAGE)
    parser.add_argument('--phase',choices=['smoke','production'],required=True)
    parser.add_argument('--clearance',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(run(args.stage_root,args.phase,args.clearance))
if __name__=='__main__':main()
