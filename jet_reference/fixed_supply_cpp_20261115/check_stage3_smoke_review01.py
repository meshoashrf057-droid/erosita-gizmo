"""Independent evidence-only checker for one completed real fixed-supply CPP jet smoke.
Never launches a model. Wind budgets use source-born compound identities and the
native sink_details dtime ledger; evolved ordinary gas is not frozen or differenced.
"""
from pathlib import Path
import argparse
import csv
import datetime
import decimal
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import h5py
import numpy as np

A = Path('/home/xingdi/gizmo-runs/compat_validation/20261115_fixed_supply_cpp_port_review01')
N = Path('/home/xingdi/gizmo-runs/mw_isolated/cgm3_n3000_orbits_cpp_20261105_084436')
STAGE_PARENT = Path('/home/xingdi/gizmo-runs/erosita_xray')
BEGIN = 1.6363392421373844
UNIT_GYR = 0.9777923542981722
DURATION_GYR = .0003
SOURCE_MANIFEST_SHA = '09621c0f0d1220c08e3fb78d9c5e3abe18890f529d15de2a8b67b82367f545c4'
SOURCE_REVIEW_SHA = '914bcb86dd7232b0dc07ca308817e8bfe87a2726e04029d784a46122864ff202'
CONTRACT_SHA = '020450d2dc9a4241a223df7c55a75e78cac7af0da7ea6ac73f68e95d320e55f2'
SINK_SCHEMA_SHA = '1b44f897736b824d6e66b2f591d231f461715624a5c21a91450f7a8e555819b2'
RUNTIME_CONTRACT_SHA = '0c81dcde6b035d7ca9f7f06d153f3280ccc080af02314e1a7ac4fd4e51bdc24a'
WARNINGS = N / 'provenance/known_runtime_warnings_baseline.json'
WARNINGS_SHA = '8119e5f0533bc8d01f948550acb2e7d0fb5760e2d3f79fea79f2061f8eb711e5'
ID_FIELDS = ('ParticleIDs', 'ParticleChildIDsNumber', 'ParticleIDGenerationNumber')
WIND_ID = 1913298393
SUPPLY = .00124
CELL_MASS = 1e-7
HEADER_UNIT_MAP = {'UnitLength_in_cm':'UnitLength_In_CGS',
                   'UnitMass_in_g':'UnitMass_In_CGS',
                   'UnitVelocity_in_cm_per_s':'UnitVelocity_In_CGS'}
REQUIRED_ROLES = {'binary','config','generated_config','source_manifest','source_review','ic',
                  'ic_manifest','ic_review','params','param_manifest','contract','sink_schema',
                  'preparation_script','runner_script','plot_review'}
EXPECTED_FLAGS = {'SINK_PARTICLES','SINK_REPOSITION_ON_POTMIN','SINK_DEBUG_FIX_MDOT_MBH',
                  'SINK_WIND_SPAWN','SINK_FOLLOW_ACCRETED_ANGMOM','SINK_FOLLOW_ACCRETED_MOMENTUM',
                  'SINK_FOLLOW_ACCRETED_COM','SINK_FB_COLLIMATED','SINK_WAKEUP_GAS',
                  'SINK_INTERACT_ON_GAS_TIMESTEP','SINK_OUTPUT_MOREINFO','OUTPUT_UNSPAWNED_SINKMASS'}
FORBIDDEN_FLAGS = {'SINK_SWALLOWGAS','SINK_GRAVACCRETION','SINK_GRAVCAPTURE_GAS',
                   'SINK_GRAVCAPTURE_NONGAS','SINK_ALPHADISK_ACCRETION','SINK_RIAF_SUBEDDINGTON_MODEL',
                   'SINK_SCALE_SPAWNINGMASS_WITH_INITIALMASS','SINK_WIND_KICK','SINK_WIND_CONTINUOUS',
                   'JET_DIRECTION_FROM_KERNEL_AND_SINK'}
SINK_FIELDS = {'Sink_Mass','Sink_InitialMass','Sink_Mdot','Sink_Specific_AngMom',
               'Sink_NProgenitors','Unspawned_Wind_Mass'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def stamp(path):
    st = Path(path).stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def checked(path, digest, inputs):
    p = Path(path)
    require(p.is_absolute() and p.is_file() and not p.is_symlink()
            and p.resolve(strict=True) == p, 'Missing or redirected evidence: '+str(p))
    before = stamp(p)
    require(re.fullmatch('[0-9a-f]{64}', digest or '') and sha(p) == digest,
            'Evidence SHA differs: '+str(p))
    require(stamp(p) == before, 'Evidence changed while hashing: '+str(p))
    inputs[str(p)] = dict(sha256=digest, stamp=before)
    return p


def load(path, digest, inputs):
    return json.loads(checked(path, digest, inputs).read_text())


def visible():
    require(os.isatty(0) and os.isatty(1) and os.environ.get('TMUX_PANE'),
            'Run only in the attached visible campaign terminal')
    c = json.loads((N/'campaign.json').read_text())
    attrs = subprocess.check_output(
        ['tmux','-L',c['terminal_server'],'display-message','-p','-t',os.environ['TMUX_PANE'],
         '#{session_name} #{session_attached} #{pane_pid} #{pid}'], text=True).strip().split()
    require(len(attrs)==4 and attrs[0]==c['terminal_session'] and int(attrs[1])>0,
            'Wrong or detached campaign session')
    pid, chain = os.getpid(), []
    while pid>1:
        chain.append(pid)
        pid=int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[1])
    require(int(attrs[2]) in chain and int(attrs[3]) in chain, 'Not a descendant of the visible pane/server')
    return dict(session=attrs[0],pane=os.environ['TMUX_PANE'],tty=os.ttyname(0),ancestry=chain)


def params(path):
    result = {}
    for line in path.read_text().splitlines():
        line = re.split(r'[%#]', line, maxsplit=1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        require(len(tokens)==2 and tokens[0] not in result, 'Duplicate/malformed parameter: '+line)
        result[tokens[0]]=tokens[1]
    return result


def identities(group):
    require(all(k in group for k in ID_FIELDS), 'Missing compound identity fields')
    arrays = [group[k][...] for k in ID_FIELDS]
    require(all(x.dtype.kind in 'iu' and x.ndim==1 and (x>=0).all() for x in arrays),
            'Invalid compound identity integer array')
    matrix = np.ascontiguousarray(np.column_stack(arrays), dtype=np.uint64)
    out = matrix.view(np.dtype([('id','<u8'),('child','<u8'),('generation','<u8')])).reshape(-1)
    require(len(np.unique(out))==len(out), 'Duplicate compound particle identity')
    return out


def scan_hdf(h):
    require('Header' in h, 'Header absent')
    counts = np.asarray(h['Header'].attrs['NumPart_Total'], dtype=np.uint64)
    require(np.array_equal(counts, h['Header'].attrs['NumPart_ThisFile'])
            and int(h['Header'].attrs['NumFilesPerSnapshot'])==1, 'Incomplete/split snapshot')
    fields = 0
    for name, group in h.items():
        if not re.fullmatch(r'PartType[0-5]', name):
            continue
        n = int(counts[int(name[-1])])
        require(n>0 and all(isinstance(ds,h5py.Dataset) and ds.shape and ds.shape[0]==n
                            for ds in group.values()), 'Particle lengths do not match Header: '+name)
        identities(group)
        for field, ds in group.items():
            for start in range(0, n, 131072):
                values = ds[start:start+131072]
                if values.dtype.kind in 'fc':
                    require(np.isfinite(values).all(), 'Nonfinite field: '+name+'/'+field)
            fields += 1
        require('Masses' in group and (group['Masses'][...] > 0).all(), 'Nonpositive particle mass')
    for index, count in enumerate(counts):
        require((int(count)>0)==('PartType'+str(index) in h), 'Missing or unexpected particle group')
    for attr,value in h['Header'].attrs.items():
        arr = np.asarray(value)
        if arr.dtype.kind in 'fc':
            require(np.isfinite(arr).all(), 'Nonfinite Header attribute: '+attr)
    return fields


def text_half_ulp(token, significant_digits=6):
    value = decimal.Decimal(token)
    if value == 0:
        return 0.0
    return float(decimal.Decimal(5).scaleb(value.copy_abs().adjusted()-significant_digits))


def ledger(paths, bh_id, inputs):
    rows = []
    for path in paths:
        checked(path,sha(path),inputs)
        for number,line in enumerate(path.read_text().splitlines(),1):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            values = dict(re.findall(r'([A-Za-z_]+)=\s*([0-9.eE+-]+)',line))
            require({'time','ID','mtot','msink','sink_mdot','dtime'} <= set(values),
                    'Unrecognized native sink_details row')
            require(int(values['ID'])==bh_id, 'Unexpected second sink in details')
            for key in ['time','mtot','msink','sink_mdot','dtime']:
                require(math.isfinite(float(values[key])), 'Nonfinite sink_details value')
            dt = float(values['dtime'])
            require(dt>=0 and abs(float(values['sink_mdot']))==0, 'Nonzero natural Mdot or negative dt')
            for key in ['mtot','msink']:
                require(abs(float(values[key])-.0004) <= text_half_ulp(values[key])+1e-15,
                        'Native details show BH mass drift')
            rows.append(dict(time=float(values['time']),dt=dt,
                             dt_rounding_bound=text_half_ulp(values['dtime']),
                             path=str(path),line=number))
    rows.sort(key=lambda r:r['time'])
    require(rows and len({r['time'] for r in rows})==len(rows), 'Missing/duplicate sink supply ledger')
    require(all(BEGIN-1e-12<=r['time']<=BEGIN+DURATION_GYR/UNIT_GYR+1e-12 for r in rows),
            'Supply details lie outside smoke')
    return rows


def cumulative_supply_bounds(rows, time):
    # Snapshots written during drift can precede source terms at exactly that time.
    # Keep same-time operations as an explicit interval, never guess their ordering.
    earlier=[r for r in rows if r['time'] < time-1e-13]
    equal=[r for r in rows if abs(r['time']-time)<=1e-13]
    low=SUPPLY*sum(max(0,r['dt']-r['dt_rounding_bound']) for r in earlier)
    high=SUPPLY*sum(r['dt']+r['dt_rounding_bound'] for r in earlier+equal)
    return low,high,len(equal)


def summary(values):
    a=np.asarray(values,dtype=float)
    return dict(min=float(np.min(a)),median=float(np.median(a)),max=float(np.max(a)))


def check(stage_root, status_sha, clearance_path, clearance_sha):
    host=visible()
    R=Path(stage_root)
    require(R.is_absolute() and R.resolve(strict=True)==R and R.parent==STAGE_PARENT
            and R.name.startswith('cgm3_n3000_cpp_fixed_supply_'), 'Unapproved stage root')
    S=R/'smoke'; out=S/'smoke_review.json'; csv_out=S/'smoke_review.csv'
    require(not out.exists() and not csv_out.exists(), 'Smoke review output already exists')
    inputs={}; own_sha=sha(Path(__file__))
    st=load(S/'status.json',status_sha,inputs)
    clearance=load(clearance_path,clearance_sha,inputs)
    require(st['schema']=='fixed-supply-jet-stage-status/v1' and st['phase']=='smoke'
            and st['stage_root']==str(R) and st['state']=='complete' and st['exit_code']==0
            and st['cwd']==str(S) and st.get('finished_utc'), 'Smoke did not complete normally')
    require(clearance['schema']=='fixed-supply-jet-launch-clearance/v1'
            and clearance['status']=='approved_for_flag2_smoke'
            and clearance['phase']=='smoke' and clearance['stage_root']==str(R)
            and st['clearance']==dict(path=str(clearance_path),sha256=clearance_sha),
            'Wrong or unbound smoke launch clearance')
    require(st['mpi_ranks']==clearance['mpi_ranks']==6
            and st['omp_threads']==clearance['omp_threads']==1
            and st['nice']==clearance['nice']==19
            and st['restart_flag']==clearance['restart_flag']==2
            and st['inner_pty_required'] is True, 'Completed smoke resource or restart contract differs')
    mpi_evidence=st['mpi_pty_evidence']
    require(mpi_evidence['pid']==st['mpi_pid'] and mpi_evidence['cwd']==str(S)
            and st['transport_pid'] in mpi_evidence['ancestry']
            and all(re.fullmatch(r'/dev/pts/[0-9]+',mpi_evidence['fd_targets'].get(k) or '')
                    for k in ('1','2')), 'Actual native MPI stdout/stderr were not in the inner PTY')
    b=st['bindings']
    require(REQUIRED_ROLES<=set(b) and b==clearance['bindings'], 'Status/clearance bindings differ')
    for role,entry in b.items():
        checked(entry['path'],entry['sha256'],inputs)
    for role,name,digest in [('source_manifest','source_port_manifest.json',SOURCE_MANIFEST_SHA),
                             ('source_review','independent_source_review.json',SOURCE_REVIEW_SHA),
                             ('contract','stage3_contract.json',CONTRACT_SHA),
                             ('sink_schema','sink_input_schema.json',SINK_SCHEMA_SHA)]:
        require(b[role]['path']==str(A/name) and b[role]['sha256']==digest, 'Frozen '+role+' differs')
    source_manifest=json.loads(Path(b['source_manifest']['path']).read_text())
    for item in source_manifest['files']:
        checked(A/'source'/item['path'],item['sha256'],inputs)
    require(b['config']['path']==str(A/'source/Config.sh'), 'Wrong isolated Config')
    generated=Path(b['generated_config']['path']).read_text()
    defines=set(re.findall(r'^\s*#define\s+([A-Za-z_]\w*)',generated,re.M))
    require(EXPECTED_FLAGS<=defines and not FORBIDDEN_FLAGS.intersection(defines)
            and not any(x.startswith(('FIRE','SINGLE_STAR','SNE_NONSINK')) for x in defines),
            'Generated Config does not match reviewed fixed-supply model')
    for flag, expected in [('SINK_DEBUG_FIX_MDOT_MBH', .00124),
                           ('SINK_WIND_SPAWN', 2.), ('SINK_FOLLOW_ACCRETED_ANGMOM', 1.)]:
        values=re.findall(r'^\s*#define\s+'+flag+r'\s+([^\n]+)',generated,re.M)
        require(len(values)==1, 'Missing or duplicate valued macro: '+flag)
        value=values[0].split('//',1)[0].split('/*',1)[0].strip().strip('()')
        require(float(value)==expected, 'Generated Config macro value differs: '+flag)
    command=st['command']
    require(isinstance(command,list) and command[-1]=='2' and b['binary']['path'] in command
            and any(command[i] in ('-np','-n') and command[i+1]=='6'
                    for i in range(len(command)-1))
            and (b['params']['path'] in command or 'stage3.param' in command),
            'Actual command is not the reviewed six-rank flag2 launch')
    P=Path(b['params']['path'])
    require(P==S/'stage3.param' and Path(b['ic']['path'])==R/'inputs/jet_000.hdf5', 'Wrong smoke paths')
    p=params(P); target=float(p['TimeMax'])
    require(float(p['TimeBegin'])==BEGIN and abs((target-BEGIN)*UNIT_GYR-DURATION_GYR)<1e-13,
            'Smoke does not cover the fixed .3 physical Myr')
    declared_ic=(S/p['InitCondFile']).resolve()
    # Native GIZMO accepts an extensionless InitCondFile stem and appends .hdf5.
    if declared_ic.suffix != '.hdf5':
        declared_ic=Path(str(declared_ic)+'.hdf5')
    require(declared_ic==Path(b['ic']['path'])
            and (S/p['OutputDir']).resolve()==S/'output', 'Parameter IC/output paths differ')
    contract=json.loads(Path(b['contract']['path']).read_text())
    pm=json.loads(Path(b['param_manifest']['path']).read_text())
    require(pm['stage_root']==str(R) and pm['phases']['smoke']['params']==b['params']
            and pm['preparation_script']==b['preparation_script'], 'Parameter preparation manifest binding differs')
    runtime_path=A/'stage3_runtime_inputs_contract.json'
    runtime=load(runtime_path,RUNTIME_CONTRACT_SHA,inputs)
    require(pm['runtime_inputs_contract']==dict(path=str(runtime_path),sha256=RUNTIME_CONTRACT_SHA)
            and runtime['file_count']==50 and runtime['total_bytes']==14159271,
            'Frozen cooling runtime contract differs')
    expected_runtime={str(S/item['relative_path']):item for item in runtime['files']}
    actual_runtime=pm['phases']['smoke']['runtime_inputs']
    require(len(actual_runtime)==len(expected_runtime)
            and {item['path'] for item in actual_runtime}==set(expected_runtime),
            'Smoke runtime cooling inventory differs')
    for item in actual_runtime:
        ref=expected_runtime[item['path']]
        require(item['source']==ref['source'] and item['sha256']==ref['sha256']
                and item['bytes']==ref['bytes'] and Path(item['path']).is_relative_to(S),
                'Runtime cooling source/copy metadata differs')
        checked(item['source'],ref['sha256'],inputs)
        checked(item['path'],ref['sha256'],inputs)
    for item in contract['parameter_translations']:
        require(float(p[item['canonical']])==float(item['value']), 'Translated sink parameter differs')
    for key,value in contract['explicit_preserved_controls'].items():
        require(float(p[key])==float(value), 'Preserved n3000 control differs: '+key)
    log_path=S/'run.log'
    require(st['run_log']['path']==str(log_path), 'Completed status log path differs')
    checked(log_path,st['run_log']['sha256'],inputs)
    log=log_path.read_text(errors='replace')
    require('reaching time-limit. stopping.' not in log.lower(), 'Smoke stopped by walltime')
    markers=list(re.finditer(r'Final time=([0-9.eE+-]+) reached[.] Simulation ends[.]',log))
    syncs=list(re.finditer(r'Sync-Point\s+([0-9]+), Time:\s*([0-9.eE+-]+)',log))
    require(markers and syncs and markers[-1].group(1)==format(target,'.6g')
            and abs(float(syncs[-1].group(2))-target)<1e-12
            and syncs[-1].end()<markers[-1].start(), 'Normal exact final Sync/marker absent')
    allocator=[float(x) for x in re.findall(r'presently allocated=([0-9.]+) MB',log)]
    require(allocator and all(math.isfinite(x) for x in allocator)
            and max(allocator)<float(p['MaxMemSize']), 'Missing/excessive internal allocator peak')
    baseline=load(WARNINGS,WARNINGS_SHA,inputs)
    old_tags={re.search(r"Tag '([^']+)'",line).group(1) for line in baseline['ignored_parameter_lines']}
    allowed_tags={x for x in old_tags if not x.startswith(('BH_','BAL_','BlackHole','SeedBlackHole'))}
    warning_tags=set(re.findall(r"Tag '([^']+)'[^\n]*is being ignored",log))
    require(warning_tags<=allowed_tags, 'Unexpected/ignored sink parameters: '+str(warning_tags-allowed_tags))
    rp=S/'resource_samples.jsonl'
    require(st['resource_samples']['path']==str(rp), 'Completed status resource path differs')
    checked(rp,st['resource_samples']['sha256'],inputs)
    resource_rows=[json.loads(line) for line in rp.read_text().splitlines() if line.strip()]
    six=[]
    for row in resource_rows:
        ranks=[q for q in row.get('processes',[]) if q.get('cwd')==str(S)
               and (q.get('name','').startswith('GIZMO') or q.get('comm','').startswith('GIZMO'))]
        if len(ranks)!=6:
            continue
        require(len({q['pid'] for q in ranks})==6
                and row.get('mpi_pid')==st['mpi_pid']
                and all(st['mpi_pid'] in q.get('ancestry',[]) for q in ranks)
                and all(q.get('command') and q['command'][0]==b['binary']['path'] for q in ranks)
                and all(int(q.get('nice',q.get('ni',-999)))==19 for q in ranks),
                'Actual six-rank command/ancestry/nice/PID evidence differs')
        require(all(str(q.get('omp_threads',q.get('environ',{}).get('OMP_NUM_THREADS','missing')))=='1'
                    for q in ranks), 'OMP1 not evidenced for every actual rank')
        six.append(dict(time=row.get('utc',row.get('checked_utc')),
                        pids=[q['pid'] for q in ranks],
                        aggregate_rss_kib=sum(int(q.get('rss_kib',0)) for q in ranks)))
    require(six and max(q['aggregate_rss_kib'] for q in six)>0, 'No six actual rank/RSS samples')
    ic=Path(b['ic']['path']); original_stars={}
    with h5py.File(ic,'r') as h:
        scan_hdf(h)
        require(abs(float(h['Header'].attrs['Time'])-BEGIN)<1e-12, 'Jet IC time differs')
        native_header_units={native:float(h['Header'].attrs[native]) for native in HEADER_UNIT_MAP.values()}
        raw_units={key:native_header_units[native] for key,native in HEADER_UNIT_MAP.items()}
        require(all(raw_units[k]==float(contract['units'][k]) for k in HEADER_UNIT_MAP),
                'Native Header units differ from the reviewed physical-unit contract')
        field_inventory={name:set(group) for name,group in h.items() if name.startswith('PartType')}
        for name in ['PartType2','PartType3','PartType4']:
            if name in h:
                original_stars[name]=identities(h[name])
        require('PartType5' in h and len(h['PartType5/ParticleIDs'])==1
                and SINK_FIELDS<=set(h['PartType5']), 'Missing sole-BH native input state')
        bh=h['PartType5'];bh_id=int(bh['ParticleIDs'][0]);bh_child=int(bh[ID_FIELDS[1]][0])
        bh_generation=int(bh[ID_FIELDS[2]][0])
        require(bh_id==90000000 and all(float(bh[k][0])==.0004 for k in ['Masses','Sink_Mass','Sink_InitialMass'])
                and float(bh['Unspawned_Wind_Mass'][0])==float(bh['Sink_Mdot'][0])==0
                and np.array_equal(bh['Sink_Specific_AngMom'][0],[0,0,1]), 'Jet IC BH state differs')
        require(all(not np.any(group['ParticleIDs'][...]==WIND_ID)
                    for name,group in h.items() if name.startswith('PartType')), 'Pre-existing wind tag')
    details=sorted((S/'output/sink_details').glob('sink_details_*.txt'))
    require(len(details)==6, 'Six native sink_details files required')
    supply_rows=ledger(details,bh_id,inputs)
    snapshots=[]
    for path in (S/'output').glob('snapshot_*.hdf5'):
        require(re.fullmatch(r'snapshot_[0-9]+\.hdf5',path.name), 'Unexpected split snapshot filename')
        with h5py.File(path,'r') as h:
            snapshots.append((float(h['Header'].attrs['Time']),path))
    snapshots.sort()
    require(len(snapshots)>=2 and abs(snapshots[-1][0]-target)<1e-12
            and all(BEGIN-1e-12<=t<=target+1e-12 for t,_ in snapshots)
            and all(a[0]<z[0] for a,z in zip(snapshots,snapshots[1:])), 'Incomplete/nonmonotone smoke sequence')
    observed_union=set();frames=[];wind_evidence=[];last_count=0
    for time,path in snapshots:
        checked(path,sha(path),inputs)
        with h5py.File(path,'r') as h:
            fields=scan_hdf(h)
            require(all(float(h['Header'].attrs[k])==v for k,v in native_header_units.items()), 'Native Header units changed')
            require(set(field_inventory)=={k for k in h if k.startswith('PartType')}
                    and all(field_inventory[k]<=set(h[k]) for k in field_inventory), 'Snapshot fields missing')
            for name,base_ids in original_stars.items():
                actual=identities(h[name])
                require(len(np.intersect1d(base_ids,actual,assume_unique=True))==len(base_ids),
                        'Original stellar compound identities were lost: '+name)
            bh=h['PartType5']
            require(len(bh['ParticleIDs'])==1 and int(bh['ParticleIDs'][0])==bh_id
                    and int(bh[ID_FIELDS[1]][0])==bh_child, 'BH ID/child changed or second BH appeared')
            for key in ['Masses','Sink_Mass','Sink_InitialMass']:
                require(abs(float(bh[key][0])-.0004)<1e-15, 'BH mass drift: '+key)
            require(float(bh['Sink_Mdot'][0])==0 and int(bh['Sink_NProgenitors'][0])==1,
                    'Natural accretion/merger state changed')
            count=int(bh[ID_FIELDS[2]][0])-bh_generation
            require(count>=last_count and count%2==0, 'BH spawn-generation counter is invalid')
            last_count=count
            gas=h['PartType0'];mask=gas['ParticleIDs'][...]==WIND_ID
            wind_ids=identities(gas)[mask]
            found={tuple(int(x) for x in item) for item in wind_ids.tolist()}
            expected={(WIND_ID,bh_child+bh_generation+i,bh_id) for i in range(count)}
            require(found==expected, 'Not all source-born wind identities survive: inspect native merge/promotion evidence')
            observed_union.update(found)
            masses=gas['Masses'][...][mask]
            require(np.allclose(masses,CELL_MASS,rtol=0,atol=1e-18), 'Wind masses changed/merged; budget needs explicit lineage')
            bucket=float(bh['Unspawned_Wind_Mass'][0])
            require(math.isfinite(bucket) and bucket>=0, 'Invalid wind reservoir')
            mass_budget=count*CELL_MASS+bucket
            low,high,equal=cumulative_supply_bounds(supply_rows,time)
            rounding_fence=1e-13*max(SUPPLY*(target-BEGIN),CELL_MASS)+1e-18
            require(low-rounding_fence<=mass_budget<=high+rounding_fence,
                    'Unique-spawn plus reservoir mass is inconsistent with actual native supply ledger')
            frame=dict(path=str(path),sha256=inputs[str(path)]['sha256'],time_code=time,
                       elapsed_physical_Myr=(time-BEGIN)*UNIT_GYR*1000,finite_dataset_count=fields,
                       wind_count=count,spawned_mass_code=count*CELL_MASS,unspawned_mass_code=bucket,
                       mass_budget_code=mass_budget,supply_lower_code=low,supply_upper_code=high,
                       same_time_source_operations_with_explicit_ordering_interval=equal,
                       bh_generation=int(bh[ID_FIELDS[2]][0]))
            if count:
                axis=np.asarray(bh['Sink_Specific_AngMom'][0],dtype=float)
                require(np.isfinite(axis).all() and np.linalg.norm(axis)>0, 'Invalid actual sink axis')
                axis/=np.linalg.norm(axis)
                dv=np.asarray(gas['Velocities'][...][mask],float)-np.asarray(bh['Velocities'][0],float)
                speed=np.linalg.norm(dv,axis=1)*raw_units['UnitVelocity_in_cm_per_s']/1e5
                projection=dv@axis
                require((speed>0).all() and (projection>0).any() and (projection<0).any(),
                        'Actual HDF5 wind sample lacks two nonzero opposing velocity lobes')
                u=np.asarray(gas['InternalEnergy'][...][mask],float)
                require((u>0).all(), 'Nonpositive actual wind internal energy')
                proxy=u*.59*(2./3.)*(raw_units['UnitVelocity_in_cm_per_s']**2)*1.67262192369e-24/1.380649e-16
                evidence=dict(path=str(path),time_code=time,wind_count=count,axis=axis.tolist(),
                              compound_identities=sorted(found),mass_code=masses.tolist(),
                              relative_speed_kms=summary(speed),absolute_axis_cosine=summary(np.abs(projection)/np.linalg.norm(dv,axis=1)),
                              positive_lobe_count=int(np.sum(projection>0)),negative_lobe_count=int(np.sum(projection<0)),
                              internal_energy_code=summary(u),fully_ionized_primordial_temperature_proxy_K=summary(proxy))
                if 'Temperature' in gas:
                    evidence['native_temperature_K']=summary(gas['Temperature'][...][mask])
                wind_evidence.append(evidence)
            frames.append(frame)
    require(last_count>=2 and len(observed_union)==last_count and wind_evidence, 'No actual paired spawn observed')
    events=re.findall(r'Task\s+([0-9]+) wants to create ([0-9.eE+-]+) mass in wind with ([0-9]+) new particles each of mass ([0-9.eE+-]+)',log)
    require(events and all(int(n)>0 and int(n)%2==0 and abs(float(m)-CELL_MASS)<=text_half_ulp(m)+1e-18
                           for _,_,n,m in events)
            and sum(int(n) for _,_,n,_ in events)==last_count,
            'Native pair-spawn console events do not match HDF5 BH counter and identities')
    require(all(stamp(Path(path))==tuple(info['stamp']) for path,info in inputs.items())
            and sha(Path(__file__))==own_sha, 'Evidence or checker changed during review')
    report=dict(schema='fixed-supply-jet-smoke-independent-review/v1',status='pass',
                checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                reviewer_source=str(Path(__file__)),reviewer_source_sha256=own_sha,stage_root=str(R),
                visible_host=host,actual_mpi_pty_evidence=mpi_evidence,
                mpi_rank_stdio_note='Native MPI stdout/stderr have a PTY; individual OpenMPI ranks may use normal output-forwarding pipes.',
                status_path=str(S/'status.json'),status_sha256=status_sha,
                launch_clearance=str(clearance_path),launch_clearance_sha256=clearance_sha,
                log_path=str(log_path),log_sha256=inputs[str(log_path)]['sha256'],
                input_ic=str(ic),input_ic_sha256=b['ic']['sha256'],
                params=str(P),params_sha256=b['params']['sha256'],bindings=b,
                simulation_returncode=0,restart_flag=2,begin_time_code=BEGIN,end_time_code=target,
                duration_physical_Myr=.3,native_header_units=native_header_units,
                normalized_physical_unit_inputs=raw_units,header_unit_name_mapping=HEADER_UNIT_MAP,
                normal_final_marker=markers[-1].group(1),
                exact_final_sync=float(syncs[-1].group(2)),final_snapshot=frames[-1],
                frames=frames,peak_reported_allocator_MiB=max(allocator),
                observed_six_rank_samples=len(six),peak_aggregate_rank_RSS_KiB=max(q['aggregate_rss_kib'] for q in six),
                original_star_counts={k:len(v) for k,v in original_stars.items()},original_stars_preserved=True,
                all_output_numeric_fields_finite=True,bh_dynamical_and_subgrid_mass_constant=True,
                sink_mdot_zero_expected=True,wind_tag=WIND_ID,spawned_particle_count=last_count,
                observed_unique_spawn_identities=sorted(observed_union),
                native_pair_spawn_events=[dict(task=int(task),recorded_reservoir_mass=mass,
                                               count=int(n),recorded_cell_mass=m) for task,mass,n,m in events],
                source_ledger=dict(files=[str(p) for p in details],operations=len(supply_rows),
                                   dtime_sum_code=sum(r['dt'] for r in supply_rows),
                                   dtime_rounding_bound_code=sum(r['dt_rounding_bound'] for r in supply_rows),
                                   nominal_duration_code=target-BEGIN,
                                   ledger_coverage_difference_code=sum(r['dt'] for r in supply_rows)-(target-BEGIN),
                                   rule='Use actual native dtime entries with decimal %g rounding bounds. Same-time drift snapshot/source ordering is bracketed explicitly.'),
                wind_kinematic_and_thermal_evidence=wind_evidence,
                configured_birth_speed_kms=10000.,configured_birth_temperature_K=1e8,
                evolved_wind_values_are_not_required_to_equal_birth_values=True,
                ordinary_gas_fields_are_allowed_to_evolve=True,total_gas_mass_difference_is_not_a_wind_budget=True,
                budget_scope='All source-born wind compound identities must persist with their actual masses through this short smoke. Missing/merged/promoted lineage fails closed; no guessed total-gas correction.',
                production_still_requires_independent_clearance=True,input_evidence=inputs)
    columns=['path','time_code','elapsed_physical_Myr','wind_count','spawned_mass_code','unspawned_mass_code',
             'mass_budget_code','supply_lower_code','supply_upper_code','finite_dataset_count','bh_generation']
    with csv_out.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        writer.writerows({k:r[k] for k in columns} for r in frames)
    with out.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status='pass',report=str(out),spawned_particle_count=last_count,
                          final_time=target,peak_reported_allocator_MiB=max(allocator)),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-root',required=True)
    parser.add_argument('--status-sha',required=True)
    parser.add_argument('--clearance',required=True)
    parser.add_argument('--clearance-sha',required=True)
    args=parser.parse_args()
    with (N/'status/compute.flock').open('r+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check(args.stage_root,args.status_sha,Path(args.clearance),args.clearance_sha)
