#!/usr/bin/env python3
"""Independent static review of one owned full-endpoint new fixed-supply jet IC.
Source HDF5s are opened read-only. This checker never initializes fields, rotates
particles, edits Header, generates an IC, launches GIZMO, or enables a jet.
"""
from pathlib import Path
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import uuid
import h5py
import numpy as np

BASE=Path('/home/xingdi/gizmo-runs')
N=BASE/'mw_isolated/cgm3_n3000_orbits_cpp_20261105_084436'
A=BASE/'compat_validation/20261115_fixed_supply_cpp_port_review01'
SOURCE=N/'local02_lmc_y_p05/production/output/snapshot_011.hdf5'
SOURCE_SHA='67520602f3b2a591299814b89d348bcd3069e26066c42941a99824adf3fce95b'
ACCEPTED=N/'accepted_orbit.json'
ACCEPTED_SHA='bc71790c089a7b7afcd38dfd0a76275908fc9b512a7b9ba913d03fce80c4a3a0'
ACCEPTED_REVIEW=N/'provenance/local02_accepted_independent_review01.json'
ACCEPTED_REVIEW_SHA='79376650beda29c07d7893c1f7079075da33a9388fc82072f757289e86ee495c'
GATE=N/'local02_lmc_y_p05/endpoint_review/gate_result.json'
GATE_SHA='b6e35c2d77540703d639ab850185fdb1a9d3e63b2bf4b044377e8b5b089b0dec'
SCHEMA=A/'sink_input_schema.json'
SCHEMA_SHA='1b44f897736b824d6e66b2f591d231f461715624a5c21a91450f7a8e555819b2'
CONTRACT=A/'stage3_contract.json'
CONTRACT_SHA='020450d2dc9a4241a223df7c55a75e78cac7af0da7ea6ac73f68e95d320e55f2'
PORT=A/'source_port_manifest.json'
PORT_SHA='09621c0f0d1220c08e3fb78d9c5e3abe18890f529d15de2a8b67b82367f545c4'
BUILD=A/'build_review01/complete.json'
AZAXIS=BASE/'erosita_xray/azaxis.py'
AZAXIS_SHA='a801866391d225b042fecbec5b0e2d0e1cf03159e5a09629dc951f1cdeff0459'
TARGET=1.6363392421373844
DEFAULT_STAGE=BASE/'erosita_xray/cgm3_n3000_cpp_fixed_supply_20261115'
EXTRA_NAMES={'Sink_Mass','Sink_InitialMass','Sink_Mdot','Sink_Specific_AngMom',
             'Sink_NProgenitors','Unspawned_Wind_Mass'}
CHUNK_ROWS=100000

def require(c,m):
    if not c:
        raise RuntimeError(m)

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def stamp(path):
    p=Path(path)
    st=p.stat()
    return dict(path=str(p),bytes=st.st_size,device=st.st_dev,inode=st.st_ino,
                hard_links=st.st_nlink,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)

def file_identity(path):
    p=Path(path)
    require(p.resolve(strict=True)==p and not p.is_symlink() and stat.S_ISREG(p.stat().st_mode),
            'Not an absolute regular nonredirected file: '+str(p))
    return dict(**stamp(p),sha256=sha(p))

def write_new(path,data):
    tmp=path.with_name('.pending-'+uuid.uuid4().hex+'-'+path.name)
    with tmp.open('x') as f:
        json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    os.link(tmp,path);tmp.unlink()

def visible_guard():
    require(os.isatty(0) and os.isatty(1) and os.environ.get('TMUX_PANE'),
            'Run the real IC review only in the attached visible Ubuntu PTY')
    c=read(N/'campaign.json')
    attrs=subprocess.check_output(
        ['tmux','-L',c['terminal_server'],'display-message','-p','-t',os.environ['TMUX_PANE'],
         '#{session_name} #{session_attached} #{pane_pid} #{pid}'],text=True).strip().split()
    require(len(attrs)==4 and attrs[0]==c['terminal_session'] and int(attrs[1])>0,
            'Wrong or detached campaign session')
    ancestry=[];pid=os.getpid()
    while pid>1:
        ancestry.append(pid)
        pid=int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[1])
    require(int(attrs[2]) in ancestry and int(attrs[3]) in ancestry,
            'Review is not descended from the attached pane/server')
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            name=(proc/'comm').read_text().strip()
            require(not name.startswith('GIZMO') and name not in {'mpirun','mpiexec','orterun','prterun'},
                    'Existing simulation/MPI PID '+proc.name)
        except (FileNotFoundError,ProcessLookupError):
            continue
    return dict(session=attrs[0],pane=os.environ['TMUX_PANE'],tty=os.ttyname(0),
                pane_pid=int(attrs[2]),server_pid=int(attrs[3]),ancestry=ancestry)

def array_equal(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype:
        return False
    # Compare NaNs as equal here, then independently require finite numeric data.
    if a.dtype.kind in 'fc':
        return bool(np.array_equal(a,b,equal_nan=True))
    return bool(np.array_equal(a,b))

def finite(a,label):
    a=np.asarray(a)
    if a.dtype.kind in 'biufc':
        require(bool(np.isfinite(a).all()),'Nonfinite numeric value: '+label)

def attrs_equal(old,new,path):
    require(set(old.attrs)==set(new.attrs),'Attribute-name set differs: '+path)
    results=[]
    for name in sorted(old.attrs):
        a,b=old.attrs[name],new.attrs[name]
        da,db=old.attrs.get_id(name).dtype,new.attrs.get_id(name).dtype
        require(da==db and array_equal(a,b),'Attribute dtype/shape/value differs: '+path+'@'+name)
        finite(a,path+'@'+name+' source');finite(b,path+'@'+name+' candidate')
        results.append(dict(name=name,dtype=str(da),shape=list(np.asarray(a).shape),exact_equal=True,finite=True))
    return results

def object_inventory(f):
    objects={'/':f}
    datasets={}
    def visit(group,prefix):
        for name in sorted(group):
            link=group.get(name,getlink=True)
            require(isinstance(link,h5py.HardLink),'External or soft HDF5 link: '+prefix+name)
            obj=group[name];key=prefix+name
            objects[key]=obj
            if isinstance(obj,h5py.Group):
                visit(obj,key+'/')
            elif isinstance(obj,h5py.Dataset):
                require(not obj.is_virtual and obj.external is None,'Virtual/external dataset: '+key)
                datasets[key]=obj
            else:
                raise RuntimeError('Unsupported HDF5 object: '+key)
    visit(f,'')
    return objects,datasets

def preserved_dataset(old,new,path):
    require(old.shape==new.shape and old.dtype==new.dtype and old.dtype.str==new.dtype.str,
            'Original dataset shape/dtype changed: '+path)
    require(old.dtype.kind in 'biufc','Unexpected nonnumeric original dataset: '+path)
    require(old.chunks==new.chunks and old.compression==new.compression
            and old.compression_opts==new.compression_opts and old.shuffle==new.shuffle
            and old.fletcher32==new.fletcher32 and old.scaleoffset==new.scaleoffset,
            'Original dataset storage metadata changed: '+path)
    first=hashlib.sha256();second=hashlib.sha256();values=0
    slices=[()] if old.ndim==0 else [slice(i,min(i+CHUNK_ROWS,old.shape[0])) for i in range(0,old.shape[0],CHUNK_ROWS)]
    for selection in slices:
        a,b=old[selection],new[selection]
        require(array_equal(a,b),'Original dataset value differs: '+path)
        finite(a,path+' source');finite(b,path+' candidate')
        first.update(np.ascontiguousarray(a).tobytes());second.update(np.ascontiguousarray(b).tobytes())
        values+=np.asarray(a).size
    require(first.digest()==second.digest(),'Original raw numeric bytes differ: '+path)
    return dict(path=path,shape=list(old.shape),dtype=str(old.dtype),values=int(values),
                exact_equal=True,finite=True,raw_data_sha256=first.hexdigest())


def generated_schema_review(pinned):
    """After the isolated build: match actual generated flags to frozen native readers."""
    require(file_identity(PORT)['sha256']==PORT_SHA,'Reviewed source port manifest changed')
    port=read(PORT)
    expected={r['path']:r['sha256'] for r in port['files']}
    source_root=A/'source'
    checked={}
    for relative in ['Config.sh','declarations/typedefs.h','declarations/precompiler_logic.h',
                     'file_io/read_ic.cc','file_io/io.cc']:
        path=source_root/relative
        record=file_identity(path)
        require(record['sha256']==expected[relative],'Reviewed native reader/config source changed: '+relative)
        checked[str(path)]=record
    build_record=file_identity(BUILD);build=read(BUILD)
    require(build['status']=='build_pass_only' and build['exit_code']==0
            and build['runtime_validated'] is False and build['smoke_validated'] is False,
            'Independent isolated build not complete or state differs')
    paths={'config':source_root/'Config.sh','generated_config':source_root/'GIZMO_config.h',
           'binary':source_root/'GIZMO_fixed_supply_cpp'}
    products={}
    for role,path in paths.items():
        actual=file_identity(path)
        recorded=build[role]
        require(recorded['path']==str(path) and recorded['sha256']==actual['sha256']
                and recorded['bytes']==actual['bytes'],'Build product binding differs: '+role)
        products[role]=actual
    header=paths['generated_config'].read_text()
    require(build['generated_config_text']==header,'Generated macro text differs from actual build')
    macros={}
    for match in re.finditer(r'^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)(?:[ \t]+([^\n]*))?$',header,re.M):
        macros[match[1]]=re.sub(r'/\*.*?\*/|//.*','',match[2] or '').strip()
    names=['SINK_PARTICLES','SINK_WIND_SPAWN','SINK_FOLLOW_ACCRETED_ANGMOM',
           'SINK_OUTPUT_MOREINFO','OUTPUT_UNSPAWNED_SINKMASS',
           'INPUT_IN_DOUBLEPRECISION','OUTPUT_IN_DOUBLEPRECISION','SINK_DEBUG_FIX_MDOT_MBH']
    require(all(name in macros for name in names),'Generated build lacks required sink/precision flags')
    require(macros['SINK_WIND_SPAWN'].strip('() ' )=='2'
            and macros['SINK_FOLLOW_ACCRETED_ANGMOM'].strip('() ' )=='1'
            and float(macros['SINK_DEBUG_FIX_MDOT_MBH'].strip('() '))==.00124,
            'Generated sink mode/axis evolution/fixed supply differs')
    # SINK_COUNTPROGS is implied by SINK_PARTICLES in the separately pinned native
    # precompiler source; it is not an explicit Config.sh-generated macro.
    text=(source_root/'declarations/precompiler_logic.h').read_text()
    require(re.search(r'#ifdef\s+SINK_PARTICLES\s*\n#define\s+SINK_COUNTPROGS\b',text),
            'Reviewed native progenitor counter implication differs')
    checked.update({str(PORT):file_identity(PORT),str(BUILD):build_record})
    checked.update({r['path']:r for r in products.values()})
    pinned.update(checked)
    return dict(status='pass_static_generated_config_and_native_source_binding',
                build_manifest=build_record,products=products,required_macros={k:macros[k] for k in names},
                native_reader_source=checked,
                input_float_dtype='float64',output_float_dtype='float64',
                progenitor_dtype='int32',progenitor_flag='SINK_COUNTPROGS implied by SINK_PARTICLES',
                runtime_initialization_or_spawning_claimed=False)

def check(args,host):
    source,candidate,manifest=map(Path,(args.source,args.candidate,args.manifest))
    require(source==SOURCE,'Source must be the original accepted true endpoint')
    stage=candidate.parent.parent
    require(stage.parent==BASE/'erosita_xray' and stage.resolve(strict=True)==stage
            and candidate==stage/'inputs/jet_000.hdf5'
            and manifest==stage/'preparation/ic_initialization.json',
            'Candidate/manifest must be in one new owned stage root under erosita_xray')
    src_before=file_identity(source);dst_before=file_identity(candidate);manifest_before=file_identity(manifest)
    require(src_before['sha256']==SOURCE_SHA,'Accepted source hash differs')
    require(dst_before['hard_links']==1 and (src_before['device'],src_before['inode'])
            !=(dst_before['device'],dst_before['inode']),'Candidate is a hardlink or aliases protected source')
    pins={ACCEPTED:ACCEPTED_SHA,ACCEPTED_REVIEW:ACCEPTED_REVIEW_SHA,GATE:GATE_SHA,
          SCHEMA:SCHEMA_SHA,CONTRACT:CONTRACT_SHA,AZAXIS:AZAXIS_SHA}
    checker=Path(__file__).resolve()
    checker_before=file_identity(checker)
    pinned={str(p):file_identity(p) for p in pins}
    require(all(pinned[str(p)]['sha256']==h for p,h in pins.items()),'Frozen gate/schema/contract/azaxis changed')
    compiled_schema=generated_schema_review(pinned)
    accepted,review,gate,schema,contract=map(read,(ACCEPTED,ACCEPTED_REVIEW,GATE,SCHEMA,CONTRACT))
    endpoint=dict(path=str(SOURCE),sha256=SOURCE_SHA,time_code=TARGET)
    require(accepted['trial']=='local02_lmc_y_p05' and accepted['passed_component_count']==12
            and accepted['endpoint']==review['endpoint_snapshot']==gate['endpoint_snapshot']==endpoint
            and review['status']=='pass' and review['passed_components']==12
            and gate['state']=='pass' and gate['passed_component_count']==12 and gate['threshold']==3,
            'Not the same actually accepted endpoint')
    require(schema['schema']=='isolated-fixed-supply-cpp-sink-input-schema/v1'
            and schema['group']=='PartType5' and schema['N_BH']==1
            and schema['input_endpoint']==str(SOURCE) and schema['input_endpoint_sha256_from_gate']==SOURCE_SHA
            and schema['retain_existing_datasets_and_header'] is True
            and contract['axis_correction']['tool']==str(AZAXIS)
            and contract['axis_correction']['target_format']=='cpp'
            and contract['axis_correction']['code_vector']==[0.,0.,1.]
            and contract['timeline']['TimeBegin']==TARGET,
            'Sink schema/axis/stage epoch differs')
    m=read(manifest)
    required=dict(schema='fixed-supply-jet-ic-initialization/v1',source=str(SOURCE),source_sha256=SOURCE_SHA,
                  source_sha256_before=SOURCE_SHA,source_sha256_after=SOURCE_SHA,output=str(candidate),
                  output_sha256=dst_before['sha256'],source_dataset_count=56,output_dataset_count=62,
                  azaxis_tool=str(AZAXIS),azaxis_tool_sha256=AZAXIS_SHA,
                  sink_schema_sha256=SCHEMA_SHA,contract_sha256=CONTRACT_SHA,input_endpoint_time=TARGET)
    require(all(m.get(k)==v for k,v in required.items()),'Preparation manifest identity/epoch/count/pins differ')
    definitions={d['name']:d for d in schema['additional_fields']}
    require(set(definitions)==EXTRA_NAMES and len(schema['additional_fields'])==6,'Not exactly the six audited sink inputs')
    extra={'PartType5/'+name for name in definitions}
    require(len(m['added_fields'])==6 and set(m['added_fields'])==extra,'Manifest added field set differs')
    az_report=Path(m['azaxis_report'])
    require(az_report.parent==stage/'preparation','Azaxis report outside new stage')
    az_record=file_identity(az_report)
    require(m['azaxis_report_sha256']==az_record['sha256'],'Azaxis report hash differs')
    z=read(az_report)
    require(z['path']==str(candidate) and z['target_format']=='cpp' and z['field']=='Sink_Specific_AngMom'
            and z['count']==1 and z['specific_angmom_vector']==[0.,0.,1.] and z['previous_values']=={}
            and z['input_sha256_before']==SOURCE_SHA and z['in_place'] is True
            and isinstance(z['output_sha256'],str) and len(z['output_sha256'])==64,
            'Original azaxis invocation report differs')
    changes=[];attributes={};sink_checks=[]
    with h5py.File(source,'r') as old,h5py.File(candidate,'r') as new:
        old_objects,old_data=object_inventory(old);new_objects,new_data=object_inventory(new)
        require(len(old_data)==56 and len(new_data)==62 and set(new_data)==set(old_data)|extra
                and not set(old_data)&extra and set(new_objects)==set(old_objects)|extra,
                'Original 56 objects not completely copied, or extra/non-audited objects added')
        inventory=[dict(path=k,shape=list(d.shape),dtype=str(d.dtype)) for k,d in sorted(old_data.items())]
        require(m['source_inventory']==inventory,'Manifest original dataset inventory differs')
        for path,obj in sorted(old_objects.items()):
            other=new_objects[path]
            require(type(obj) is type(other),'Original HDF5 object type changed: '+path)
            attributes[path]=attrs_equal(obj,other,path)
        for path,dataset in sorted(old_data.items()):
            changes.append(preserved_dataset(dataset,new_data[path],path))
        ho,hn=old['Header'].attrs,new['Header'].attrs
        require(float(ho['Time'])==float(hn['Time'])==TARGET,'Original Header Time was shifted')
        counts=np.asarray(ho['NumPart_Total'])
        require(np.array_equal(counts,ho['NumPart_ThisFile']) and int(counts[5])==1,'Source count inconsistency')
        for key in old:
            if key.startswith('PartType'):
                pt=int(key[len('PartType'):])
                require(len(old[key+'/ParticleIDs'])==len(new[key+'/ParticleIDs'])==int(counts[pt]),
                        'Particle count changed: '+key)
        require(np.array_equal(new['PartType5/ParticleIDs'][:],schema['existing_BH_ParticleIDs'])
                and np.array_equal(new['PartType5/Masses'][:],schema['existing_BH_Masses_code']),
                'Existing sole BH identity/dynamical mass changed')
        for name,d in sorted(definitions.items()):
            path='PartType5/'+name;dataset=new[path]
            shape=tuple(1 if x=='N_BH' else int(x) for x in d['shape'])
            value=d.get('initial_value_code',d.get('initial_value'))
            expected=np.full(shape,value,dtype=np.dtype(d['dtype'])) if np.asarray(value).ndim==0 else np.broadcast_to(np.asarray(value,dtype=np.dtype(d['dtype'])),shape)
            require(dataset.shape==shape and dataset.dtype==np.dtype(d['dtype'])
                    and array_equal(dataset[:],expected) and not len(dataset.attrs),
                    'New sink shape/dtype/value/attributes differ: '+path)
            finite(dataset[:],path)
            sink_checks.append(dict(path=path,shape=list(shape),dtype=str(dataset.dtype),
                                    value=dataset[:].tolist(),exact_expected_value=True,finite=True))
        u=float(ho['UnitLength_In_CGS'])/float(ho['UnitVelocity_In_CGS'])/3.15576e16
        ages_old=(TARGET-old['PartType4/StellarFormationTime'][:])*u
        ages_new=(TARGET-new['PartType4/StellarFormationTime'][:])*u
        require(array_equal(ages_old,ages_new),'Original stellar physical ages changed')
        finite(ages_old,'source stellar ages');finite(ages_new,'candidate stellar ages')
        ages=dict(count=int(ages_old.size),unit='physical Gyr',exact_equal=True,
                  minimum=float(ages_old.min()),maximum=float(ages_old.max()))
    require(stamp(source)=={k:v for k,v in src_before.items() if k!='sha256'}
            and stamp(candidate)=={k:v for k,v in dst_before.items() if k!='sha256'}
            and file_identity(source)==src_before and file_identity(candidate)==dst_before
            and file_identity(manifest)==manifest_before,'IC/manifest changed during review')
    require(all(file_identity(Path(p))==record for p,record in pinned.items())
            and file_identity(az_report)==az_record and file_identity(checker)==checker_before,
            'Pinned source evidence/checker changed during review')
    return dict(schema='fixed-supply-jet-ic-independent-review/v1',status='pass',checked_utc=utc(),
                source=str(source),source_sha256=SOURCE_SHA,candidate=str(candidate),
                candidate_sha256=dst_before['sha256'],manifest=str(manifest),manifest_sha256=manifest_before['sha256'],
                source_identity_before=src_before,source_identity_after=file_identity(source),
                candidate_identity_before=dst_before,candidate_identity_after=file_identity(candidate),
                source_dataset_count=56,candidate_dataset_count=62,added_fields=sorted(extra),
                original_dataset_checks=changes,original_object_attributes=attributes,new_sink_checks=sink_checks,
                all_original_56_datasets_exact_equal=True,all_original_values_finite=True,
                all_original_attributes_and_header_exact_equal=True,all_phase_space_unchanged=True,
                particle_composite_identity_unchanged=True,all_masses_and_metals_unchanged=True,
                original_stellar_ages=ages,axis_code_vector=[0.,0.,1.],
                no_particle_rotation_or_header_epoch_shift=True,owned_independent_copy=True,
                azaxis_report=az_record,pinned_evidence=pinned,compiled_config_schema_review=compiled_schema,visible_host=host,
                checker=str(checker),checker_sha256=checker_before['sha256'],
                runtime_or_smoke_validated=False,
                scope='Static new-stage IC equivalence only. No GIZMO initialization, spawning, mass reservoir behavior, or production clearance claimed.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default=str(SOURCE))
    p.add_argument('--candidate',default=str(DEFAULT_STAGE/'inputs/jet_000.hdf5'))
    p.add_argument('--manifest',default=str(DEFAULT_STAGE/'preparation/ic_initialization.json'))
    p.add_argument('--report',default=str(DEFAULT_STAGE/'preparation/ic_independent_review01.json'))
    args=p.parse_args()
    # Refuse redirected paths or existing reports before any output operation.
    report=Path(args.report)
    candidate=Path(args.candidate)
    stage=candidate.parent.parent
    require(report.is_absolute() and report.parent==stage/'preparation' and report.parent.resolve(strict=True)==report.parent
            and report.suffix=='.json' and not report.exists() and not report.is_symlink(),
            'Report must be a fresh JSON in the same owned preparation directory')
    with (N/'status/compute.flock').open('r+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        host=visible_guard()
        try:
            result=check(args,host)
        except BaseException as exc:
            failure=dict(schema='fixed-supply-jet-ic-independent-review/v1',status='fail',checked_utc=utc(),
                         source=args.source,candidate=args.candidate,manifest=args.manifest,error=repr(exc),
                         checker=str(Path(__file__).resolve()),checker_sha256=sha(Path(__file__)),visible_host=host,
                         runtime_or_smoke_validated=False)
            for key,path in [('source',args.source),('candidate',args.candidate),('manifest',args.manifest)]:
                try:failure[key+'_sha256']=sha(Path(path))
                except OSError:failure[key+'_sha256']=None
            write_new(report,failure)
            raise
        write_new(report,result)
        print(json.dumps({k:result[k] for k in ['status','source','source_sha256','candidate','candidate_sha256',
                                              'source_dataset_count','candidate_dataset_count']},indent=2),flush=True)
        print('STAGE3_NEW_IC_INDEPENDENT_STATIC_REVIEW_PASSED',report,flush=True)

if __name__=='__main__':
    main()
