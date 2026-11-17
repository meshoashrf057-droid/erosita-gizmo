"""Independently check calibration01 against its raw host, satellites and accepted initial particles."""
from pathlib import Path
import hashlib,json,sys
from datetime import datetime,timezone
from contextlib import ExitStack
import h5py
import numpy as np

campaign=Path(__file__).resolve().parents[1]
mw=campaign.parent
sys.path.insert(0,str(mw))
import tg19_backend as backend
trial=campaign/'calibration01'
side=trial/'orbit_000.orbit.json';merged=trial/'orbit_000.hdf5'
cgm=mw/'cgm3_isolated_20261103_155517'
initial_host=cgm/'mw_cgm3_initial.hdf5'
old_accept=mw/'shooting_cpp_20261031_082656/accepted_orbit.json'

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def read(p): return json.loads(Path(p).read_text())
def require_equal(x,y,what):
    if not np.array_equal(x,y): raise ValueError('Mismatch: '+what)
def require_close(x,y,tol,what):
    if not np.allclose(x,y,atol=tol,rtol=0): raise ValueError('Mismatch: '+what)
def stat_signature(p):
    v=Path(p).stat(); return v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns

sc=read(side); accepted=read(old_accept); prior_ic=Path(accepted['initial_ic'])
freeze=read(cgm/'provenance/future_satellite_sources.json')
manifest=read(trial/'input_manifest_before.json')
sources=[Path(i['path']) for i in manifest]
cfg=read(campaign/'campaign.json')
expected_sources=[Path(cfg['host_endpoint']), *map(Path,freeze['source_files'][1:])]
if sources!=expected_sources: raise ValueError('Unexpected merge sources')
all_inputs=[*sources,merged,prior_ic,initial_host,old_accept,side]
st={str(p):stat_signature(p) for p in all_inputs}
h={str(p):sha(p) for p in all_inputs}
if [h[str(p)] for p in sources] != ['1f38047d72b5c6bdfd8a324489f0988c8fbb8250b4415a47e7ed902c94fb89ac',*freeze['source_sha256'][1:]]: raise ValueError('Frozen source mismatch')
if h[str(merged)] != sc['output_ic_sha256'] or h[str(prior_ic)]!=accepted['initial_ic_sha256']: raise ValueError('Wrong output or seed')
if h[str(initial_host)]!='8bf6e31f2d1dd55a0f93ab57ed6076e7364666991b078f3607dd0d9265245304': raise ValueError('Wrong initial host metadata')
finput=read(campaign/'provenance/cgm3_family_inputs.json')
expect=json.loads(json.dumps(freeze['model_key_inputs']))
expect['source_sha256'][0]=h[str(sources[0])]
expect['production_parameters'].update(CpuTimeBetRestartFile=900.,CritPhysDensity=3000.,MaxKernelRadius=1e56,MinSizeTimestep=1e-12,
    TimeBegin=cfg['start_time_code'],TimeOfFirstSnapshot=cfg['start_time_code'],TimeMax=cfg['target_time_code'])
expect['stage2_start_code']=cfg['start_time_code']
expect['stage2_target_code']=cfg['target_time_code']
if finput!=expect or sc['model_key_inputs']!=expect: raise ValueError('Stale family')
fam='cpp-shooting-family:'+hashlib.sha256(json.dumps(expect,sort_keys=True,allow_nan=False).encode()).hexdigest()
require_equal(fam,sc['family_key'],'family key')
if sc['alpha_applied'] or np.any(sc['shooting_increment_12d']) or np.any(sc['cumulative_offset_12d']): raise ValueError('Applied unwanted extra correction')
invariants=[]; max_absolute_accepted_shift_errors={}; origins_count={}; origins_mass_msun={}
with ExitStack() as stack:
    old,output,host_ic,*files=[stack.enter_context(h5py.File(p,'r')) for p in [prior_ic,merged,initial_host,*sources]]
    old_audit=json.loads(old['MergeProvenance/AuditJSON'].asstr()[()])
    embedded=json.loads(output['MergeProvenance/AuditJSON'].asstr()[()])
    if embedded!={k:v for k,v in sc.items() if k!='output_ic_sha256'}: raise ValueError('External and embedded provenance differ')
    require_equal(sc['stage2_start_code'],cfg['start_time_code'],'start')
    require_equal(sc['stage2_target_code'],cfg['target_time_code'],'target')
    expected_times=[cfg['start_time_code'],*freeze['source_times_code'][1:]]
    for f,t in zip(files,expected_times): require_equal(f['Header'].attrs['Time'],t,'individual source time')
    require_equal(output['Header'].attrs['Time'],cfg['start_time_code'],'merged start time')
    require_equal(sc['source_times_code'],expected_times,'individual epoch provenance')
    offsets=[cfg['start_time_code']-t for t in expected_times]
    require_equal(sc['source_birth_time_offsets_code'],offsets,'birth-time offsets')
    require_close((cfg['target_time_code']-cfg['start_time_code'])*sc['units']['time_gyr'],1.,1e-14,'exact one physical Gyr')
    require_equal(output['Header'].attrs['NumPart_Total'],np.sum([i['Header'].attrs['NumPart_Total'] for i in files],axis=0),'counts')
    require_equal(output['Header'].attrs['NumPart_ThisFile'],output['Header'].attrs['NumPart_Total'],'single file counts')
    for group in ('MWModelReference','ReservedParticleIDs'):
        a,b=host_ic[group].attrs,output[group].attrs
        if set(a)!=set(b): raise ValueError(group+' keys differ')
        for k in a: require_equal(a[k],b[k],group+'/'+k)
    wanted=np.asarray(old_audit['applied_initial_12d'])
    for f in [old,output]:
        actual=np.concatenate([f['MagellanicOrbitReference'][p+'_AppliedInitial'+v][:] for p in ['LMC','SMC'] for v in ['Position_kpc','Velocity_kms']])
        require_equal(actual,wanted,'real accepted embedded 12D')
    require_equal(sc['applied_initial_12d'],wanted,'accepted to new 12D')
    require_equal(sc['rotation_matrices'],freeze['model_key_inputs']['rotation_matrices'],'frozen rotations')
    for key in old['MagellanicOrbitReference']:
        if 'Shooting' not in key:
            require_equal(output['MagellanicOrbitReference'][key][:],old['MagellanicOrbitReference'][key][:],key)
    u=sc['units']; old_host_frame=backend.find_host_center_and_velocity(old); new_host_frame=backend.find_host_center_and_velocity(files[0])
    for state,frame in [(old_audit,old_host_frame),(sc,new_host_frame)]:
        for j,(name,unit) in enumerate([('source_centers_kpc','length_kpc'),('source_bulk_kms','velocity_kms')]):
            require_close(np.asarray(frame[j])*u[unit],state[name][0],1e-10,'host frame '+name)
    host_delta=[np.asarray(new_host_frame[j])-old_host_frame[j] for j in range(2)]
    for pt in (0,1,4,5):
        gname='PartType'+str(pt); outgroup=output[gname]; origin_map=output['MergeParticleOrigins'][gname][:]; prior=old[gname]
        old_origin=old['MergeParticleOrigins'][gname][:]
        if set(outgroup)!=set(prior): raise ValueError(gname+' particle field schema changed')
        off=0
        for cid,(file,system) in enumerate(zip(files,['MW','LMC','SMC'])):
            if gname not in file: continue
            src=file[gname]; count=len(src['ParticleIDs'])
            if set(src)!=set(outgroup): raise ValueError(system+'/'+gname+' schema')
            require_equal(origin_map[off:off+count],np.full(count,cid,dtype=origin_map.dtype),system+' row tags')
            ids=src['ParticleIDs'][:]
            in_own=(ids>=cid*100000000)&(ids<(cid+1)*100000000)
            if not np.all(in_own):raise ValueError(system+' IC block identity mismatch')
            fkey=system+'/'+gname
            origins_count[fkey]=count
            origins_mass_msun[fkey]=float(src['Masses'][:].sum()*u['mass_msun'])
            old_start=0
            if cid:
                ix=np.flatnonzero(old_origin==cid)
                if len(ix)!=count or (len(ix)>1 and not np.all(np.diff(ix)==1)): raise ValueError('Historical satellite mapping')
                old_start=int(ix[0])
            for key,ds in src.items():
                pair=fkey+'/'+key; maxerr=0.
                if ds.dtype != outgroup[key].dtype or ds.shape[1:]!=outgroup[key].shape[1:]: raise ValueError('Field dtype/width changed '+pair)
                for j in range(0,count,100000):
                    end=min(j+100000,count); raw=ds[j:end]; generated=outgroup[key][off+j:off+end]
                    if key in ('Acceleration','Potential'):
                        if np.any(generated!=0): raise ValueError('Gravity placeholder '+pair)
                    elif key=='StellarFormationTime':
                        expected_birth=raw.astype(np.float64)+offsets[cid]
                        require_equal(generated,expected_birth,pair+' epoch-relabelled birth')
                        require_close(cfg['start_time_code']-generated,expected_times[cid]-raw,2e-14,pair+' preserved physical age')
                    elif cid and key in ('Coordinates','Velocities','Vorticity'):
                        rotation=np.array(sc['rotation_matrices'][cid])
                        if key=='Vorticity': calculated=raw@rotation.T
                        else:
                            mode=int(key=='Velocities'); units=u['velocity_kms' if mode else 'length_kpc']
                            centers=np.array(sc['source_bulk_kms' if mode else 'source_centers_kpc'])
                            start=6*(cid-1)+3*mode
                            calculated=((raw*units-centers[cid])@rotation.T+centers[0]+wanted[start:start+3])/units
                        require_close(generated,calculated,1e-10,pair)
                    else:
                        require_equal(generated,raw,pair)
                    if cid:
                        oldrows=prior[key][old_start+j:old_start+end]
                        if key in ('Coordinates','Velocities'):
                            mode=int(key=='Velocities'); err=np.max(np.abs((generated-oldrows)-host_delta[mode]))
                            maxerr=max(maxerr,float(err))
                            if maxerr>1e-9: raise ValueError('Actual accepted satellite not replayed: '+pair+' error '+str(maxerr))
                        elif key=='StellarFormationTime':
                            require_equal(generated,oldrows.astype(np.float64)+offsets[cid],'accepted IC age-preserving epoch relabel: '+pair)
                        else:
                            require_equal(generated,oldrows,'accepted IC: '+pair)
                invariants.append(pair)
                if cid and key in ('Coordinates','Velocities'):
                    max_absolute_accepted_shift_errors[pair]=maxerr
            off+=count
        require_equal(off,len(origin_map),gname+' range coverage')
    require_equal(output['Header'].attrs['TargetFormat'],'cpp','native target')
for p in all_inputs:
    if stat_signature(p)!=st[str(p)]: raise ValueError('Input changed '+str(p))
report=dict(status='pass',output_ic_sha256=h[str(merged)],sidecar_sha256=h[str(side)],family_key=fam,
            initial_12d=wanted.tolist(),actual_accepted_ic_sha256=h[str(prior_ic)],source_sha256=[h[str(p)] for p in sources],
            actual_input_hashes=h,fields_checked_against_sources=invariants,field_count=len(invariants),
            counts_by_source_and_type=origins_count,mass_msun_by_source_and_type=origins_mass_msun,
            max_absolute_difference_vs_actual_accepted_after_new_host_shift=max_absolute_accepted_shift_errors,
            convention='Reload each raw source. Verify row origin, every field and accepted particles. Satellite center and bulk follow each host; potential/acceleration are zeroed for recomputation. Temperature, density, kernel, internal energy, metal vectors, stellar ages and composite identities retained; satellite birth epochs relabelled by exactly the source-to-host time offset. Old family physical gate is not inherited.',
            created_utc=datetime.now(timezone.utc).isoformat())
with (trial/'ic_verification.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({k:report[k] for k in ('status','output_ic_sha256','sidecar_sha256','family_key','field_count','counts_by_source_and_type','max_absolute_difference_vs_actual_accepted_after_new_host_shift')},indent=2),flush=True)
