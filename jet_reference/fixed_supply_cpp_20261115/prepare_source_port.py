"""Prepare isolated fixed-supply C++ sources only. Never compile or run GIZMO."""
from pathlib import Path
import datetime,difflib,hashlib,json,os,shutil,stat
BASE=Path(__file__).resolve().parent
ORIGINAL=Path('/home/xingdi/gizmo-public-gizmo-cpp')
LEGACY=Path('/home/xingdi/gizmo-public')
DEST=BASE/'source'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def write_json(name,value):
    with (BASE/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
assert not DEST.exists()
files=[];excluded=[]
for current,dirs,names in os.walk(ORIGINAL):
    for name in list(dirs):
        if name in ('.git','.idea','__pycache__','.pytest_cache'):
            excluded.append({'path':str((Path(current)/name).relative_to(ORIGINAL)),'reason':'metadata/cache directory'})
            dirs.remove(name)
    for name in names:
        p=Path(current)/name;rel=p.relative_to(ORIGINAL)
        if name in ('GIZMO','GIZMO_config.h','compile_time_info.cc') or ':Zone.Identifier' in name or p.suffix in ('.o','.a','.so','.pyc') or name.endswith(('.log','.out','.err')):
            excluded.append({'path':str(rel),'reason':'binary/generated/build/cache/metadata product'});continue
        if p.is_symlink():raise RuntimeError('Unexpected source symlink: '+str(p))
        if not stat.S_ISREG(p.stat().st_mode):raise RuntimeError('Nonregular source file: '+str(p))
        files.append({'path':str(rel),'bytes':p.stat().st_size,'sha256':sha(p)})
guard_paths=[ORIGINAL/'Config.sh',ORIGINAL/'GIZMO',LEGACY/'Config.sh',LEGACY/'GIZMO']
guard_paths += [LEGACY/x for x in ('allvars.h','init.c','timestep.c','begrun.c','galaxy_sf/blackholes/blackhole.c','galaxy_sf/blackholes/blackhole_swallow_and_kick.c','galaxy_sf/blackholes/blackhole_environment.c')]
guards={str(p):sha(p) for p in guard_paths if p.is_file()}
DEST.mkdir()
for item in files:
    source=ORIGINAL/item['path'];target=DEST/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target);assert sha(target)==item['sha256']
write_json('source_input_manifest.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_root':str(ORIGINAL),'destination':str(DEST),'files':files,'excluded':excluded,'protected_reference_sha256':guards})
edits={}
def replace(rel,before,after):
    p=DEST/rel;s=p.read_text()
    assert s.count(before)==1,(rel,'expected unique replacement',s.count(before),before[:80])
    if rel not in edits:edits[rel]=(ORIGINAL/rel).read_text()
    p.write_text(s.replace(before,after,1))
replace('declarations/allvars.h',
'  double SinkRadiativeEfficiency;   /*!< Radiative efficiency determined by the spin value, default value is 0.1 */\n',
'''  double SinkRadiativeEfficiency;   /*!< Radiative efficiency determined by the spin value, default value is 0.1 */
#ifdef SINK_DEBUG_FIX_MDOT_MBH
  double Sink_fb_duty_cycle; /*!< Legacy external-supply duty; >=1 is continuous. */
  double Sink_fb_period; /*!< Legacy period in physical Gyr; phase uses absolute All.Time. */
#endif
''')
replace('declarations/particle_data.h',
'#ifdef SINK_FOLLOW_ACCRETED_ANGMOM\n    MyFloat Sink_Specific_AngMom[3];',
'''#ifdef SINK_WAKEUP_GAS
    int LowestSinkTimeBin; /*!< Lowest interacting sink bin for legacy gas-step cap; TIMEBINS is unset. */
#endif
#ifdef SINK_FOLLOW_ACCRETED_ANGMOM
    MyFloat Sink_Specific_AngMom[3];''')
replace('core/init.cc',
'#ifdef SINK_PARTICLES\n#if (SINGLE_STAR_SINK_FORMATION & 8)',
'''#ifdef SINK_PARTICLES
#ifdef SINK_WAKEUP_GAS
        if(P[i].Type == 0) {P[i].LowestSinkTimeBin = TIMEBINS;}
#endif
#if (SINGLE_STAR_SINK_FORMATION & 8)''')
replace('core/begrun.cc',
'''        strcpy(tag[nt], "SinkFeedbackFactor");
        addr[nt] = &All.SinkFeedbackFactor;
        id[nt++] = REAL;
''',
'''        strcpy(tag[nt], "SinkFeedbackFactor");
        addr[nt] = &All.SinkFeedbackFactor;
        id[nt++] = REAL;

#ifdef SINK_DEBUG_FIX_MDOT_MBH
        strcpy(tag[nt], "Sink_fb_duty_cycle");
        strcpy(alternate_tag[nt], "BH_fb_duty_cycle");
        addr[nt] = &All.Sink_fb_duty_cycle;
        id[nt++] = REAL;
        strcpy(tag[nt], "Sink_fb_period");
        strcpy(alternate_tag[nt], "BH_fb_period");
        addr[nt] = &All.Sink_fb_period;
        id[nt++] = REAL;
#endif
''')
replace('core/begrun.cc',
'    /* now we\'re going to do a bunch of checks */',
'''#ifdef SINK_DEBUG_FIX_MDOT_MBH
    /* Invalid legacy duty/period has no meaningful fixed-supply interpretation. */
    if(!(All.Sink_fb_duty_cycle > 0) || !isfinite(All.Sink_fb_duty_cycle) ||
       !(All.Sink_fb_period > 0) || !isfinite(All.Sink_fb_period))
    {
        if(ThisTask==0) {printf("Fixed external supply requires finite positive Sink_fb_duty_cycle and Sink_fb_period.\\n");}
        endrun(1);
    }
#endif

    /* now we're going to do a bunch of checks */''')
replace('sinks/sink.cc',
'        double dm = P[n].Sink_Mdot * dt;\n',
'''        double dm = P[n].Sink_Mdot * dt;
#ifdef SINK_DEBUG_FIX_MDOT_MBH
        /* Faithful BH_DEBUG_FIX_MDOT_MBH port: external supply, not Sink_Mdot or Mdot/Mbh.
           Preserve the legacy absolute-Time phase; do not subtract the jet start time. */
        dm = 0;
        const double period_sink = All.Sink_fb_period / UNIT_TIME_IN_GYR;
        const double period_sink_on = All.Sink_fb_duty_cycle * period_sink;
        if(All.Sink_fb_duty_cycle >= 1) {dm = SINK_DEBUG_FIX_MDOT_MBH * dt;}
        else if(fmod(All.Time, period_sink) < period_sink_on)
        {
            dm = 2. * (SINK_DEBUG_FIX_MDOT_MBH / All.Sink_fb_duty_cycle)
                 * pow(sin(M_PI * All.Time / period_sink_on), 2) * dt;
        }
#endif
''')
replace('sinks/sink.cc',
'        P[n].Mass -= radiation_loss; P[n].Sink_Mass -= radiation_loss;\n',
'''#ifndef SINK_DEBUG_FIX_MDOT_MBH
        P[n].Mass -= radiation_loss; P[n].Sink_Mass -= radiation_loss;
#endif
''')
replace('sinks/sink.cc',
'''        if(dm_wind > P[n].Sink_Mass) {dm_wind = P[n].Sink_Mass;}
        P[n].Sink_Mass -= dm_wind;
''',
'''        if(dm_wind > P[n].Sink_Mass) {dm_wind = P[n].Sink_Mass;}
#ifndef SINK_DEBUG_FIX_MDOT_MBH
        P[n].Sink_Mass -= dm_wind;
#endif
''')
replace('sinks/sink_swallow_and_kick.cc',
'        P[i].Mass -= P[j].Mass; /* make sure the operation is mass conserving! */\n',
'''#ifndef SINK_DEBUG_FIX_MDOT_MBH
        P[i].Mass -= P[j].Mass; /* make sure the operation is mass conserving! */
#endif
''')
replace('sinks/sink_environment.cc',
'''#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    MyFloat Sink_Specific_AngMom[3];
''',
'''#ifdef SINK_WAKEUP_GAS
    int TimeBin;
#endif
#if defined(SINK_RETURN_ANGMOM_TO_GAS)
    MyFloat Sink_Specific_AngMom[3];
''')
replace('sinks/sink_environment.cc',
'''    in->KernelRadius = P[i].KernelRadius; in->ID = P[i].ID;
''',
'''    in->KernelRadius = P[i].KernelRadius; in->ID = P[i].ID;
#ifdef SINK_WAKEUP_GAS
    in->TimeBin = P[i].TimeBin;
#endif
''')
replace('sinks/sink_environment.cc',
'''                j = ngblist[n]; /* since we use the -threaded- version above of ngb-finding, its super-important this is the lower-case ngblist here! */
                if( (P[j].Mass > 0) && (P[j].Type != 5) && (P[j].ID != local.ID) )
''',
'''                j = ngblist[n]; /* since we use the -threaded- version above of ngb-finding, its super-important this is the lower-case ngblist here! */
#ifdef SINK_WAKEUP_GAS
                /* Only gas consumes this field. Protect the comparison and update together:
                   the legacy separate comparison/atomic-write was not a thread-safe minimum. */
                if(P[j].Type == 0)
                {
#ifdef _OPENMP
#pragma omp critical(fixed_supply_lowest_sink_timebin)
#endif
                    {
                        if(local.TimeBin < P[j].LowestSinkTimeBin) {P[j].LowestSinkTimeBin = local.TimeBin;}
                    }
                }
#endif
                if( (P[j].Mass > 0) && (P[j].Type != 5) && (P[j].ID != local.ID) )
''')
replace('core/timestep.cc',
'''#ifdef SINK_PARTICLES
    if(P[p].Type == 5)
''',
'''#ifdef SINK_PARTICLES
#ifdef SINK_WAKEUP_GAS
    /* Legacy BH_WAKEUP_GAS: cap gas to 0.99*(2*the lowest interacting sink step).
       The unset sentinel is not a real bin and must not be shifted. Reset only
       when the cap is applied, preserving the legacy lifetime of a recorded bin. */
    if(P[p].Type == 0 && P[p].LowestSinkTimeBin >= 0 && P[p].LowestSinkTimeBin < TIMEBINS)
    {
        const double dt_sink = 2. * GET_PHYSICAL_TIMESTEP_FROM_TIMEBIN(P[p].LowestSinkTimeBin,p);
        if(dt > dt_sink) {dt = 0.99 * dt_sink; P[p].LowestSinkTimeBin = TIMEBINS;}
    }
#endif
    if(P[p].Type == 5)
''')
replace('Config.sh',
'''BLACK_HOLES
BH_REPOSITION_ON_POTMIN=2

#BH_DEBUG_FIX_MDOT_MBH=(0.00124)
#BH_WIND_SPAWN=2
#BH_FOLLOW_ACCRETED_ANGMOM=1
#BH_FOLLOW_ACCRETED_MOMENTUM
#BH_FOLLOW_ACCRETED_COM
#BH_FB_COLLIMATED

#BH_WAKEUP_GAS
#BH_INTERACT_ON_GAS_TIMESTEP
#BH_OUTPUT_MOREINFO
''',
'''# Isolated faithful legacy fixed-external-supply jet; no natural accretion or swallowing preset.
SINK_PARTICLES
SINK_REPOSITION_ON_POTMIN
SINK_DEBUG_FIX_MDOT_MBH=(0.00124)
SINK_WIND_SPAWN=2
SINK_FOLLOW_ACCRETED_ANGMOM=1
SINK_FOLLOW_ACCRETED_MOMENTUM
SINK_FOLLOW_ACCRETED_COM
SINK_FB_COLLIMATED
SINK_WAKEUP_GAS
SINK_INTERACT_ON_GAS_TIMESTEP
SINK_OUTPUT_MOREINFO
OUTPUT_UNSPAWNED_SINKMASS
''')
after=[{'path':x['path'],'bytes':(DEST/x['path']).stat().st_size,'sha256':sha(DEST/x['path'])} for x in files]
for x in files:assert sha(ORIGINAL/x['path'])==x['sha256']
for p,h in guards.items():assert sha(p)==h
diff=''
for rel,before in edits.items():
    diff+=''.join(difflib.unified_diff(before.splitlines(keepends=True),(DEST/rel).read_text().splitlines(keepends=True),fromfile='original/'+rel,tofile='source/'+rel))
with (BASE/'fixed_supply_cpp.patch').open('x') as f:f.write(diff)
write_json('source_port_manifest.json',{'prepared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'source_prepared_not_compiled','source_root':str(ORIGINAL),'source_copy':str(DEST),'files':after,'modified_files':list(edits),'protected_references_unchanged':True,'source_input_manifest_sha256':sha(BASE/'source_input_manifest.json'),'patch_sha256':sha(BASE/'fixed_supply_cpp.patch'),'compiler_invoked':False,'jet_ic_created':False,'simulation_started':False})
print(json.dumps({'copied_file_count':len(files),'modified_files':list(edits),'excluded_count':len(excluded),'source_prepared_not_compiled':True},indent=2))
