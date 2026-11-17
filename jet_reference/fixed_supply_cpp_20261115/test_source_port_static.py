"""Static and synthetic boundary checks only; never imports scientific launchers or invokes a compiler."""
from pathlib import Path
import hashlib, json, math, re, unittest
B = Path(__file__).resolve().parent
S = B / 'source'
O = Path('/home/xingdi/gizmo-public-gizmo-cpp')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def text(p): return (S / p).read_text()
def active_config(p):
    return {a.split('=',1)[0]: a.split('=',1)[1] if '=' in a else None
            for line in p.read_text().splitlines()
            if (a := line.split('#',1)[0].strip())}
EXPECTED = {
 'SINK_PARTICLES':None, 'SINK_REPOSITION_ON_POTMIN':None,
 'SINK_DEBUG_FIX_MDOT_MBH':'(0.00124)', 'SINK_WIND_SPAWN':'2',
 'SINK_FOLLOW_ACCRETED_ANGMOM':'1', 'SINK_FOLLOW_ACCRETED_MOMENTUM':None,
 'SINK_FOLLOW_ACCRETED_COM':None, 'SINK_FB_COLLIMATED':None,
 'SINK_WAKEUP_GAS':None, 'SINK_INTERACT_ON_GAS_TIMESTEP':None,
 'SINK_OUTPUT_MOREINFO':None, 'OUTPUT_UNSPAWNED_SINKMASS':None}
MODIFIED = {'Config.sh','core/init.cc','core/begrun.cc','core/timestep.cc',
 'declarations/particle_data.h','declarations/allvars.h','sinks/sink.cc',
 'sinks/sink_swallow_and_kick.cc','sinks/sink_environment.cc'}
def legacy_dm(t,dt,duty,period_gyr,time_gyr):
    period=period_gyr/time_gyr
    on=duty*period
    if duty>=1: return .00124*dt
    if math.fmod(t,period)<on: return 2*(.00124/duty)*math.sin(math.pi*t/on)**2*dt
    return 0.
def port_dm(t,dt,duty,period_gyr,time_gyr):
    if not (math.isfinite(duty) and duty>0 and math.isfinite(period_gyr) and period_gyr>0):
        raise ValueError('finite positive duty/period required')
    period=period_gyr/time_gyr
    active=period*duty
    if duty>=1: return .00124*dt
    if math.fmod(t,period)>=active: return 0.
    return .00248*dt/duty*(math.sin(math.pi*t/active)**2)
def cap(dt,bin_,timebase=1.,timebins=60):
    if 0<=bin_<timebins:
        sinkdt=2.*((1<<bin_) if bin_ else 0)*timebase
        if dt>sinkdt: return .99*sinkdt,timebins
    return dt,bin_
class SourceContracts(unittest.TestCase):
 def test_original_tree_and_protected_references_unchanged(self):
    d=json.loads((B/'source_input_manifest.json').read_text())
    for f in d['files']:
        with self.subTest(path=f['path']): self.assertEqual(sha(O/f['path']),f['sha256'])
    for fn,digest in d['protected_reference_sha256'].items():
        with self.subTest(protected=fn): self.assertEqual(sha(Path(fn)),digest)
 def test_port_manifest_all_files_and_patch(self):
    d=json.loads((B/'source_port_manifest.json').read_text())
    for f in d['files']:
        with self.subTest(path=f['path']):
            self.assertEqual(sha(S/f['path']),f['sha256'])
            self.assertEqual((S/f['path']).stat().st_size,f['bytes'])
    self.assertEqual(sha(B/'fixed_supply_cpp.patch'),d['patch_sha256'])
    self.assertFalse(d['compiler_invoked']); self.assertFalse(d['jet_ic_created']); self.assertFalse(d['simulation_started'])
 def test_exact_nine_modified_paths(self):
    d=json.loads((B/'source_input_manifest.json').read_text())
    changed={f['path'] for f in d['files'] if sha(S/f['path'])!=f['sha256']}
    self.assertEqual(changed,MODIFIED)
 def test_no_compiled_or_generated_products(self):
    for p in S.rglob('*'):
        self.assertFalse(p.is_symlink(),str(p))
        self.assertNotIn(p.name,('.git','GIZMO','GIZMO_fixed_supply_cpp','GIZMO_config.h','compile_time_info.cc'))
        self.assertNotIn(p.suffix,('.o','.pyc','.a','.so'))
 def test_config_preserves_every_nonjet_setting(self):
    before=active_config(O/'Config.sh'); after=active_config(S/'Config.sh')
    self.assertEqual({k:v for k,v in before.items() if not k.startswith(('BH_','BLACK_HOLES'))},
                     {k:v for k,v in after.items() if k not in EXPECTED})
    self.assertEqual({k:after.get(k,'MISSING') for k in EXPECTED},EXPECTED)
 def test_forbidden_presets_absent(self):
    after=active_config(S/'Config.sh')
    for k in after:
        self.assertFalse(k.startswith(('FIRE','SINGLE_STAR','SNE_NONSINK')),k)
        self.assertNotIn(k,('SINK_SWALLOWGAS','SINK_ALPHADISK_ACCRETION','SINK_SCALE_SPAWNINGMASS_WITH_INITIALMASS',
                            'SINK_GRAVACCRETION','SINK_BONDI','SINK_GRAVCAPTURE_GAS','SINK_GRAVCAPTURE_NONGAS'))
 def test_fixed_formula_and_absolute_phase_source(self):
    s=text('sinks/sink.cc'); start=s.index('#ifdef SINK_DEBUG_FIX_MDOT_MBH',s.index('double dm = P[n].Sink_Mdot * dt;'))
    end=s.index('double radiation_loss',start); block=s[start:end]
    for clause in ['All.Sink_fb_period / UNIT_TIME_IN_GYR','All.Sink_fb_duty_cycle * period_sink',
                   'All.Sink_fb_duty_cycle >= 1','dm = SINK_DEBUG_FIX_MDOT_MBH * dt',
                   'fmod(All.Time, period_sink) < period_sink_on',
                   '2. * (SINK_DEBUG_FIX_MDOT_MBH / All.Sink_fb_duty_cycle)',
                   'pow(sin(M_PI * All.Time / period_sink_on), 2) * dt']:
        self.assertIn(clause,block)
    self.assertNotIn('TimeBegin',block); self.assertNotIn('Sink_Mdot =',block)
    self.assertIn('dt = P[n].dt_since_last_gas_search;',s[:start])
 def test_registered_controls_and_validity_guard(self):
    s=text('core/begrun.cc')
    for canonical,legacy in [('Sink_fb_duty_cycle','BH_fb_duty_cycle'),('Sink_fb_period','BH_fb_period')]:
        self.assertIn('strcpy(tag[nt], "'+canonical+'");',s)
        self.assertIn('strcpy(alternate_tag[nt], "'+legacy+'");',s)
        self.assertIn('addr[nt] = &All.'+canonical+';',s)
        self.assertIn('!isfinite(All.'+canonical+')',s)
        self.assertIn('!(All.'+canonical+' > 0)',s)
 def test_three_mass_debit_exemptions_and_reservoir_debit(self):
    s=text('sinks/sink.cc')
    self.assertIn('#ifndef SINK_DEBUG_FIX_MDOT_MBH\n        P[n].Mass -= radiation_loss; P[n].Sink_Mass -= radiation_loss;\n#endif',s)
    self.assertIn('#ifndef SINK_DEBUG_FIX_MDOT_MBH\n        P[n].Sink_Mass -= dm_wind;\n#endif',s)
    s=text('sinks/sink_swallow_and_kick.cc')
    self.assertRegex(s,r'#ifndef SINK_DEBUG_FIX_MDOT_MBH\s+P\[i\]\.Mass -= P\[j\]\.Mass;[^\n]*\n#endif\s+P\[i\]\.unspawned_wind_mass -= P\[j\]\.Mass;')
 def test_native_wind_ratio_and_absolute_particlemass_unchanged(self):
    self.assertIn('(1.-All.Sink_accreted_fraction) / All.Sink_accreted_fraction * dm;',text('sinks/sink.cc'))
    s=text('sinks/sink_swallow_and_kick.cc')
    self.assertIn('#else\n    return All.Sink_outflow_particlemass;\n#endif',s)
    self.assertIn('P[n].unspawned_wind_mass += dm_wind;',text('sinks/sink.cc'))
 def test_gas_timebin_state_transfer_initialization_and_minimum(self):
    self.assertIn('int LowestSinkTimeBin;',text('declarations/particle_data.h'))
    self.assertIn('if(P[i].Type == 0) {P[i].LowestSinkTimeBin = TIMEBINS;}',text('core/init.cc'))
    s=text('sinks/sink_environment.cc')
    self.assertIn('in->TimeBin = P[i].TimeBin;',s)
    self.assertEqual(s.count('if(local.TimeBin < P[j].LowestSinkTimeBin)'),1)
    self.assertIn('#pragma omp critical(fixed_supply_lowest_sink_timebin)',s)
    self.assertIsNotNone(re.search(r'if\(P\[j\]\.Type == 0\).*?#pragma omp critical',s,re.S))
    self.assertIn('if(P[j].Type == 0)',s)
 def test_timestep_cap_uses_valid_bin_and_legacy_reset(self):
    self.assertIn('#define  TIMEBINS        60',text('declarations/typedefs.h'))
    self.assertIn('#define GET_INTEGERTIME_FROM_TIMEBIN(bin) ((bin ? (((integertime) 1) << bin) : 0))',text('declarations/macros.h'))
    s=text('core/timestep.cc')
    self.assertIn('P[p].LowestSinkTimeBin >= 0 && P[p].LowestSinkTimeBin < TIMEBINS',s)
    self.assertIn('2. * GET_PHYSICAL_TIMESTEP_FROM_TIMEBIN(P[p].LowestSinkTimeBin,p)',s)
    self.assertIn('if(dt > dt_sink) {dt = 0.99 * dt_sink; P[p].LowestSinkTimeBin = TIMEBINS;}',s)
 def test_required_hdf5_blocks_have_native_readers(self):
    s=text('file_io/read_ic.cc'); io=text('file_io/io.cc')
    for block,field in [('IO_SINKMASS','Sink_Mass'),('IO_SINK_FORM_MASS','Sink_InitialMass'),
                        ('IO_SINKMDOT','Sink_Mdot'),('IO_SINK_ANGMOM','Sink_Specific_AngMom'),
                        ('IO_SINKPROGS','Sink_NProgenitors'),('IO_UNSPMASS','Unspawned_Wind_Mass')]:
        self.assertIn('case '+block+':',s); self.assertIn('"'+field+'"',io)
    self.assertIn('#if defined(SINK_WIND_SPAWN) && defined(OUTPUT_UNSPAWNED_SINKMASS)',s)
 def test_preprocessor_nesting_balanced_in_modified_cpp_files(self):
    for fn in MODIFIED-{'Config.sh'}:
        stack=[]
        for i,line in enumerate(text(fn).splitlines(),1):
            m=re.match(r'^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b',line)
            if not m: continue
            op=m.group(1)
            if op in ('if','ifdef','ifndef'): stack.append(i)
            elif op=='endif':
                self.assertTrue(stack,(fn,i)); stack.pop()
            else: self.assertTrue(stack,(fn,i))
        self.assertFalse(stack,(fn,stack))
class SyntheticBoundaries(unittest.TestCase):
 def test_legacy_agreement_across_periods_and_duties(self):
    for unit in [.5,.9777923542981722,1.]:
      for duty in [.1,.3,.5,1.,1.4]:
       for period in [.2,1.,2.]:
        for t in [0.,1e-10,.125,.3,.999999,1.,1.6363392421373844,2.37,4.09]:
         with self.subTest(unit=unit,duty=duty,period=period,t=t):
          self.assertTrue(math.isclose(port_dm(t,.003,duty,period,unit),legacy_dm(t,.003,duty,period,unit),rel_tol=2e-14,abs_tol=1e-18))
 def test_continuous_supply_is_not_mdot_times_bh_mass(self):
    for duty in [1.,1.01,100.]:
        self.assertAlmostEqual(port_dm(1.636,.01,duty,1.,.9777923542981722),.0000124,places=18)
 def test_duty_half_peak_and_off_window(self):
    self.assertAlmostEqual(port_dm(.25,.01,.5,1.,1.),4*.00124*.01,places=18)
    self.assertEqual(port_dm(.5,.01,.5,1.,1.),0.)
    self.assertEqual(port_dm(.75,.01,.5,1.,1.),0.)
 def test_legacy_absolute_sine_phase_not_reset_at_each_period(self):
    observed=port_dm(1.05,.01,.3,1.,1.)
    reset_phase=2*(.00124/.3)*math.sin(math.pi*.05/.3)**2*.01
    self.assertGreater(abs(observed-reset_phase),1e-6)
    self.assertTrue(math.isclose(observed,legacy_dm(1.05,.01,.3,1.,1.),rel_tol=1e-14))
 def test_invalid_duty_period_fail_closed(self):
    for bad in [0.,-1.,math.nan,math.inf,-math.inf]:
        with self.subTest(bad=bad):
            with self.assertRaises(ValueError): port_dm(1.,.1,bad,1.,1.)
            with self.assertRaises(ValueError): port_dm(1.,.1,1.,bad,1.)
 def test_timestep_cap_and_reset_boundaries(self):
    self.assertEqual(cap(20.,3),(15.84,60))
    self.assertEqual(cap(16.,3),(16.,3))
    self.assertEqual(cap(10.,3),(10.,3))
    self.assertEqual(cap(20.,60),(20.,60))
    self.assertEqual(cap(20.,29,timebins=29),(20.,29))
    self.assertEqual(cap(20.,-1),(20.,-1))
    self.assertEqual(cap(20.,0),(0.,60)) # legacy zero-bin behavior, not silently reinterpreted
 def test_30_physical_myr_and_spawn_pair_budget(self):
    unit=3.085678e21/1e5/(31557600e9)
    delta=.03/unit; begin=1.6363392421373844
    self.assertTrue(math.isclose(delta,.030681360790075957,abs_tol=1e-16,rel_tol=0))
    self.assertTrue(math.isclose(begin+delta,1.6670206029274603,abs_tol=1e-15,rel_tol=0))
    pair_code=2e-7/.00124
    self.assertLess(pair_code*unit*1000,.2)
    self.assertGreater(pair_code*unit*1000,.15)
    self.assertLess(.00124*delta,.0004) # requested full-stage supply below legacy per-step cap mass
if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    with (B/'static_checks.log').open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    report={'scope':'source/text and synthetic Python contracts only; no compiler, IC, GIZMO, or runtime physics exercised',
            'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
            'static_checks_passed':result.wasSuccessful(),'runtime_validated':False,
            'log_sha256':sha(B/'static_checks.log'),'test_script_sha256':sha(Path(__file__)),
            'port_manifest_sha256':sha(B/'source_port_manifest.json')}
    (B/'static_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
