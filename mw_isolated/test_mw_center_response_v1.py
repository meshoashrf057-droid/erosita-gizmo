from pathlib import Path
import json,hashlib,sys,tempfile,subprocess
import numpy as np
import h5py

FIELDS=['ParticleIDs','ParticleChildIDsNumber','ParticleIDGenerationNumber']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def symmetric(n,scale):
    rng=np.random.default_rng(42+n)
    half=rng.uniform(-scale,scale,size=(n//2,3))
    return np.vstack([half,-half])
def make_fixture(root, common=False):
    rows=[]; snapshot_paths=[]; paths=[]
    for label,shift in [('trial10',0.),('trial11',.2)]:
        ic=root/f'mw_lmc_smc_relaxed_{label}_000.hdf5'
        snap=root/f'{label}_snapshot_011.hdf5'
        ids=np.r_[np.arange(1000,1400),np.arange(120000000,120000080),np.arange(210000000,210000080)].astype('u8')
        child=np.zeros(len(ids),dtype='u8'); ids[1]=ids[0];child[1]=1
        gen=np.zeros(len(ids),dtype='u8')
        origin=np.r_[np.zeros(400),np.ones(80),np.full(80,2)].astype('u1')
        initialpos=np.vstack([symmetric(400,.5)+500,symmetric(80,.1)+[530,480,470],symmetric(80,.1)+[520,460,450]])
        oldvel=np.vstack([np.tile([10,20,30],(400,1)),np.tile([40,50,60],(80,1)),np.tile([70,80,90],(80,1))]).astype(float)
        attrs=dict(PresentTargetTime_Code=1.6363120263358653,MW_BH_ID=90000000,LMC_ID_START=100000000,LMC_ID_END=199999999,SMC_ID_START=200000000,SMC_ID_END=299999999,LMC_OLD_STAR_ID_START=120000000,LMC_OLD_STAR_ID_END=129999999,SMC_OLD_STAR_ID_START=210000000,SMC_OLD_STAR_ID_END=219999999)
        for path,is_snapshot in [(ic,False),(snap,True)]:
            pos=initialpos.copy();vel=oldvel.copy();ii=ids.copy();cc=child.copy();gg=gen.copy()
            if is_snapshot:
                pos[:400,0]+=shift
                pos[400:480,0]+=shift*.25
                pos[480:,0]+=shift*.5
                pos=np.vstack([pos,symmetric(20,.1)+[502+shift,500,500]])
                vel=np.vstack([vel,np.tile([100,20,30],(20,1))])
                ii=np.r_[ii,np.arange(9000,9020,dtype='u8')];cc=np.r_[cc,np.zeros(20,dtype='u8')];gg=np.r_[gg,np.ones(20,dtype='u8')]
            with h5py.File(path,'w') as f:
                h=f.create_group('Header');h.attrs.update(Time=1.6363120263358653 if is_snapshot else .6136,NumFilesPerSnapshot=1,NumFiles=1,ComovingIntegrationOn=0,HubbleParam=1,Redshift=0,MassTable=np.zeros(6))
                g=f.create_group('PartType4')
                for k,v in dict(Coordinates=pos,Velocities=vel,Masses=np.ones(len(ii)),ParticleIDs=ii,ParticleChildIDsNumber=cc,ParticleIDGenerationNumber=gg).items():g[k]=v
                bh=f.create_group('PartType5');bh['ParticleIDs']=np.array([90000000],dtype='u8')
                bhx=500.1+(shift if common else 2*shift) if is_snapshot else 500.
                bh['Coordinates']=np.array([[bhx,500,500]]);bh['Velocities']=np.array([[10.,20,30]])
                if not is_snapshot:
                    orbit=f.create_group('MagellanicOrbitReference');orbit.attrs.update(attrs)
                    f.create_group('MergeParticleOrigins')['PartType4']=origin
            paths.append(path)
        hp=np.array([500.1+(shift if common else 2*shift),500,500]);hv=np.array([(400*10+20*100)/420,20,30])
        lp=np.array([530+shift*.25,480,470]);sp=np.array([520+shift*.5,460,450]);lv=np.array([40,50,60]);sv=np.array([70,80,90])
        h=dict(file=str(snap),time_code=1.6363120263358653,elapsed_gyr=1.,host_center_method='MW BH position + inner-star velocity')
        for prefix,pos,vel in [('host',hp,hv),('lmc',lp-hp,lv-hv),('smc',sp-hp,sv-hv)]:
            for a,v in zip('xyz',pos):h[f'{prefix}_{a}_kpc']=v
            for a,v in zip('xyz',vel):h[f'{prefix}_v{a}_kms']=v
        sig=dict(algorithm='fixed-original-star-identities-shrinking-core-v4',tracer_policy='original-stars',identity_fields=FIELDS,host_frame='BH-position+MW-inner10kpc-mass-weighted-star-velocity-v3',lmc_core_radius_kpc=8.,smc_core_radius_kpc=3.)
        prov=dict(schema_version=1,reference_ic=str(ic),reference_ic_sha256=sha(ic),model_key='fixture-model',trial_label=label,target_time_code=1.6363120263358653,unit_time_gyr=.9777923542981722,target_covered=True,shooting_usable=True,evaluation_method='exact target snapshot',target_snapshot_sources=[dict(file=str(snap),time_code=1.6363120263358653,weight=1.)],measurement_signature=sig,stellar_tracer_policy='original-stars',reference_tracer_counts={'lmc':80,'smc':80})
        orbit=dict(trial_label=label,source_files=['mw_fixture','lmc_fixture','smc_fixture'],model_key='fixture-model',output_ic_sha256=sha(ic),units={'length_kpc':1.,'velocity_kms':1.,'mass_msun':1e10,'time_gyr':.9777923542981722},source_sha256=['a'*64,'b'*64,'c'*64],source_centers_kpc=[[0,0,0]]*3,source_bulk_kms=[[0,0,0]]*3,rotation_matrices=[np.eye(3).tolist()]*3,stage2_start_code=.6136,stage2_target_code=1.6363120263358653,lookback_gyr=1.,model_key_inputs={'fixture':True})
        rows.append(dict(trial=label,provenance=prov,orbit=orbit,history=[h],sigma=[1.]*12,observed=[0.]*12))
    p=root/'history.json';p.write_text(json.dumps(dict(schema_version=1,trials=rows)))
    return p,paths


import unittest
from audit_mw_center_response_v1 import audit_pair

class CenterAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='mw_center_audit_test_')
        self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def test_bh_offset_and_frozen_population_readonly(self):
        hist,files=make_fixture(self.root)
        before={str(p):sha(p) for p in files+[hist]}
        result=audit_pair(hist)
        for row in result['comparison']['apertures'].values():
            np.testing.assert_allclose(row['stellar_center_delta_kpc'],[.2,0,0],atol=1e-10)
            np.testing.assert_allclose(row['bh_minus_stellar_delta_kpc'],[.2,0,0],atol=1e-10)
            np.testing.assert_allclose(row['satellites']['lmc']['relative_bh_position_delta_kpc'],[-.35,0,0],atol=1e-10)
        for t in result['trials']:
            self.assertEqual(t['fixed_mw_matched_count'],400)
            self.assertEqual(t['official_host_nstar'],420)
            np.testing.assert_allclose(t['stellar_centers']['5']['fixed_mw_velocity_10kpc']['velocity_kms'],[10,20,30],atol=1e-12)
        self.assertEqual(before,{str(p):sha(p) for p in files+[hist]})
    def test_common_translation_not_mislabeled_bh_offset(self):
        hist,_=make_fixture(self.root,common=True)
        result=audit_pair(hist)
        for row in result['comparison']['apertures'].values():
            np.testing.assert_allclose(row['stellar_center_delta_kpc'],[.2,0,0],atol=1e-10)
            np.testing.assert_allclose(row['bh_minus_stellar_delta_kpc'],[0,0,0],atol=1e-10)
    def test_ic_content_fingerprint(self):
        hist,files=make_fixture(self.root)
        with h5py.File(files[0],'r+') as f:f['PartType4/Velocities'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'指纹'): audit_pair(hist)
    def test_complete_identity_duplicate_is_rejected(self):
        hist,files=make_fixture(self.root)
        with h5py.File(files[1],'r+') as f:f['PartType4/ParticleChildIDsNumber'][1]=0
        with self.assertRaisesRegex(ValueError,'重复完整'): audit_pair(hist)
    def test_wrong_endpoint_time(self):
        hist,files=make_fixture(self.root)
        with h5py.File(files[1],'r+') as f:f['Header'].attrs['Time']=1.5
        with self.assertRaisesRegex(ValueError,'时间'): audit_pair(hist)
    def test_near_unity_unit_factor_does_not_break_reproduction(self):
        hist,_=make_fixture(self.root)
        p=json.loads(hist.read_text());factor=1.0000001356294
        for t in p['trials']:t['orbit']['units']['length_kpc']=factor
        hist.write_text(json.dumps(p));result=audit_pair(hist)
        np.testing.assert_allclose(result['comparison']['bh_position_delta_kpc'],np.array([.4,0,0])*factor,atol=1e-10)
        for row in result['comparison']['apertures'].values():
            np.testing.assert_allclose(row['bh_minus_stellar_delta_kpc'],np.array([.2,0,0])*factor,atol=1e-10)

if __name__=='__main__': unittest.main()
