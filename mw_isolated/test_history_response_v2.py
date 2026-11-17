"""Known-map tests, not live GIZMO validation."""
import json
import copy
import unittest
import numpy as np
from history_response_v2 import propose_candidates, INITIAL_SCALE
from propose_history_shooting_v2 import select_anchor, evaluate_live_steps, resolve_parent_index, validate_history


class HistoryResponseTests(unittest.TestCase):
    def setUp(self):
        self.base = np.arange(12,dtype=float)/10
        self.static = np.eye(12)
        self.static[3,6] = 0.2
        self.live = self.static.copy()
        self.live[0,0] = -2.0  # Measured response has opposite sign to static prior.
        self.live[0,1] = 0.3
        self.live[1,1] = 1.5
        self.residual = np.zeros(12); self.residual[:2] = [5.,-4.]
        self.hx = np.tile(self.base,(3,1))
        self.hx[0,0] -= 2.0
        self.hx[1,1] -= 2.0
        self.hr = self.residual+(self.hx-self.base)@self.live.T
        self.forward = lambda x: self.static@x + np.arange(12)*0.4

    def propose(self, **kwargs):
        args=dict(base=self.base,residual=self.residual,sigma=np.ones(12),
                  history_initials=self.hx,history_residuals=self.hr,forward=self.forward)
        args.update(kwargs)
        return propose_candidates(**args)

    def test_linear_coupled_mismatch_learns_sign_and_predicts_live(self):
        report=self.propose()
        selected=report['selected']
        self.assertIsNotNone(selected)
        self.assertEqual(len(report['candidates']),9)
        self.assertEqual(report['diagnostics']['history_rank'],2)
        step=np.array(selected['increment_12d'])
        self.assertGreater(step[0],0.)
        live_residual=self.residual+self.live@step
        self.assertLess(np.linalg.norm(live_residual),np.linalg.norm(self.residual))
        self.assertTrue(np.all(np.abs(live_residual[:2])<np.abs(self.residual[:2])))
        # Strong ridge intentionally leaves a bias toward the static prior.
        self.assertLess(np.linalg.norm(np.array(selected['predicted_residual_12d'])-live_residual),0.25)
        for candidate in report['candidates']:
            if candidate['ridge_relative']==0.001:
                exact=self.residual+self.live@candidate['increment_12d']
                np.testing.assert_allclose(candidate['predicted_residual_12d'],exact,atol=3e-5,rtol=0)
        # Pure static response has the wrong sign for the first component.
        self.assertGreater(selected['static_predicted_residual_12d'][0],self.residual[0])
        json.dumps(report,allow_nan=False)

    def test_common_caps_preserve_direction(self):
        report=self.propose(max_position_step=.1,max_velocity_step=.2)
        selected=report['selected']; self.assertIsNotNone(selected)
        step=np.array(selected['increment_12d']); raw=np.array(selected['raw_increment_12d'])
        np.testing.assert_allclose(step,selected['step_factor']*raw,atol=1e-13)
        for i,cap in enumerate((.1,.2,.1,.2)):
            self.assertLessEqual(np.linalg.norm(step[i*3:i*3+3]),cap+1e-12)

    def test_constant_or_rank_one_history_fails(self):
        hx=np.tile(self.base,(3,1))
        with self.assertRaisesRegex(ValueError,'independent'):
            self.propose(history_initials=hx,history_residuals=np.tile(self.residual,(3,1)),anchor_index=2)
        hx[0,0]-=2; hx[1,0]-=1
        with self.assertRaisesRegex(ValueError,'independent'):
            self.propose(history_initials=hx)

    def test_missing_anchor_match_fails(self):
        hx=self.hx.copy();hx[-1,0]+=.01
        with self.assertRaisesRegex(ValueError,'Anchor'):
            self.propose(history_initials=hx)
        hr=self.hr.copy();hr[-1,0]+=.01
        with self.assertRaisesRegex(ValueError,'Anchor'):
            self.propose(history_residuals=hr)

    def test_earlier_anchor_uses_correct_static_endpoint_and_keeps_secants(self):
        # A new rejected observation follows the former base. Its static
        # endpoint differs strongly, exposing any accidental [-1] anchoring.
        hx = np.vstack([self.hx, self.base + np.eye(12)[2]*3.])
        hr = self.residual+(hx-self.base)@self.live.T
        self.assertGreater(np.linalg.norm(hr[-1]), np.linalg.norm(self.residual))
        result = self.propose(history_initials=hx, history_residuals=hr, anchor_index=2)
        inferred = self.propose(history_initials=hx, history_residuals=hr)
        np.testing.assert_allclose(result['selected']['increment_12d'], inferred['selected']['increment_12d'])
        self.assertEqual(result['diagnostics']['anchor_history_index'],2)
        np.testing.assert_allclose(result['diagnostics']['history_scaled_initial_steps'],
                                   (np.diff(hx,axis=0)/INITIAL_SCALE).T)
        candidate=result['selected']; step=np.array(candidate['increment_12d'])
        np.testing.assert_allclose(candidate['static_predicted_residual_12d'],
                                   self.residual+self.static@step, atol=1e-12)
        np.testing.assert_allclose(candidate['predicted_residual_12d'],
                                   self.residual+self.live@step, atol=0.25)
        # Opposite-sign measured response still comes from retained history.
        self.assertGreater(step[0],0.)

    def test_explicit_anchor_and_ambiguous_inference_checks(self):
        for index in [-1,3,True,1.5,0]:
            with self.subTest(index=index),self.assertRaisesRegex(ValueError,'[Aa]nchor'):
                self.propose(anchor_index=index)
        hx=np.vstack([self.hx,self.base]); hr=np.vstack([self.hr,self.residual])
        with self.assertRaisesRegex(ValueError,'exactly one'):
            self.propose(history_initials=hx,history_residuals=hr)
        result=self.propose(history_initials=hx,history_residuals=hr,anchor_index=2)
        self.assertEqual(result['diagnostics']['anchor_history_index'],2)

    def test_invalid_and_nonfinite_inputs(self):
        for kwargs in [dict(alpha=0.),dict(alpha=np.nan),dict(max_position_step=-1),
                       dict(sigma=np.zeros(12)),dict(sigma=np.ones(11)),
                       dict(history_initials=self.hx[:2]),
                       dict(history_residuals=np.zeros((4,12))),
                       dict(residual=np.full(12,np.nan)),
                       dict(forward=lambda x:np.full(12,np.inf))]:
            with self.subTest(kwargs=list(kwargs)),self.assertRaises(ValueError):
                self.propose(**kwargs)

    def test_zero_residual_needs_no_static_evaluation(self):
        def forbidden(x):
            raise AssertionError('forward should not be evaluated')
        hr=self.hr.copy();hr[-1]=0.
        result=self.propose(residual=np.zeros(12),history_residuals=hr,forward=forbidden)
        self.assertEqual(result['status'],'zero_residual')
        self.assertIsNone(result['selected'])

    def test_selection_minimizes_worst_across_ridge_and_schema(self):
        report=self.propose()
        accepted=[c for c in report['candidates'] if c['accepted'] and c['passes_all_ridge_acceptance_criteria']]
        selected=report['selected']
        best=min(c['worst_across_ridge_residual_norm'] for c in accepted)
        self.assertEqual(selected['worst_across_ridge_residual_norm'],best)
        for c in accepted:
            self.assertTrue(0<=c['outside_history_fraction']<=1+1e-12)
            self.assertLess(c['max_abs_normalized'],5.)
            self.assertLess(c['predicted_residual_norm'],np.linalg.norm(self.residual))
            self.assertEqual(len(c['cross_ridge_predictions']),3)

    def test_selected_candidate_passes_all_ridge_models_not_only_its_own(self):
        # Unequally sampled directions expose regularization disagreement.
        rng=np.random.default_rng(0)
        static=np.eye(12)+rng.normal(0,.2,(12,12))
        live=static.copy();live[:,:2]+=rng.normal(0,2,(12,2))
        hx=np.zeros((3,12));hx[0,0]=1.;hx[1,1]=.03
        residual=rng.normal(0,2,12);residual[:2]=[5.,-4.]
        report=propose_candidates(np.zeros(12),residual,np.ones(12),hx,
                                  residual+hx@live.T,lambda x:static@x)
        excluded=[c for c in report['candidates'] if c['accepted'] and not c['passes_all_ridge_acceptance_criteria']]
        self.assertTrue(excluded)
        self.assertIsNotNone(report['selected'])
        self.assertTrue(report['selected']['passes_all_ridge_acceptance_criteria'])
        for c in excluded:
            self.assertIn('selection_excluded_reason',c)
            self.assertNotEqual(c['id'],report['selected']['id'])


class AnchorSelectionAndAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.residuals=np.zeros((4,12)); self.residuals[:,0]=[9.,7.,6.,6.5]
        self.sigma=np.ones(12)
        self.payload={'trials':[]}
        for i in range(4):
            predicted=np.zeros(12); predicted[0]=[0.,6.,5.,4.5][i]
            self.payload['trials'].append(dict(trial=f'trial{i+4:02d}',
                provenance=dict(reference_ic=f'/remote/trial{i+4:02d}.hdf5',reference_ic_sha256=f'hash{i}'),
                orbit=dict(shooting_diagnostics=dict(predicted_residual_12d=predicted),
                           shooting_base_ic=f'/remote/trial{i+3:02d}.hdf5',
                           shooting_residual_provenance=dict(reference_ic_sha256=f'hash{i-1}'))))

    def test_best_anchor_is_not_last_and_requires_both_identity_fields(self):
        i,report=select_anchor(self.payload,'/local/trial06.hdf5','hash2',self.residuals,self.sigma)
        self.assertEqual(i,2); self.assertTrue(report['history_order_preserved'])
        self.assertEqual(report['best_trials'],['trial06'])
        with self.assertRaisesRegex(ValueError,'不是'):
            select_anchor(self.payload,'trial07.hdf5','hash3',self.residuals,self.sigma)
        for name,sha in [('trial06.hdf5','hash3'),('renamed.hdf5','hash2')]:
            with self.assertRaisesRegex(ValueError,'SHA256'):
                select_anchor(self.payload,name,sha,self.residuals,self.sigma)

    def test_actual_prediction_loss_ratio_rejects_worsened_live_step(self):
        evaluations=evaluate_live_steps(self.payload,self.residuals,self.sigma)
        self.assertTrue(evaluations[0]['accepted_as_improvement'])
        last=evaluations[-1]
        self.assertFalse(last['accepted_as_improvement'])
        self.assertAlmostEqual(last['actual_loss_reduction'],.5*(6**2-6.5**2))
        self.assertAlmostEqual(last['actual_to_predicted_reduction_ratio'],(6**2-6.5**2)/(6**2-4.5**2))
        # Nonpositive predicted improvement cannot define a meaningful rho.
        self.payload['trials'][-1]['orbit']['shooting_diagnostics']['predicted_residual_12d'][0]=7
        self.assertIsNone(evaluate_live_steps(self.payload,self.residuals,self.sigma)[-1]['actual_to_predicted_reduction_ratio'])

    def test_branched_parent_uses_earlier_base_not_last_rejected_trial(self):
        branch=dict(trial='trial08',provenance=dict(reference_ic='/remote/trial08.hdf5',reference_ic_sha256='hash4'),
                    orbit=dict(shooting_base_ic='/local/trial06.hdf5',
                               shooting_residual_provenance=dict(reference_ic_sha256='hash2'),
                               shooting_diagnostics=dict(predicted_residual_12d=np.r_[4.5,np.zeros(11)])))
        self.payload['trials'].append(branch)
        residuals=np.vstack([self.residuals,np.r_[5.,np.zeros(11)]])
        self.assertEqual(resolve_parent_index(self.payload['trials'],4),2)
        last=evaluate_live_steps(self.payload,residuals,self.sigma)[-1]
        self.assertEqual(last['base_trial'],'trial06')
        self.assertAlmostEqual(last['actual_loss_reduction'],.5*(6**2-5**2))
        branch['orbit']['shooting_residual_provenance']['reference_ic_sha256']='hash3'
        with self.assertRaisesRegex(ValueError,'父轮次'):
            resolve_parent_index(self.payload['trials'],4)

    def test_probe_loss_is_measurement_outcome_not_rejection(self):
        diag=self.payload['trials'][-1]['orbit']['shooting_diagnostics']
        for method in ('live-probe','live-response-measurement-probe'):
            diag['method']=method
            report=evaluate_live_steps(self.payload,self.residuals,self.sigma)[-1]
            with self.subTest(method=method):
                self.assertTrue(report['is_measurement_probe'])
                self.assertEqual(report['method'],diag['method'])
                self.assertFalse(report['actual_loss_improved'])
                self.assertIsNone(report['accepted_as_improvement'])
                self.assertEqual(report['outcome_role'],'measurement_outcome_not_descent_acceptance')


class HistoryProvenanceTests(unittest.TestCase):
    def fixture(self):
        order=[f'{s}_{c}' for s in ('LMC','SMC') for c in ('x','y','z','vx','vy','vz')]
        trials=[]
        for i,parent_index in enumerate((None,0,1,2,2)):
            label=f'trial{i+4:02d}'
            initial=np.zeros(12); initial[i]=1.
            residual=np.zeros(12); residual[0]=[9.,7.,6.,6.5,5.][i]
            provenance=dict(schema_version=1,trial_label=label,model_key='same-model',
                measurement_signature='same-measurement',target_time_code=1.,
                stellar_tracer_policy='original-stars',target_covered=True,shooting_usable=True,
                reference_ic=f'/source/{label}.hdf5',reference_ic_sha256=f'ic{i}',residual_csv_sha256=f'csv{i}')
            orbit=dict(trial_label=label,model_key='same-model',source_sha256=['mw','lmc','smc'],
                rotation_matrices=[],units={},stage2_start_code=0.,stage2_target_code=1.,model_key_inputs={},
                output_ic_sha256=f'ic{i}',applied_initial_12d=initial.tolist(),nominal_initial_12d=np.zeros(12).tolist(),
                cumulative_offset_12d=initial.tolist())
            if parent_index is not None:
                parent=trials[parent_index]
                old=np.array(parent['orbit']['applied_initial_12d'])
                orbit.update(shooting_base_ic=parent['provenance']['reference_ic'],
                    previous_offset_12d=old.tolist(),this_iteration_increment_12d=(initial-old).tolist(),
                    manual_increment_12d=np.zeros(12).tolist(),shooting_residual_provenance=copy.deepcopy(parent['provenance']),
                    shooting_diagnostics=dict(base_initial_12d=old.tolist(),residual_12d=parent['residual'],sigma_12d=np.ones(12).tolist(),
                                              predicted_residual_12d=(residual*.8).tolist()))
            trials.append(dict(trial=label,orbit=orbit,provenance=provenance,sigma=np.ones(12).tolist(),
                observed=np.zeros(12).tolist(),residual=residual.tolist(),simulated=residual.tolist(),zscore=residual.tolist()))
        return dict(schema_version=1,coordinate_order=order,trials=trials)

    def test_verified_branch_keeps_original_order(self):
        payload=self.fixture();before=copy.deepcopy(payload)
        x,r,s,o=validate_history(payload)
        self.assertEqual(payload,before)
        self.assertEqual(resolve_parent_index(payload['trials'],4),2)
        np.testing.assert_array_equal(x,np.eye(12)[:5])
        self.assertEqual(evaluate_live_steps(payload,r,s)[-1]['base_trial'],'trial06')

    def test_branch_provenance_and_actual_increment_must_match_parent(self):
        for key in ('previous_offset_12d','this_iteration_increment_12d'):
            payload=self.fixture();payload['trials'][-1]['orbit'][key][0]+=.1
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_history(payload)
        for key in ('reference_ic_sha256','residual_csv_sha256','measurement_signature'):
            payload=self.fixture();payload['trials'][-1]['orbit']['shooting_residual_provenance'][key]='wrong'
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_history(payload)
        payload=self.fixture();payload['trials'][2],payload['trials'][3]=payload['trials'][3],payload['trials'][2]
        with self.assertRaisesRegex(ValueError,'父轮次'):
            validate_history(payload)


if __name__=='__main__':
    unittest.main()
