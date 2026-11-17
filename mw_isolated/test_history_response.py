"""Known-map tests, not live GIZMO validation."""
import json
import unittest
import numpy as np
from history_response import propose_candidates


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
            self.propose(history_initials=hx,history_residuals=np.tile(self.residual,(3,1)))
        hx[0,0]-=2; hx[1,0]-=1
        with self.assertRaisesRegex(ValueError,'independent'):
            self.propose(history_initials=hx)

    def test_latest_rows_must_match(self):
        hx=self.hx.copy();hx[-1,0]+=.01
        with self.assertRaisesRegex(ValueError,'Latest initial'):
            self.propose(history_initials=hx)
        hr=self.hr.copy();hr[-1,0]+=.01
        with self.assertRaisesRegex(ValueError,'Latest residual'):
            self.propose(history_residuals=hr)

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
        accepted=[c for c in report['candidates'] if c['accepted']]
        selected=report['selected']
        best=min(c['worst_across_ridge_residual_norm'] for c in accepted)
        self.assertEqual(selected['worst_across_ridge_residual_norm'],best)
        for c in accepted:
            self.assertTrue(0<=c['outside_history_fraction']<=1+1e-12)
            self.assertLess(c['max_abs_normalized'],5.)
            self.assertLess(c['predicted_residual_norm'],np.linalg.norm(self.residual))
            self.assertEqual(len(c['cross_ridge_predictions']),3)


if __name__=='__main__':
    unittest.main()
