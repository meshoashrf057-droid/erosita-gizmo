"""Read-only revalidation of endpoints, each bound to its own CGM3 initial IC and measurements."""
from pathlib import Path
import json
import h5py
import numpy as np
import gate_endpoint as gate
from propose_alpha import ROOT,FAMILY_KEY,family_of,expected_parameters,require,read,sha

def verify_evidence(base):
    frozen,audit=family_of(base)
    require(base.is_dir() and base.resolve().parent==ROOT,'Endpoint must belong to current campaign')
    g=read(base/'endpoint_review/gate_result.json')
    require(g['state']=='orbit_failed' and g['model_key']==FAMILY_KEY and Path(g['run_directory']).resolve()==base,
            'Only a completed unsuccessful exact-family gate is valid')
    status_path=base/'production/status.json'
    status=read(status_path)
    require(status.get('state') in ('completed','simulation_completed') and status.get('simulation_returncode')==0,
            "Simulation did not finish successfully")
    if status['state']=='completed':
        require(status.get('returncode')==0 and status.get('wrapper_returncode',0)==0 and
                status.get('orbit_validation_returncode',0)==0,"Completion status unsuccessful")
    ic_sha,side,reference=gate.bind_run(base,status)
    require(side==audit and ic_sha==g['reference_ic_sha256'],"IC bound to previous gate changed")
    expected_parameters(frozen,base)
    snapshots=gate.snapshot_evidence(base,status)
    folder=Path(g['residual_csv']).resolve().parent
    require(folder in (base/'orbit_validation',base/'endpoint_review/validation'),"Unbound validation directory")
    paths_by_name,decision,d12=gate.verify_products(base,folder,reference,ic_sha,audit,snapshots,gate.semantic_functions())
    _,end,_=snapshots[-1]
    file_hash={str(p):sha(p) for p in paths_by_name.values()}
    require(file_hash==g['product_hashes'] and sha(status_path)==g['production_status_sha256'] and
            sha(end)==g['endpoint_snapshot']['sha256'] and str(end)==g['endpoint_snapshot']['path'],
            "Run, endpoint or validator artifacts changed")
    require(not decision['orbit_pass'] and np.isclose(d12,g['actual_D12'])
            and np.isclose(decision['max_abs_sigma'],g['max_abs_sigma']),"Actual gate and published scores differ")
    require(sha(gate.VALIDATOR)==g['validator_sha256']==gate.VALIDATOR_SHA,"Validator changed")
    src=[Path(p).resolve(strict=True) for p in audit['source_files']]
    require([sha(p) for p in src]==frozen['source_sha256'],"Source snapshot hash mismatch")
    require(sha(Path(audit['gizmo']['path']))==frozen['gizmo_sha256'],"GIZMO changed")
    with h5py.File(base/'orbit_000.hdf5','r') as f:
        embedded=json.loads(f['MergeProvenance/AuditJSON'].asstr()[()])
        require(embedded=={k:v for k,v in audit.items() if k!='output_ic_sha256'},"Embedded audit differs")
        orbit=f['MagellanicOrbitReference']
        actual=np.r_[tuple(orbit[s+'_AppliedInitial'+q][:] for s in ('LMC','SMC') for q in ('Position_kpc','Velocity_kms'))]
        require(np.array_equal(actual,audit['applied_initial_12d']),"Embedded parent state differs")
    require(np.array_equal(g['residual_12d'],np.concatenate([decision['values'][sat]['residual'] for sat in ('lmc','smc')]))
            and np.array_equal(g['normalized_residual_12d'],decision['normalized_residual_12d'])
            and g['measurement_signature']==frozen['measurement_signature'] and
            g['residual_csv_sha256']==sha(Path(g['residual_csv'])),'Gate residual vector/CSV or measurement changed')
    return frozen,audit,g,decision

