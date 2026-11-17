#!/usr/bin/env python3
"""Hold the existing global compute lease across gas19 alpha native+validator+gate.
The original production_and_gate owns the campaign-local lease inside drive_stage.
Never take that local lease here: doing so would deadlock the unchanged driver.
"""
from pathlib import Path
import argparse,fcntl,subprocess,sys
sys.path.insert(0,str(Path(__file__).parent))
import gas19_alpha_review01 as model
from gas19_alpha_runtime_review01 import guard
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial-dir',required=True,type=Path);a=p.parse_args()
    trial=a.trial_dir.resolve(strict=True)
    model.require(trial==model.ROOT/'alpha01','Only this reviewed single alpha01')
    model.visible_only()
    with model.GLOBAL_LEASE.open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        model.require(not model.safety.compute_processes(),'Another GIZMO/MPI exists')
        guard(trial)
        command=['/home/xingdi/gizmo-runs/.venv/bin/python','-B','-u',
                 str(model.ROOT/'scripts/production_and_gate.py'),'--trial-dir',str(trial)]
        # No PIPE: preserve the actual visible PTY and original native/validator logs.
        return subprocess.run(command,cwd=model.ROOT).returncode
if __name__=='__main__':sys.exit(main())
