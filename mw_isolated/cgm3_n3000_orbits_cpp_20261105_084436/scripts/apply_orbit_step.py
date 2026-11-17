"""Produce one new CGM3 initial IC from a target-bound, independently re-derived alpha proposal."""
import argparse
import importlib.util
from pathlib import Path
import sys
import propose_alpha as model

def apply(proposal, trial):
    path=Path(proposal).resolve(strict=True)
    target=Path(trial).resolve()
    model.require(target.parent==model.ROOT and path==model.ROOT/(target.name+'_proposal.json'),
                  'Proposal/target names must belong to this campaign')
    q=model.read(path)
    model.require(q.get('version')=='cgm3-bounded-alpha-v1' and q.get('family_key')==model.FAMILY_KEY,
                  'Not a CGM3 coarse-alpha proposal')
    base,target=model.paths(q['base_run'],target)
    model.require(q['target_run']==str(target),'Proposal belongs to a different target trial')
    camp=model.read(model.ROOT/'campaign.json')
    family,_=model.family_of(base)
    model.require(camp.get('family_key')==model.FAMILY_KEY and camp.get('family_inputs')==family,
                  'Campaign family differs from frozen inputs')
    model.awaiting_failure(base)
    model.visible_only()
    before=model.sha(path)
    with model.exclusive_compute():
        fresh=model.make_proposal(base,target,q['alpha'],q['max_position_step_kpc'],q['max_velocity_step_kms'])
        model.require(fresh==q and model.sha(path)==before,'Proposal, live evidence, static recomputation or code changed')
        model.awaiting_failure(base)
        corepath=model.MW/'shooting_cpp_20261031_082656/apply_orbit_step.py'
        spec=importlib.util.spec_from_file_location('rigid_orbit_step',corepath)
        core=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core)
        return core.apply(path,target,model.ROOT/'campaign.json',target.name)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposal',required=True,type=Path)
    p.add_argument('--trial-dir',required=True,type=Path)
    a=p.parse_args()
    apply(a.proposal,a.trial_dir)

if __name__=='__main__': main()
