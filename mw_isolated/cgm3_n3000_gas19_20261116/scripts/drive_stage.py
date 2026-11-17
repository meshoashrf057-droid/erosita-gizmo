"""Run one reviewed stage in the attached ubuntu terminal; never overwrite or restart."""
import argparse, fcntl, json, os, sys, threading
from pathlib import Path
import run_trial
from stage_safety import ROOT, compute_processes, memory, require, new_json

def sample_until_done(trial, mode, stop, stream):
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    while not stop.is_set():
        item={"utc":run_trial.utc(),"boot_id":boot,"meminfo_kib":memory(),"processes":compute_processes()}
        stream.write(json.dumps(item)+"\n"); stream.flush()
        stop.wait(5 if mode=="smoke" else 30)

def run(trial, mode):
    require(trial.parent==ROOT and os.isatty(0),"invalid terminal or trial")
    require(not (trial/mode/"status.json").exists(),"stage already attempted")
    with (ROOT/"status/compute.flock").open("a") as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (trial/mode/"resource_samples.jsonl").open("x") as samples:
            stop=threading.Event()
            monitor=threading.Thread(target=sample_until_done,args=(trial,mode,stop,samples))
            monitor.start()
            try:
                return run_trial.run(trial,mode)
            finally:
                stop.set(); monitor.join()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--trial-dir",type=Path,required=True)
    parser.add_argument("--mode",choices=["smoke","production"],required=True)
    args=parser.parse_args()
    sys.exit(run(args.trial_dir.resolve(),args.mode))
