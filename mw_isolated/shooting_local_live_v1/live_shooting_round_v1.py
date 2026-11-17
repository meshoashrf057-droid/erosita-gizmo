#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""trial10 同基点 shooting：生成测量分支、验证、提案、导入、复核。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"pipeline"))
from local_live_evidence import (COORDS,SCHEMA,ANCHOR_SHA,arr,equal,
    sha256,paths,read_audit,read_evidence,same_evidence,validate_probe,validate_common,normalized_params,check_runtime_paths,pair_diagnostics)
from local_live_math import propose_step
from local_live_params import assess_base_param, assert_base_param_compatible, recover_base_param
from local_live_binary import select_binary, audit_binaries
from local_live_build_review import load_review, validate_review, assert_build_pair

VERSION = "local-live-round-v1.5-reviewed-t10-rebuild / 2026-10-23"
STATE_SCHEMA = "local-live-round-state-v2"
PROBES = [("trial11",0,.5),("trial12",1,.5),("trial13",7,.5)]


def write_new(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")


def verify_package():
    manifest=json.loads((HERE/"package_sha256.json").read_text())
    for name,expected in manifest.items():
        if sha256(HERE/name)!=expected:
            raise ValueError(f"脚本包文件已改变：{name}；请使用完整配套版本")


def run_cmd(command,env=None):
    import shlex
    print("执行："+shlex.join(map(str,command)),flush=True)
    subprocess.run(list(map(str,command)),check=True,env=env)


def current_settings(anchor,base_param,gizmo_executable=None,build_review=None):
    base_param=Path(base_param).expanduser().resolve(strict=True)
    assert_base_param_compatible(anchor,base_param)
    select_binary(anchor,gizmo_executable,build_review=build_review)
    return base_param


def probe_definitions():
    return [dict(trial=t,coordinate=COORDS[i],index=i,step=h) for t,i,h in PROBES]


def read_state(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema")!=STATE_SCHEMA or data.get("anchor_trial")!="trial10":
        raise ValueError("不是 t10-only 本轮 state JSON；不能复用旧版依赖历史文件的 state")
    if data["anchor_evidence"]["ic_sha256"]!=ANCHOR_SHA:
        raise ValueError("state 的 trial10 指纹不匹配")
    if data.get("history_dependency") is not False:
        raise ValueError("本轮必须独立于旧历史文件")
    if data.get("param_comparison_policy")!="effective-production-v1":
        raise ValueError("state 参数校验规则不是当前版本，请勿混用旧 state")
    if sha256(data["base_param"])!=data.get("base_param_sha256"):
        raise ValueError("本批实际参数文件在 prepare 后已改变；请保留冻结的参数副本")
    if data.get("probes")!=probe_definitions() or data.get("candidate_trial")!="trial14":
        raise ValueError("state 的探测方向/步长/轮次定义已改变")
    if data.get("target_max")!=2.8 or data.get("step_caps_kpc")!=[.5,.5,.5]:
        raise ValueError("state 的预测目标或限步已改变")
    if type(data.get("mpi_tasks")) is not int or data["mpi_tasks"]<1:
        raise ValueError("state 的 mpi-tasks 必须为正整数")
    a=read_evidence(data["work_dir"],"trial10")
    same_evidence(data["anchor_evidence"],a)
    build_review=validate_review(data.get("build_review"),a)
    selected=select_binary(a,data["execution_gizmo"]["path"],build_review=build_review)
    if selected!=data["execution_gizmo"]:
        raise ValueError("state 中记录的实际 GIZMO 与核验结果不同")
    current_settings(a,data["base_param"],selected["path"],build_review)
    return data,a


def verify_generated(state,trial,step,method="live-probe"):
    p=paths(state["work_dir"],trial)
    audit=read_audit(p["ic"])
    orbit=json.loads(p["orbit"].read_text())
    if audit!={k:v for k,v in orbit.items() if k!="output_ic_sha256"} or sha256(p["ic"])!=orbit["output_ic_sha256"]:
        raise ValueError(f"{trial}: IC/orbit 不一致")
    a=state["anchor_evidence"]
    for key in ("model_key","source_sha256"):
        if audit[key]!=a[key]:raise ValueError(f"{trial}: {key} 不一致")
    if audit["base_param_sha256"]!=state["base_param_sha256"]:
        raise ValueError(f"{trial}: 未使用本批已核对的实际参数副本")
    build_review=validate_review(state.get("build_review"),a)
    assert_build_pair(a["gizmo_sha256"],audit["gizmo"]["sha256"],build_review)
    if audit.get("build_review")!=build_review:
        raise ValueError(f"{trial}: 构建复核记录与本批不一致")
    if audit["gizmo"]!=state["execution_gizmo"]:
        raise ValueError(f"{trial}: 未使用本批指定的同构建 GIZMO 路径")
    if audit["trial_label"]!=trial or audit["shooting_diagnostics"]["method"]!=method:
        raise ValueError("生成轮次/方法错误")
    provenance=audit["shooting_residual_provenance"]
    if provenance["reference_ic_sha256"]!=a["ic_sha256"] or provenance["residual_csv_sha256"]!=a["residual_csv_sha256"]:
        raise ValueError("生成分支使用了错误的基点")
    equal(audit["this_iteration_increment_12d"],step,"生成增量")
    equal(audit["manual_increment_12d"],np.zeros(12),"手动增量")
    equal(audit["applied_initial_12d"],arr(a["applied_initial_12d"])+step,"实际初态")
    equal(audit["previous_offset_12d"],a["audit"]["cumulative_offset_12d"],"继承偏移")
    equal(audit["rotation_matrices"],a["rotation_matrices"],"盘面")
    equal([audit["stage2_start_code"],audit["stage2_target_code"]],[a["stage2_start_code"],a["target_time_code"]],"时间")
    if normalized_params(p["runtime_param"])!=a["runtime_param_normalized"]:
        raise ValueError(f"{trial}: production 参数改变")
    check_runtime_paths(p["runtime_param"],p["ic"],p["output"],Path(state["base_param"]).parent,require_hdf5_basename=True)
    if not (p["run"]/"smoke/run.sh").is_file() or not (p["run"]/"production/run.sh").is_file():
        raise ValueError("运行目录未完整生成；不覆盖，请检查该轮生成日志")
    return p


def verify_run_files(state,p):
    from satellite_relax_common import run_script
    a=state["anchor_evidence"];cwd=Path(state["base_param"]).parent
    production=normalized_params(p["runtime_param"])
    for mode in ("smoke","production"):
        folder=p["run"]/mode;param=folder/"stage2.param"
        check_runtime_paths(param,p["ic"],folder/"output",cwd,require_hdf5_basename=True)
        if mode=="smoke":
            smoke=normalized_params(param)
            exceptions={"TimeMax","TimeOfFirstSnapshot","TimeBetSnapshot","MaxSizeTimestep"}
            if {k:v for k,v in smoke.items() if k not in exceptions}!={k:v for k,v in production.items() if k not in exceptions}:
                raise ValueError("smoke 与 production 的物理设置不一致")
        with tempfile.TemporaryDirectory() as temp:
            expected=Path(temp)/"run.sh"
            run_script(expected,state["execution_gizmo"],param,cwd,2,folder/"gizmo_run.log",state["mpi_tasks"])
            if expected.read_bytes()!=(folder/"run.sh").read_bytes():
                raise ValueError(f"{folder/'run.sh'} 已改变，拒绝执行不匹配的 flag/输入/输出命令")


def merge_command(state,trial,mode):
    a=state["anchor_evidence"]
    command=[sys.executable,HERE/"pipeline/merge_relaxed_mw_lmc_smc_local_live_v1.py","next",
        "--base-param",state["base_param"],"--gizmo-executable",state["execution_gizmo"]["path"],
        "--shooting-base-ic",a["ic_path"],
        "--shooting-residual-csv",a["residual_csv_path"],"--shooting-method",mode,
        "--require-shooting-provenance","--mpi-tasks",str(state["mpi_tasks"]),
        "--output",paths(state["work_dir"],trial)["ic"],"--trial-label",trial]
    if state.get("build_review") is not None:
        command.extend(["--reviewed-build",state["build_review"]["path"]])
    return command


def assert_unused(p):
    for name in (p["ic"],p["orbit"],p["run"],p["ic"].with_suffix(".hdf5.partial")):
        if name.exists():raise ValueError(f"拒绝覆盖：{name}")


def prepare(args):
    root=args.work_dir.expanduser().resolve(strict=True)
    # 只读取当前 trial10；不存在的 trial11 在本次作为新分支生成。
    ensure_validation(root,"trial10")
    a=read_evidence(root,"trial10")
    if a["ic_sha256"]!=ANCHOR_SHA:raise ValueError("不是已经审计的 trial10 IC")
    review_path=getattr(args,"reviewed_build",None)
    build_review=load_review(review_path,a) if review_path else None
    selected=select_binary(a,getattr(args,"gizmo_executable",None),build_review=build_review)
    base=current_settings(a,args.base_param,selected["path"],build_review)
    if build_review:
        print("已采用本次专门复核的 t10 重编译记录；保留旧基准 SHA，冻结本批实际新 SHA。",flush=True)
        print("同源码提交/显式选项不等于终点逐位相同；复用旧终点以数值差异可忽略为假设，候选仍须 live 验证。",flush=True)
    if args.mpi_tasks<1:raise ValueError("mpi-tasks 必须为正")
    state=dict(schema=STATE_SCHEMA,version=VERSION,work_dir=str(root),
        anchor_trial="trial10",anchor_evidence=a,
        base_param=str(base),base_param_sha256=sha256(base),mpi_tasks=args.mpi_tasks,history_dependency=False,
        param_comparison_policy="effective-production-v1",
        execution_gizmo=selected,
        build_review=build_review,
        probes=probe_definitions(),
        target_max=2.8,step_caps_kpc=[.5,.5,.5],candidate_trial="trial14")
    state_path=args.state.expanduser().resolve()
    if args.resume:
        if not state_path.exists() or json.loads(state_path.read_text())!=state:
            raise ValueError("resume 需要同一份完整 state 和未改变的输入")
    else:
        if state_path.exists():raise ValueError("state 已存在；未完成生成可用 --resume")
        # 所有新分支先做覆盖检查，再写 state 和任何 IC。
        for t,_,_ in PROBES:assert_unused(paths(root,t))
        assert_unused(paths(root,"trial14"))
        write_new(state_path,state)
    for trial,index,h in PROBES:
        p=paths(root,trial);step=np.zeros(12);step[index]=h
        if args.resume and p["ic"].exists():
            verify_generated(state,trial,step)
            print(f"已核对 {trial}，跳过已完成的 IC 生成。",flush=True)
            continue
        assert_unused(p)
        run_cmd(merge_command(state,trial,"live-probe")+["--probe-coordinate",COORDS[index],"--probe-step",str(h)])
        verify_generated(state,trial,step)
    print(f"\n准备完成：{state_path}")
    for trial,index,h in PROBES:
        print(f"{trial} = trial10 + {COORDS[index]} {h:g} kpc")
    print("尚未运行 GIZMO。三条独立响应测量都从 trial10 出发，不累计修正。")
    print("预算：新增 3 次全程 GIZMO 测量；提案可用后再做 1 次候选验证，不保证本批收敛。")


def run_trial(args):
    state,a=read_state(args.state)
    if args.trial in ("trial11","trial12","trial13"):
        _,index,h=next(p for p in PROBES if p[0]==args.trial)
        step=np.zeros(12);step[index]=h;method="live-probe"
    elif args.trial=="trial14":
        proposal=json.loads(Path(args.proposal).read_text())
        step=arr(proposal["proposed_step_12d"]);method="local-live-joint"
        from local_live_import import load_local_live_step
        load_local_live_step(args.proposal,a["applied_initial_12d"],a["residual_12d"],a["sigma_12d"],
            a["ic_path"],a["ic_sha256"],a["residual_csv_path"],a["residual_csv_sha256"],a["model_key"])
    else:raise ValueError("仅运行本轮 trial11/12/13/14")
    p=verify_generated(state,args.trial,step,method)
    verify_run_files(state,p)
    for mode in ("smoke","production"):
        folder=p["run"]/mode
        if any((folder/"output").iterdir()) or (folder/"gizmo_run.log").exists():
            raise ValueError(f"{mode} 已有输出/日志；拒绝覆盖。中断续跑请按 GIZMO restart 流程处理。")
    # 顺序运行，smoke 失败即停止；production 仍从原 IC 重新启动，不接 smoke 终态。
    env=dict(os.environ,NPROC=str(state["mpi_tasks"]))
    run_cmd(["bash",p["run"]/"smoke/run.sh"],env=env)
    import h5py
    snapshots=[]
    for path in (p["run"]/"smoke/output").glob("snapshot_*.hdf5"):
        with h5py.File(path,"r") as f:snapshots.append((float(f["Header"].attrs["Time"]),path))
    if not snapshots:raise ValueError("smoke 没有快照，停止 production")
    first=min(snapshots,key=lambda x:x[0])[1]
    run_cmd([sys.executable,HERE/"pipeline/merge_relaxed_mw_lmc_smc_local_live_v1.py","verify",
        "--reference-ic",p["ic"],"--snapshot",first])
    run_cmd(["bash",p["run"]/"production/run.sh"],env=env)


def ensure_validation(root,trial):
    p=paths(root,trial)
    missing=[str(p[k]) for k in ("ic","orbit","runtime_param") if not p[k].is_file()]
    if not p["output"].is_dir() or not any(p["output"].glob("snapshot_*.hdf5")):
        missing.append(str(p["output"])+"/snapshot_*.hdf5")
    if missing:
        raise ValueError(f"{trial}: 缺少本轮输入/已有演化结果：\n  "+"\n  ".join(missing))
    folder=p["csv"].parent
    if folder.exists():
        required=(p["csv"],p["csv"].with_suffix(".provenance.json"))
        missing=[str(f) for f in required if not f.is_file()]
        if missing:
            raise ValueError(f"{trial}: 验证目录不完整，缺少：\n  "+"\n  ".join(missing)
                +f"\n请先将 {folder} 改名保留，再执行本脚本 validate --work-dir {Path(root).resolve()} --trial {trial}；不需要重跑 GIZMO。")
        print(f"使用已有验证目录，随后严格检查：{folder}",flush=True)
        return
    run_cmd([sys.executable,HERE/"pipeline/validation_dashboard_sat_6d_gaswake_v3.py",
        "--snapshot-dir",p["output"],"--reference-ic",p["ic"],"--stellar-tracers","original-stars",
        "--output-dir",folder])


def validate(args):
    root=args.work_dir.expanduser().resolve(strict=True)
    ensure_validation(root,args.trial)
    evidence=read_evidence(root,args.trial)
    if args.trial=="trial10" and evidence["ic_sha256"]!=ANCHOR_SHA:
        raise ValueError("不是已经审计的 trial10 IC")
    z=arr(evidence["residual_12d"])/arr(evidence["sigma_12d"])
    print(f"{args.trial} 验证通过来源检查：max|r/sigma|={np.max(np.abs(z)):.6f}，norm={np.linalg.norm(z):.6f}")
    pair=pair_diagnostics(evidence["observed_12d"],evidence["residual_12d"])
    print(f"LMC–SMC 间距：{pair['simulated_separation_kpc']:.6f} kpc；目标：{pair['target_separation_kpc']:.6f} kpc")


def checked_anchor(root):
    ensure_validation(root,"trial10")
    a=read_evidence(root,"trial10")
    if a["ic_sha256"]!=ANCHOR_SHA:
        raise ValueError("不是已经审计的 trial10 IC")
    return a


def param_check(args):
    a=checked_anchor(args.work_dir.expanduser().resolve(strict=True))
    report=assess_base_param(a,args.base_param)
    print("原 base-param 字节一致：",report["byte_identical"])
    print("生成后的 production 参数与 t10 一致：",report["compatible"])
    for row in report["differences"]:
        print(f"  {row['key']}: t10={row['anchor']!r}；当前生成值={row['candidate']!r}")
    if args.output:
        write_new(args.output.expanduser().resolve(),report)
    if not report["compatible"]:
        raise ValueError("有效运行参数不同；可用 recover-param 从 t10 留存的 production/stage2.param 生成专用副本")
    print("无需恢复旧注释/格式；有效设置一致。未修改参数或启动 GIZMO。")


def recover_param(args):
    a=checked_anchor(args.work_dir.expanduser().resolve(strict=True))
    report=recover_base_param(a,args.output.expanduser().resolve())
    print("已从 t10 留存的 production/stage2.param 建立兼容的参数副本：",args.output.expanduser().resolve())
    print("原 mw_isolated.param、t10 参数和快照保持不变；未启动 GIZMO。")
    return report


def binary_check(args):
    root=args.work_dir.expanduser().resolve(strict=True)
    ic=paths(root,"trial10")["ic"]
    audit=read_audit(ic)
    if sha256(ic)!=ANCHOR_SHA or audit.get("trial_label")!="trial10":
        raise ValueError("不是已经审计的 trial10 IC")
    anchor=dict(gizmo_path=audit["gizmo"]["path"],gizmo_sha256=audit["gizmo"]["sha256"])
    report=audit_binaries(anchor,args.search_dir,args.candidate)
    if report["exact_match_paths"]:
        report["suggested_prepare_command"]=[sys.executable,str(HERE/"live_shooting_round_v1.py"),"prepare",
            "--work-dir",str(root),"--base-param",str(root/"shooting_t10_runtime.param"),
            "--gizmo-executable",report["exact_match_paths"][0],"--mpi-tasks","6"]
    write_new(args.output.expanduser().resolve(),report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print("报告：",args.output.expanduser().resolve())
    print("仅检查文件，没有执行、替换或重新编译 GIZMO。")
    if report["exact_match_paths"]:
        import shlex
        print("找到可执行的同构建副本；使用已恢复参数、6 个 MPI 进程的准备指令：")
        print(shlex.join(report["suggested_prepare_command"]))
    else:
        print("本次搜索未找到可直接使用的匹配程序。先查看报告，不要修改 t10 指纹或直接用新构建减去旧终点。")
    print("这里只生成/核对诊断文件；没有运行 GIZMO，没有修改 IC 或快照。")


def propose(args):
    state,a=read_state(args.state)
    out=args.output.expanduser().resolve()
    if out.exists():raise ValueError("拒绝覆盖已有提案 JSON")
    evidence=[];columns=[]
    for trial,index,h in PROBES:
        ensure_validation(state["work_dir"],trial)
        e=read_evidence(state["work_dir"],trial)
        columns.append(validate_probe(a,e,index,h,build_review=state.get("build_review")));evidence.append(e)
        if e["gizmo_sha256"]!=state["execution_gizmo"]["sha256"] or e["gizmo_path"]!=state["execution_gizmo"]["path"]:
            raise ValueError(f"{trial}: 不属于本批冻结的 GIZMO 执行文件")
    J=np.column_stack(columns);sigma=arr(a["sigma_12d"]);r=arr(a["residual_12d"])
    report=propose_step(r/sigma,J,np.array(state["step_caps_kpc"]),target_max=state["target_max"])
    step=np.zeros(12);step[[0,1,7]]=report["delta"]
    pred=r+sigma*(J@np.asarray(report["delta"]))
    result=dict(schema=SCHEMA,version=VERSION,coordinate_order=COORDS,
        base_trial="trial10",shooting_model_key=a["model_key"],base_ic_path=a["ic_path"],
        base_ic_sha256=a["ic_sha256"],source_residual_csv_path=a["residual_csv_path"],
        source_residual_csv_sha256=a["residual_csv_sha256"],base_initial_12d=a["applied_initial_12d"],
        current_residual_12d=a["residual_12d"],sigma_12d=a["sigma_12d"],
        proposed_step_12d=step.tolist(),candidate_initial_12d=(arr(a["applied_initial_12d"])+step).tolist(),
        predicted_residual_12d=pred.tolist(),measured_response_normalized_12x3=J.tolist(),
        anchor_evidence=a,probe_evidence=evidence,history_dependency=False,
        execution_base_param_path=state["base_param"],execution_base_param_sha256=sha256(state["base_param"]),
        param_comparison_policy="effective-production-v1",
        execution_gizmo=state["execution_gizmo"],
        build_review=state.get("build_review"),
        target_max=state["target_max"],
        base_pair_diagnostics=pair_diagnostics(a["observed_12d"],r),
        predicted_pair_diagnostics=pair_diagnostics(a["observed_12d"],pred),
        step_caps_kpc=state["step_caps_kpc"],alpha_applied=False,solver_report=report,
        notes=["三个方向均为 trial10 附近单边有限差分；不是完整 12x12 live Jacobian。",
            "尚未测量正反方向曲率和重复性；2.8 是优化缓冲目标，不是已知统计置信界。",
            "原观测 sigma 与实际逐分量 3sigma 验收未改变。",
            "不会直接修改终态卫星位置、速度或 CGM；候选须从原弛豫末态重新演化。"])
    if state.get("build_review"):
        result["notes"].append("使用经复核的同提交/显式选项重编译；旧终点与新构建的数值等价未实测，响应可能包含构建差异。")
    write_new(out,result)
    print(f"\n结果：{out}\n基点 max|r/sigma|={report['base_max_abs']:.6f}")
    print(f"预测 max|r/sigma|={report['projected_max_abs']:.6f}；norm={report['projected_l2']:.6f}")
    print("本轮拟议 [LMC_x,LMC_y,SMC_y] 增量(kpc)：",report["delta"])
    print("响应秩/条件数：",report["rank"],report["condition_number"])
    pair=result["predicted_pair_diagnostics"]
    print(f"预测 LMC–SMC 间距={pair['simulated_separation_kpc']:.6f} kpc；目标={pair['target_separation_kpc']:.6f} kpc（派生诊断，不重复加权）")
    if report["ready_for_live_validation"]:
        print("可生成 trial14 候选；尚未 live 通过。使用 apply 后再 run --trial trial14。")
    else:
        print("当前证据下不生成 trial14。原因：",", ".join(report["reasons"]))
        print("保留本 JSON；不要调大容差或强行放宽限步。需要审查新增响应后决定是否补速度方向或转入 AGN 探索。")


def apply(args):
    state,a=read_state(args.state)
    from local_live_import import load_local_live_step
    step,_=load_local_live_step(args.proposal,a["applied_initial_12d"],a["residual_12d"],a["sigma_12d"],
        a["ic_path"],a["ic_sha256"],a["residual_csv_path"],a["residual_csv_sha256"],a["model_key"])
    assert_unused(paths(state["work_dir"],"trial14"))
    run_cmd(merge_command(state,"trial14","local-live-joint")+["--shooting-proposal-json",Path(args.proposal).resolve()])
    verify_generated(state,"trial14",step,"local-live-joint")
    print("trial14 IC 已生成，增量只应用一次。尚未运行 GIZMO。")


def review(args):
    state,a=read_state(args.state);out=args.output.expanduser().resolve()
    if out.exists():raise ValueError("拒绝覆盖 review JSON")
    proposal=json.loads(Path(args.proposal).read_text())
    from local_live_import import load_local_live_step
    step,_=load_local_live_step(args.proposal,a["applied_initial_12d"],a["residual_12d"],a["sigma_12d"],
        a["ic_path"],a["ic_sha256"],a["residual_csv_path"],a["residual_csv_sha256"],a["model_key"])
    verify_generated(state,"trial14",step,"local-live-joint")
    ensure_validation(state["work_dir"],"trial14")
    actual=read_evidence(state["work_dir"],"trial14");validate_common(a,actual,build_review=state.get("build_review"))
    sigma=arr(a["sigma_12d"]);z=arr(actual["residual_12d"])/sigma
    zb=arr(a["residual_12d"])/sigma;zp=arr(proposal["predicted_residual_12d"])/sigma
    bm,am,pm=[float(np.max(np.abs(v))) for v in (zb,z,zp)]
    ratio=(bm-am)/(bm-pm) if bm-pm>1e-10 else None
    passed=bool(np.all(np.abs(z)<=3.0))
    accepted=bool(am<bm and ratio is not None and ratio>=.1)
    base_pair=pair_diagnostics(a["observed_12d"],a["residual_12d"])
    actual_pair=pair_diagnostics(actual["observed_12d"],actual["residual_12d"])
    report=dict(schema="local-live-review-v2",anchor_evidence=a,candidate_evidence=actual,
        build_review=state.get("build_review"),execution_gizmo=state["execution_gizmo"],
        base_pair_diagnostics=base_pair,
        predicted_pair_diagnostics=pair_diagnostics(a["observed_12d"],proposal["predicted_residual_12d"]),
        actual_pair_diagnostics=actual_pair,
        pair_separation_absolute_residual_improved=bool(abs(actual_pair["separation_residual_kpc"])<abs(base_pair["separation_residual_kpc"])),
        proposal_path=str(Path(args.proposal).resolve()),proposal_sha256=sha256(args.proposal),
        base_max_abs=bm,predicted_max_abs=pm,actual_max_abs=am,
        actual_to_predicted_max_reduction_ratio=ratio,actual_l2=float(np.linalg.norm(z)),
        actual_normalized_residual=z.tolist(),prediction_error_sigma=(z-zp).tolist(),
        orbit_componentwise_3sigma_pass=passed,accepted_as_shooting_improvement=accepted,
        status="orbit_pass" if passed else ("improved_but_not_passed" if accepted else "keep_trial10"),
        note="仅轨道逐分量验收；overall pass、CGM/气泡科学有效性仍看完整 validator 和科学诊断。")
    write_new(out,report)
    print(f"\n实际 max|r/sigma|={am:.6f}；12D norm={np.linalg.norm(z):.6f}")
    print("轨道逐分量 3sigma：",passed,"；预测收益比：",ratio)
    print(f"实际 LMC–SMC 间距={actual_pair['simulated_separation_kpc']:.6f} kpc；目标={actual_pair['target_separation_kpc']:.6f} kpc")
    print("结论：",report["status"],"；报告：",out)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("prepare",help="只依赖 trial10；生成 trial11/12/13，不启动 GIZMO")
    q.add_argument("--work-dir",type=Path,default=Path("."))
    q.add_argument("--base-param",type=Path,default=Path("mw_isolated.param"))
    q.add_argument("--mpi-tasks",type=int,default=6)
    q.add_argument("--gizmo-executable",type=Path,help="允许指定与 t10 完全相同 SHA 的备份程序；不覆盖当前 GIZMO")
    q.add_argument("--reviewed-build",type=Path,help="本次已核对的 t10 重编译记录；仅接受指定的旧/新 SHA，不是任意跳过检查")
    q.add_argument("--resume",action="store_true")
    q=sub.add_parser("param-check",help="只读比较当前参数生成的生产设置与 t10，列出实际差异")
    q.add_argument("--work-dir",type=Path,default=Path("."))
    q.add_argument("--base-param",type=Path,default=Path("mw_isolated.param"))
    q.add_argument("--output",type=Path,help="可选差异 JSON；拒绝覆盖")
    q=sub.add_parser("recover-param",help="从 t10 留存的 production 参数创建专用副本，不需要原 base-param 备份")
    q.add_argument("--work-dir",type=Path,default=Path("."))
    q.add_argument("--output",type=Path,default=Path("shooting_t10_runtime.param"))
    q=sub.add_parser("binary-check",help="只读核对当前 GIZMO 并搜索相同 SHA 的备份，不执行程序")
    q.add_argument("--work-dir",type=Path,default=Path("."))
    q.add_argument("--search-dir",type=Path,action="append",default=[],help="递归搜索文件名含 gizmo 的 ELF 文件；可重复")
    q.add_argument("--candidate",type=Path,action="append",default=[],help="检查指定的备份文件；可重复")
    q.add_argument("--output",type=Path,default=Path("gizmo_binary_audit_t10.json"))
    q=sub.add_parser("validate",help="从现有快照生成/核对诊断；trial10 无需 state")
    q.add_argument("--work-dir",type=Path,default=Path("."))
    q.add_argument("--trial",choices=("trial10","trial11","trial12","trial13","trial14"),required=True)
    q=sub.add_parser("run",help="顺序运行该轮 smoke 和 production")
    q.add_argument("--trial",choices=("trial11","trial12","trial13","trial14"),required=True)
    q.add_argument("--proposal",default="trial14_t10_only_proposal.json")
    q=sub.add_parser("propose",help="验证本次三条 probe 并计算有界联合候选")
    q.add_argument("--output",type=Path,default=Path("trial14_t10_only_proposal.json"))
    q=sub.add_parser("apply",help="校验合格提案并生成 trial14 IC")
    q.add_argument("--proposal",default="trial14_t10_only_proposal.json")
    q=sub.add_parser("review",help="验证 trial14 的实际改善和 3sigma")
    q.add_argument("--proposal",default="trial14_t10_only_proposal.json")
    q.add_argument("--output",type=Path,default=Path("trial14_t10_only_review.json"))
    for q in sub.choices.values():
        q.add_argument("--state",type=Path,default=Path("local_live_round_t10_only.json"))
    args=p.parse_args(argv);print("脚本版本：",VERSION,flush=True)
    verify_package()
    globals()[{"run":"run_trial","param-check":"param_check","recover-param":"recover_param","binary-check":"binary_check"}.get(args.command,args.command)](args)


if __name__=="__main__":
    try:main()
    except (ValueError,OSError,KeyError,RuntimeError,subprocess.CalledProcessError) as exc:
        print("错误：",exc,file=sys.stderr);sys.exit(1)
