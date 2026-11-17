#!/usr/bin/env python3
"""同一基点 live shooting 的只读证据检查。不会修改源文件或目标容差。"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import re
import h5py
import numpy as np
from local_live_build_review import validate_review, assert_build_pair

COORDS = [f"{s}_{c}" for s in ("LMC", "SMC") for c in ("x", "y", "z", "vx", "vy", "vz")]
SCHEMA = "local-live-proposal-v2"
ANCHOR_SHA = "914ca086c68e1e5d04d6aff53ee3284cd67092241e330bd5ee3dd6c9224a9579"
SIGNATURE = dict(algorithm="fixed-original-star-identities-shrinking-core-v4",
    tracer_policy="original-stars",
    identity_fields=["ParticleIDs", "ParticleChildIDsNumber", "ParticleIDGenerationNumber"],
    host_frame="BH-position+MW-inner10kpc-mass-weighted-star-velocity-v3",
    lmc_core_radius_kpc=8.0, smc_core_radius_kpc=3.0)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(data)
    return h.hexdigest()


def arr(values, n=12):
    a = np.asarray(values, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError(f"需要 {n} 个有限数值")
    return a


def equal(a, b, label, atol=1e-7):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,rtol=0,atol=atol):
        raise ValueError(label + " 不一致")


def stem(trial):
    if not re.fullmatch(r"trial\d+", trial):
        raise ValueError("trial 必须形如 trial12")
    return f"mw_lmc_smc_relaxed_{trial}_000"


def paths(root, trial, validation_subdir="magellanic_validation_fixed"):
    root = Path(root).expanduser().resolve()
    if Path(validation_subdir).name != validation_subdir or validation_subdir in (".", ".."):
        raise ValueError("验证目录必须是单个目录名")
    name = stem(trial)
    output = root/(name+"_run")/"production"/"output"
    return dict(ic=root/(name+".hdf5"), orbit=root/(name+".orbit.json"),
        output=output, csv=output/validation_subdir/"magellanic_present_6d.csv",
        runtime_param=output.parent/"stage2.param", run=root/(name+"_run"))


def read_audit(ic):
    with h5py.File(ic, "r") as f:
        return json.loads(f["MergeProvenance/AuditJSON"].asstr()[()])


def read_params(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("%",1)[0].split("#",1)[0].strip()
        if not line:
            continue
        fields = line.split(None,1)
        if len(fields) != 2 or fields[0] in result:
            raise ValueError(f"重复/无效 param 行：{line}")
        result[fields[0]] = fields[1].strip()
    return result


def normalized_params(path):
    # 仅不同分支的输入输出路径可不同；时间步、物理量、冷却文件等仍严格比较。
    ignored = {"InitCondFile", "OutputDir", "RestartFile", "SnapshotFileBase"}
    out = {}
    for k,v in read_params(path).items():
        if k in ignored:
            continue
        try:
            x = float(v)
        except ValueError:
            out[k] = v
        else:
            if not np.isfinite(x):
                raise ValueError("param 含非有限数值")
            out[k] = x
    return out


def check_runtime_paths(param_path, ic_path, output_dir, working_dir, *, require_hdf5_basename=False):
    """按运行 cwd 解析路径；HDF5 输入前缀由 GIZMO 补 .hdf5。

    历史记录兼容完整文件名，但新生成/执行的参数必须使用无后缀前缀。
    这里不通过尝试不同工作目录、放宽轮次或修改文件来绕过检查。
    """
    values=read_params(param_path)
    cwd=Path(working_dir).expanduser().resolve()
    for key,expected in (("InitCondFile",ic_path),("OutputDir",output_dir)):
        raw=values.get(key,"")
        if not raw:
            raise ValueError(f"{param_path}: 缺少 {key}")
        actual=Path(raw).expanduser()
        if not actual.is_absolute():actual=cwd/actual
        actual=actual.resolve()
        expected=Path(expected).expanduser().resolve()
        ok=actual==expected
        candidates=[actual]
        if key=="InitCondFile" and values.get("ICFormat","3")=="3":
            # 必须先给用户原始前缀补后缀，再解析软链接。
            prefix=Path(raw).expanduser()
            if not prefix.is_absolute():prefix=cwd/prefix
            suffixed=Path(str(prefix)+".hdf5").resolve()
            split_first=Path(str(prefix)+".0.hdf5")
            candidates.append(suffixed)
            if split_first.exists():
                raise ValueError(f"{param_path}: InitCondFile 存在优先读取的分片 {split_first}；本轮要求已审核的单文件 IC")
            if actual==expected and suffixed.exists():
                raise ValueError(f"{param_path}: InitCondFile 完整路径与追加 .hdf5 后的文件有歧义：{suffixed}")
            ok=suffixed==expected or (actual==expected and not require_hdf5_basename)
        if not ok:
            details="\n  ".join((f"原值={raw}",f"运行目录={cwd}",
                "解析候选="+", ".join(map(str,candidates)),f"本轮期望={expected}"))
            hint="；新运行的 HDF5 InitCondFile 必须去掉 .hdf5 后缀" if key=="InitCondFile" and require_hdf5_basename else ""
            raise ValueError(f"{param_path}: {key} 未指向本轮文件/目录{hint}\n  {details}")
    if values.get("SnapshotFileBase")!="snapshot" or values.get("RestartFile")!="restart":
        raise ValueError("SnapshotFileBase/RestartFile 与本轮定义不同")


def read_evidence(work_dir, trial, validation_subdir="magellanic_validation_fixed"):
    p = paths(work_dir, trial, validation_subdir)
    sidecar = p["csv"].with_suffix(".provenance.json")
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    orbit = json.loads(p["orbit"].read_text(encoding="utf-8"))
    audit = read_audit(p["ic"])
    ic_hash, csv_hash = sha256(p["ic"]), sha256(p["csv"])
    if audit != {k:v for k,v in orbit.items() if k != "output_ic_sha256"}:
        raise ValueError(f"{trial}: HDF5 AuditJSON 与 orbit.json 不一致")
    if orbit.get("output_ic_sha256") != ic_hash or provenance.get("reference_ic_sha256") != ic_hash:
        raise ValueError(f"{trial}: 实际 IC 指纹不一致")
    if provenance.get("residual_csv_sha256") != csv_hash:
        raise ValueError(f"{trial}: 残差 CSV 指纹不一致")
    if provenance.get("schema_version") != 1 or provenance.get("trial_label") != trial or audit.get("trial_label") != trial:
        raise ValueError(f"{trial}: provenance/trial 标识不一致")
    if Path(provenance.get("reference_ic", "")).name != p["ic"].name:
        raise ValueError(f"{trial}: reference IC 文件名错误")
    if provenance.get("measurement_signature") != SIGNATURE or provenance.get("stellar_tracer_policy") != "original-stars":
        raise ValueError(f"{trial}: 本轮必须使用原有 original-stars 测量定义")
    if not provenance.get("model_key") or provenance["model_key"] != audit["model_key"]:
        raise ValueError(f"{trial}: 模型不一致")
    if provenance.get("target_covered") is not True or provenance.get("shooting_usable") is not True:
        raise ValueError(f"{trial}: 终点不可用于 shooting")
    if provenance.get("evaluation_method") != "exact target snapshot":
        raise ValueError(f"{trial}: 需要目标时刻的实际快照，不接受邻近帧或插值")
    target = float(audit["stage2_target_code"])
    equal([provenance["target_time_code"]], [target], "目标时间")
    sources = provenance.get("target_snapshot_sources", [])
    if len(sources) != 1 or sources[0].get("weight") != 1.0:
        raise ValueError(f"{trial}: 需要单个实际终点")
    target_file = Path(sources[0]["file"]).expanduser().resolve(strict=True)
    if target_file.parent != p["output"]:
        raise ValueError(f"{trial}: 终点快照不在本轮 production/output")
    equal([sources[0]["time_code"]], [target], "目标快照时间")
    with h5py.File(target_file,"r") as f:
        equal([f["Header"].attrs["Time"]], [target], "实际终点 Header/Time")
    initial = arr(audit["applied_initial_12d"])
    equal(initial, arr(audit["nominal_initial_12d"])+arr(audit["cumulative_offset_12d"]), "累计初态")
    with h5py.File(p["ic"],"r") as f:
        equal([f["Header"].attrs["Time"]],[audit["stage2_start_code"]],"初始时间")
        g = f["MagellanicOrbitReference"]
        measured = np.concatenate([g[s+"_AppliedInitial"+q][...] for s in ("LMC","SMC")
            for q in ("Position_kpc","Velocity_kms")])
        equal(measured, initial, "HDF5 applied 初态")
    order = [(s,c) for s in ("LMC","SMC") for c in ("x","y","z","vx","vy","vz")]
    with p["csv"].open(newline="",encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    by_key = {(r["satellite"].upper(),r["component"].lower()):r for r in rows}
    if len(rows) != 12 or set(by_key) != set(order):
        raise ValueError(f"{trial}: 残差必须为 12 个唯一分量")
    data = {key:arr([by_key[q][key] for q in order]) for key in ("simulated","observed","residual","sigma","zscore")}
    if np.any(data["sigma"]<=0):
        raise ValueError("sigma 必须为正")
    for i,q in enumerate(order):
        if by_key[q]["unit"] != ("kpc" if i%6<3 else "km/s"):
            raise ValueError("残差单位不匹配")
    equal(data["residual"],data["simulated"]-data["observed"],"残差公式")
    equal(data["zscore"],data["residual"]/data["sigma"],"归一化残差")
    equal(data["observed"],np.concatenate(audit["model_key_inputs"]["targets"]),"观测目标")
    check_runtime_paths(p["runtime_param"],p["ic"],p["output"],Path(audit["base_param"]).parent)
    runtime = normalized_params(p["runtime_param"])
    equal([runtime.get("TimeBegin",np.nan),runtime.get("TimeMax",np.nan)],
        [audit["stage2_start_code"],target],"production 时间范围")
    return dict(work_dir=str(Path(work_dir).expanduser().resolve()),validation_subdir=validation_subdir,
        trial=trial,ic_path=str(p["ic"]),ic_sha256=ic_hash,
        orbit_path=str(p["orbit"]),orbit_sha256=sha256(p["orbit"]),
        residual_csv_path=str(p["csv"]),residual_csv_sha256=csv_hash,
        provenance_path=str(sidecar),provenance_sha256=sha256(sidecar),
        applied_initial_12d=initial.tolist(),residual_12d=data["residual"].tolist(),
        sigma_12d=data["sigma"].tolist(),observed_12d=data["observed"].tolist(),
        model_key=audit["model_key"],source_sha256=audit["source_sha256"],
        measurement_signature=provenance["measurement_signature"],target_time_code=target,
        stage2_start_code=audit["stage2_start_code"],units=audit["units"],
        rotation_matrices=audit["rotation_matrices"],base_param_path=audit["base_param"],
        base_param_sha256=audit["base_param_sha256"],gizmo_path=audit["gizmo"]["path"],
        gizmo_sha256=audit["gizmo"]["sha256"],runtime_param_normalized=runtime,
        runtime_param_sha256=sha256(p["runtime_param"]),audit=audit,
        target_snapshot_path=str(target_file))


def same_evidence(recorded, fresh):
    if recorded != fresh:
        keys = sorted(k for k in set(recorded)|set(fresh) if recorded.get(k)!=fresh.get(k))
        raise ValueError("证据文件或运行配置已变化："+", ".join(keys))


def validate_common(a,b,build_review=None):
    review = validate_review(build_review, a)
    assert_build_pair(a["gizmo_sha256"], b["gizmo_sha256"], review)
    if b.get("audit", {}).get("build_review") != review:
        raise ValueError(f"{b['trial']}: 构建审查记录与本批不一致")
    for key in ("model_key","source_sha256","measurement_signature","units",
                "runtime_param_normalized"):
        if a[key] != b[key]:
            raise ValueError(f"{b['trial']}: 与基点的 {key} 不同")
    for key in ("sigma_12d","observed_12d","rotation_matrices","target_time_code","stage2_start_code"):
        equal(a[key],b[key],key)


def validate_probe(anchor,probe,coordinate_index,step,build_review=None):
    validate_common(anchor,probe,build_review=build_review)
    expected = np.zeros(12); expected[coordinate_index] = step
    equal(arr(probe["applied_initial_12d"])-arr(anchor["applied_initial_12d"]),expected,"探测实际增量",atol=1e-6)
    a,b = anchor["audit"],probe["audit"]
    if Path(b.get("shooting_base_ic","")).name != Path(anchor["ic_path"]).name:
        raise ValueError("探测不是从同一个基点生成")
    equal(b["previous_offset_12d"],a["cumulative_offset_12d"],"继承偏移")
    equal(b["this_iteration_increment_12d"],expected,"记录增量")
    equal(b["manual_increment_12d"],np.zeros(12),"手动增量")
    p = b["shooting_residual_provenance"]
    if p.get("reference_ic_sha256") != anchor["ic_sha256"] or p.get("residual_csv_sha256") != anchor["residual_csv_sha256"]:
        raise ValueError("探测的 parent IC/CSV 指纹错误")
    if p.get("measurement_signature") != anchor["measurement_signature"] or p.get("measurement_transition"):
        raise ValueError("探测测量定义发生变化")
    diag = b["shooting_diagnostics"]
    if diag.get("method") != "live-probe" or b.get("shooting_alpha") is not None:
        raise ValueError("需要不乘 alpha 的独立 live-probe")
    if diag.get("alpha_applied") is not False or diag.get("alpha_applied_by_importer") is not False:
        raise ValueError("探测增量不能乘 alpha")
    equal(diag["base_initial_12d"],anchor["applied_initial_12d"],"探测基点初态")
    equal(diag["residual_12d"],anchor["residual_12d"],"探测基点残差")
    equal(diag["sigma_12d"],anchor["sigma_12d"],"探测 sigma")
    return (arr(probe["residual_12d"])-arr(anchor["residual_12d"]))/arr(anchor["sigma_12d"])/step


def pair_diagnostics(observed, residual):
    """由同一 12D 结果派生 SMC−LMC 的间距和相对速度，仅作诊断。

    不引入额外观测、独立误差或 chi-square 权重。标量模长残差与
    相对向量残差不同；同时保留两者，避免以模长相近代替方向相符。
    """
    target = arr(observed)
    actual = target + arr(residual)
    target_r = target[6:9] - target[:3]
    actual_r = actual[6:9] - actual[:3]
    target_v = target[9:12] - target[3:6]
    actual_v = actual[9:12] - actual[3:6]
    target_d, actual_d = float(np.linalg.norm(target_r)), float(np.linalg.norm(actual_r))
    target_speed, actual_speed = float(np.linalg.norm(target_v)), float(np.linalg.norm(actual_v))
    return dict(
        vector_convention="SMC minus LMC",
        target_relative_position_kpc=target_r.tolist(),
        simulated_relative_position_kpc=actual_r.tolist(),
        relative_position_residual_kpc=(actual_r-target_r).tolist(),
        target_separation_kpc=target_d,
        simulated_separation_kpc=actual_d,
        separation_residual_kpc=actual_d-target_d,
        target_relative_velocity_kms=target_v.tolist(),
        simulated_relative_velocity_kms=actual_v.tolist(),
        relative_velocity_residual_kms=(actual_v-target_v).tolist(),
        target_relative_speed_kms=target_speed,
        simulated_relative_speed_kms=actual_speed,
        relative_speed_residual_kms=actual_speed-target_speed,
        independent_observation=False,
        note="由同一 12D 数据派生，不作为独立观测重复加权；无单独 sigma 验收。",
    )
