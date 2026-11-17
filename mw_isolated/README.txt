shooting 历史响应校准 v1
========================

用途
  读取 trial04/05/06 的历史和 trial06 的原始静态势，在当前初态计算
  12D 静态差分响应，用两条实际 live 方向校准静态响应误差。
  只输出候选 JSON；不写 IC、不修改快照/目标/容差/物理参数，不运行 GIZMO。

安装
  在 ~/gizmo-runs/mw_isolated 解压本包，得到 shooting_history_calibration_v1 目录。
  现有 merge_relaxed_mw_lmc_smc.py、satellite_relax_common.py、shooting_local.py、
  tg19_backend.py 保持原样放在 mw_isolated。使用产生 trial06 的版本。
  同目录放 shooting_response_audit_04_06.json。
  依赖与原 merger 一致：Python、numpy、scipy、h5py。

运行
  cd ~/gizmo-runs/mw_isolated
  python shooting_history_calibration_v1/propose_history_shooting_v1.py \
    --pipeline-dir . \
    --base-ic ./mw_lmc_smc_relaxed_trial06_000.hdf5 \
    --base-param ./mw_isolated.param \
    --history-json ./shooting_response_audit_04_06.json \
    --alpha 0.3 \
    --max-position-step-kpc 5 \
    --max-velocity-step-kms 10 \
    --output ./trial07_shooting_proposal.json

  最新残差默认读取 base IC 对应的
  <base_stem>_run/production/output/magellanic_validation_fixed/magellanic_present_6d.csv。
  路径不同可用 --residual-csv 指定，但 CSV 的来源必须仍匹配 trial06。
  原三个独立弛豫末态的路径从 base IC 内自动继承；需要这些原文件仍可访问。
  程序会检查 source/base SHA256、单位、观测目标、时间、示踪定义、历史累计关系。
  输出文件必须不存在；重复计算请换输出名。

如何解释
  固定初态尺度为每颗卫星位置20 kpc、速度30 km/s；终态除以观测sigma。
  初态差分仍用kpc和km/s原单位，中央差分步长0.02 kpc / 0.02 km/s。
  历史修正秩最多为历史方向数；本例只有2，不能等同完整实测12x12 Jacobian。
  ridge_relative 和 damping_relative 是数值正则化网格参数，不改变物理模型。
  ridge弱/强以及阻尼不同，可能给出不同方向；报告保留整个网格供审查。
  默认每颗卫星本次位置增量不超过5 kpc，速度增量不超过10 km/s。
  这些是试探步长，不是观测误差，也不意味着扰动足够线性。
  接受条件仅针对校准模型预测：总残差下降、最大标准化残差下降、
  当前不通过的分量各自改善。预测不是 live 结果，过3sigma仍需完整GIZMO演化验证。
  outside_history_fraction 是初态方向在历史子空间外的长度占比，不是概率或置信度。
  如果没有候选，仍输出完整报告，不自动改目标/扩大容差/启动昂贵演化。

下一步
  先检查 trial07_shooting_proposal.json 中完整12D预测、不同网格下的方向稳定性，
  再选择是否用其生成一个独立的小步 live 校准。不能把 proposal 当作已通过的IC。
  仅这一阶段需要源快照读取和几十次静态ODE积分，不需要新增GIZMO运行。

验证范围
  数值核心以合成耦合映射验证符号、历史响应修正、有限性检查和步长上限。
  历史记录检查已用用户真实04–06导出测试。
  用户大型源快照不在本地测试环境，因此未声称已算出真实trial07候选或完成live验证。
