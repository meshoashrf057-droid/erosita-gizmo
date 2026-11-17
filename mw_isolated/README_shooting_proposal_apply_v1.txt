应用已经审查的历史响应候选，生成 trial07 IC
===========================================

把本压缩包直接解压到 ~/gizmo-runs/mw_isolated。
新增入口 merge_relaxed_mw_lmc_smc_history_v1.py，原 merge_relaxed_mw_lmc_smc.py 保留。
新入口依赖同目录原有 tg19_backend.py、shooting_local.py、satellite_relax_common.py，
以及本包的 shooting_proposal_io.py。无需替换 GIZMO 二进制、重新弛豫或改物理模块。

这次选择 trial07_shooting_proposal.json 中 selected=candidate_02。
该增量已经包含 alpha=0.3 和限步处理。下面的 alpha/cap 参数仅用于一致性校验，
不会再次乘 alpha、缩放增量或另加一轮旧 local-defect。
程序严格核对 base IC、残差来源、目标、单位、初态和模型，重放两次静态积分确认预测。

生成：
cd ~/gizmo-runs/mw_isolated
python merge_relaxed_mw_lmc_smc_history_v1.py next \
  --base-param ./mw_isolated.param \
  --shooting-base-ic ./mw_lmc_smc_relaxed_trial06_000.hdf5 \
  --shooting-residual-csv ./mw_lmc_smc_relaxed_trial06_000_run/production/output/magellanic_validation_fixed/magellanic_present_6d.csv \
  --shooting-method history-proposal \
  --shooting-proposal-json ./trial07_shooting_proposal.json \
  --shooting-alpha 0.3 \
  --shooting-max-position-step-kpc 5 \
  --shooting-max-velocity-step-kms 10 \
  --require-shooting-provenance \
  --output ./mw_lmc_smc_relaxed_trial07_000.hdf5 \
  --trial-label trial07

源输入仍为 base IC 中记载的 MW/LMC/SMC 三个独立弛豫末态。
不是从 trial06 的1Gyr演化终态继续落入。
生成的初态应为（MW相对坐标）：
LMC r/kpc = [17.6384759812, 178.9700734983, 0.2953839114]
LMC v/km/s = [13.7310610168, -85.6686971820, -108.5557614362]
SMC r/kpc = [-4.3010174006, 26.8786911075, 85.4733803482]
SMC v/km/s = [-14.9250234446, 131.7887654645, -58.7388428172]

运行近零时间读入检查：
bash ./mw_lmc_smc_relaxed_trial07_000_run/smoke/run.sh

检查正常后运行正式演化（脚本使用 flag2）：
bash ./mw_lmc_smc_relaxed_trial07_000_run/production/run.sh

完成后：
python validation_dashboard_sat_6d_gaswake_v3.py \
  --snapshot-dir ./mw_lmc_smc_relaxed_trial07_000_run/production/output \
  --reference-ic ./mw_lmc_smc_relaxed_trial07_000.hdf5 \
  --stellar-tracers original-stars \
  --output-dir ./mw_lmc_smc_relaxed_trial07_000_run/production/output/magellanic_validation_fixed

请不要把“候选模型预测所有分量<3sigma”当成已经live通过。
所选候选在历史ridge正则化变化下接近一致，但不同LM阻尼下方向会变化。
78.8%指缩放后的初态增量在历史两维子空间外的范数占比，不是成功概率。
输出orbit.json和IC内AuditJSON保存来源、完整候选、预测和实际施加增量，方便下轮校准。
原有输出文件或运行目录存在时拒绝覆盖；不要把旧 trial07 数据混入新运行。

验证范围：数值导入检查和合成HDF5合并测试；没有声称已运行真实trial07 GIZMO。
