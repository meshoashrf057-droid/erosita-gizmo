MW/LMC/SMC shooting：较好基准 + 单坐标 live 响应测量
版本：2026-10-14

本次建议
========
trial07 的来源记录、增量、累计 offset 均核对一致；实测范数比 trial06 更大。
保留 trial06 作为下一次的基准，把 trial07 留在历史里作为已测响应。
仅组合目前三条已测方向，线性诊断无法明显改善剩余 y 偏差。
因此下一次先测一个新的独立方向，而不是再一次声称联合修正可使全分量通过。

推荐 trial08：从 trial06 初态出发，仅 LMC 的 MW 相对初始 y 加 1 kpc。
其余 11 个整体初态坐标分量不变，内部结构来自相同三个弛豫末态。
这是 live-response measurement：实际需要完整 GIZMO 演化，单靠静态积分不够。
它不保证残差下降，也不保证 6D 通过；测量将提供一个新的 12D 响应方向。
若表现更差，仍保留较好的基准，使用这个结果更新响应。
本次不是把 trial07 的终态继续演化，也不是重跑原 trial06。

放置位置与依赖
==============
把本 ZIP 中的文件解压到 ~/gizmo-runs/mw_isolated，和原文件放在同一目录。
沿用原目录中的：
  tg19_backend.py
  satellite_relax_common.py
  shooting_local.py
  merge_relaxed_mw_lmc_smc.py
  validation_dashboard_sat_6d_gaswake_v3.py
依赖 NumPy、SciPy、h5py；使用你现有 .venv。
新文件名不会覆盖上述旧入口。这里不改 GIZMO 源码、编译项、param 或快照。
脚本生成新 IC 及配套 run 目录；若输出已存在会停止。

1. 生成 trial08 IC
=================
cd ~/gizmo-runs/mw_isolated
unzip shooting_anchor_probe_v2.zip

python merge_relaxed_mw_lmc_smc_probe_v1.py next \
  --base-param ./mw_isolated.param \
  --shooting-base-ic ./mw_lmc_smc_relaxed_trial06_000.hdf5 \
  --shooting-residual-csv ./mw_lmc_smc_relaxed_trial06_000_run/production/output/magellanic_validation_fixed/magellanic_present_6d.csv \
  --shooting-method live-probe \
  --probe-coordinate LMC_y \
  --probe-step 1.0 \
  --require-shooting-provenance \
  --output ./mw_lmc_smc_relaxed_trial08_000.hdf5 \
  --trial-label trial08

应看到 LMC dpos=[0,1,0] kpc、dvel=[0,0,0] km/s；SMC 的两个增量全零。
这里不设置 alpha。live-probe 不优化、不乘 alpha、不叠加手动 offset。
读入 base CSV 是为了记录测量基准及检查来源，不会用它追加 shooting 修正。
静态两个终点仅作诊断，预测变差也不会取消 probe。
若静态积分失败，会在写出前停止：历史导出需要有限的静态诊断字段。
位置探测幅度硬上限 2 kpc，速度探测幅度硬上限 5 km/s。

2. 读入检查及完整演化
====================
bash ./mw_lmc_smc_relaxed_trial08_000_run/smoke/run.sh &&
bash ./mw_lmc_smc_relaxed_trial08_000_run/production/run.sh

生成的两套运行脚本均使用 flag2。production 保持原 1 Gyr 落入时长。

3. 演化完成后的统一检查
======================
bash ./validate_trial08_probe.sh

它先运行原 original-stars validator，再调用新版分支感知历史导出器。
输出：shooting_response_audit_04_08.json。
导出器要求新输出文件不存在；重复执行时请给新文件名，保留历史结果。
这一小 JSON 即可供下一次响应分析；不需要上传大型快照。
如果 validator 未覆盖目标时刻，后续导出会拒绝，不能用中途输出代替完整终点。

为什么要新版历史导出
====================
trial08 的 parent 是 trial06，trial07 仍按时间顺序留在列表中。
audit_shooting_history_v2.py 按真实 base 文件名及 SHA/CSV 来源连接步骤。
它允许 04->05->06->07，随后从06分支到08，逐项核对累计 offset 与诊断输入。
旧 exporter v1 强制相邻 parent，因此不要拿它导出包含这个分支的历史。

一并提供的候选 v2（本次无需先运行）
===================================
propose_history_shooting_v2.py + history_response_v2.py 支持以历史中较好基准
为 anchor，同时保留之后较差的试验。它不再强制 base 等于历史最后一行。
可核对历史实际/预测损失下降比、区分 probe 测量与修正步，并要求所有 ridge
预测满足候选条件才选择。默认候选限步为位置2kpc/速度5km/s。
它仍然使用静态先验，并没有获得完整 live 12x12 Jacobian。
因此本次推荐先运行上述单方向测量，不需要再发起一轮候选预测来替代它。
后续若使用该工具，base IC 必须为所提供历史里加权范数最小的轮次。

验证范围
========
包含纯数值单元测试，覆盖早先 anchor、分支来源链、单坐标增量、限幅、
静态预测变差的 probe、错误输入等。合成 HDF5 集成检查核对非零既有累计
offset 下只平移 LMC y、其他粒子字段保留、输入只读及 flag2 脚本。
这些检查不替代你的真实 GIZMO live 结果。
trial07_review.json 是对上传小文件的审计；它不等同于直接读取所有 IC 粒子。
