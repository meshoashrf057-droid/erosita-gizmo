# 科研流程与脚本入口

本仓库保存用户的 GIZMO 科研 Python 工具，以及用于说明历史运行的脚本。初值、模拟、数值诊断、投影和全天图是不同阶段；版本号更大不等于可以替代当前物理模型。

**当前 gas19 模型降低了 LMC/SMC 初始气体质量，保留其余无碰撞结构，再分别弛豫并重新落入。** 它是新实验，不能继承旧质量模型的 12D 合格结论。仓库中的历史运行脚本不代表当前运行状态。

**用户已取消之后所有模拟的烟测步骤。** 历史代码中的 smoke 命令和结果仅作为记录保留，不应自动恢复这一流程。输入字段、单位和参数核对仍然需要；合成 Python 测试不是短模拟烟测。

## 1. 公共模块与路径

保留原目录层级，不要把所有 Python 文件平铺到一起：

- 根目录 `gizmo_compat.py`：C/C++ 快照的 yt 兼容读取。
- 根目录 `gizmo_schema.py`、`gizmo_units.py`、`gizmo_ic_compat.py`：字段、单位和初值写入检查。
- 根目录 `gizmo_diagnostics_compat.py`：诊断共享适配。
- `mw_isolated/`：MW 初值、卫星弛豫、合并、轨道、旋转曲线。
- `erosita_xray/`：用户可视化、全天图、气体诊断、喷流输入准备。
- 带日期的 campaign 目录：该次运行的实现记录，不是通用启动器。

以下命令从仓库根目录执行，激活自己的 Python 环境，用 `python` 指向它。变量中的路径只是占位：

~~~bash
export DATA="/path/to/erosita-data"
export GIZMO="/path/to/native-gizmo/GIZMO"
export BASE_PARAM="/path/to/selected-model/base.param"
~~~

GIZMO 参数文件及部分生成器不能处理带空白字符的路径，建议数据与运行目录不含空格。

## 2. MW 初值与孤立演化

| 入口（均在 mw_isolated） | 模型边界 |
|---|---|
| `make_ic_live_mw_cgm_isolated_v3.py` | live DM、恒星盘、核球、冷 ISM、热 CGM 和中心 BH 的联合初值。 |
| `make_ic_live_mw_cgm_isolated_v3_highres.py` | 同系列 ID 布局兼容版；`--id-layout auto` 防止高分辨率 ID 分区重叠，粒子质量仍须指定。 |
| `make_ic_live_mw_cgm_isolated_v2.py` | 不含冷 ISM 的旧联合模型。 |
| `make_ic_live_mw_disk_stage1_v1.py` | 旧碰撞无关 MW 盘阶段。 |
| `make_ic_mou_hot_cgm_v4.py` | 另一套 Mou-like MW + McMillan ISM 模型。 |
| `make_ic_mou_hot_cgm_noism_v5.py` | 上一模型的无 ISM 诊断分支，不能自动代替当前有冷盘模型。 |

下面展示显式指定模型的 CLI；先设置所选模型的 `CGM_MASS_R200`。脚本默认值不等于当前三倍 CGM 实验的全部配置：

~~~bash
python mw_isolated/make_ic_live_mw_cgm_isolated_v3_highres.py \
  --output "$DATA/new_model/mw_000.hdf5" \
  --target-format cpp --id-layout auto --seed 42 \
  --dm-particle-mass 5e6 --star-particle-mass 5e5 \
  --cgm-particle-mass 1e5 --ism-particle-mass 1e5 \
  --metallicity 0.3 --cgm-mass-r200 "$CGM_MASS_R200"
~~~

其他未列参数沿用脚本默认值，因此该命令不是历史模型精确复现声明。`--reference-model-ic` 可继承旧 v3 IC 的结构和热力学参数，显式参数优先，粒子质量与 seed 不继承。

生成初值与正式演化是独立操作。孤立检查入口为 `check_live_mw_cgm_isolated_relaxation_v3.py`；旧模型也保留对应 checker。历史诊断 FAIL 与用户已接受的持续内流、总 SFR 和恒星旋转模型选择要区分，不能自动据此改变模型。

## 3. LMC/SMC 初值与弛豫

老版通用入口为 `mw_isolated/make_lmc_relax_ic.py`、`make_smc_relax_ic.py`，依赖同目录 `satellite_relax_common.py` 和 `tg19_backend.py`。

**当前低气体版本位于 `mw_isolated/cgm3_n3000_gas19_20261116/code/`。** 它的 common 新增必填 `--gas-total-mass`；只运行顶层老版会遗漏这一质量覆写。

~~~bash
python mw_isolated/cgm3_n3000_gas19_20261116/code/make_lmc_relax_ic.py \
  --target-format cpp --base-param "$BASE_PARAM" \
  --gizmo-executable "$GIZMO" --run-dir "$DATA/new_lmc" \
  --relax-code .6136 --stellar-age-gyr 5 \
  --gas-total-mass 1.1e9 --gas-particle-mass 1e5 \
  --dm-particle-mass 5e6 --star-particle-mass 5e5

python mw_isolated/cgm3_n3000_gas19_20261116/code/make_smc_relax_ic.py \
  --target-format cpp --base-param "$BASE_PARAM" \
  --gizmo-executable "$GIZMO" --run-dir "$DATA/new_smc" \
  --relax-code .6136 --stellar-age-gyr 5 \
  --gas-total-mass 7.9e8 --gas-particle-mass 1e5 \
  --dm-particle-mass 5e6 --star-particle-mass 5e5
~~~

`.6136` 是该次采用的代码时间，约 0.5999734 物理 Gyr。新实验需要精确物理时长时按 Header 单位换算，不把代码时间直接当物理 Gyr。

生成器记录所选二进制 SHA 和参数，不编译 GIZMO。它生成的历史 Bash launcher 使用 `sha256sum`，Mac 需提供该命令或使用移植后的启动方式。不能直接重复执行带 once 的旧 campaign launcher。

## 4. 合并与轨道初态

主要入口：

- `mw_isolated/merge_relaxed_mw_lmc_smc.py merge`：合并三个真实演化快照。
- `mw_isolated/replay_relaxed_orbit.py`：把已有轨道的实际初始位姿用于明确指定的新演化源。这是新实验，不继承旧通过结论。
- `prepare_gizmo_stage2_handoff_v1.py`：显式快照交接工具；快照交接不等于无损二进制 checkpoint 续算。
- `prepare_gizmo_metal_restart_v1.py`：保留的旧金属状态交接工具，先核对其编译宏与读取模式要求。

~~~bash
python mw_isolated/merge_relaxed_mw_lmc_smc.py merge \
  --mw-snapshot "$DATA/mw/output/snapshot_final.hdf5" \
  --lmc-run-dir "$DATA/new_lmc" --smc-run-dir "$DATA/new_smc" \
  --lmc-snapshot "$DATA/new_lmc/production/output/snapshot_final.hdf5" \
  --smc-snapshot "$DATA/new_smc/production/output/snapshot_final.hdf5" \
  --base-param "$BASE_PARAM" --gizmo-executable "$GIZMO" \
  --target-format cpp --lookback-gyr 1.0 \
  --output "$DATA/new_orbit/orbit_000.hdf5" --trial-label new_orbit
~~~

`snapshot_final.hdf5` 是占位名称，须换成实际末帧，并保留卫星生成/弛豫 manifest。合并器保留内部状态、旋转和粒子来源；时间重标时平移 stellar birth time 以保持星龄。不要重复累加已有 offset。

当前 gas19 的 `scripts/build_calibration_ic.py` 专门绑定旧 accepted 初态、冻结来源及 calibration01 目录；它仅作为该次构建记录保存，不是泛化 Mac 合并入口。

## 5. 正式落入、12D 与轨道修正

先在 Mac 编译与所选物理配置一致的 GIZMO，再按输入种类和生成参数选择读取模式。原 Linux 可执行文件不能在 Mac 运行；Linux 二进制 restart 不能假定跨架构兼容。

`RestartFlag=2` 是完整快照状态初始化模式，本身不等于烟测。取消短烟测不表示可随意更换正式模拟读取模式。

轨道诊断优先使用 **mw_isolated 目录下**冻结原始恒星身份的版本：

~~~bash
python mw_isolated/validation_dashboard_sat_6d_gaswake_v3.py \
  --snapshot-dir "$DATA/new_orbit/production/output" \
  --reference-ic "$DATA/new_orbit/orbit_000.hdf5" \
  --output-dir "$DATA/new_orbit/validation"
~~~

固定观测值与 sigma 储存在参考 IC 的 `MagellanicOrbitReference`。不要以 erosita_xray 中较老的同名脚本覆盖此版本。

正式通过要求同 IC、原始恒星身份、实际目标时刻和真正末帧，12 项各自 `|z| <= 3`，D12 只辅助。最近帧或插值不能冒充真终点。轨道通过不等于气体盘形态或束缚结构通过。

| 类别 | mw_isolated 中的代码 |
|---|---|
| 终点 gate | `shooting_cpp_20261031_082656/check_orbit_gate.py` |
| 局部步与探针 | `shooting_local.py`、`shooting_probe.py`、`shooting_joint_y.py` |
| 历史响应 | `history_response_v2.py`、`propose_history_shooting_v2.py`、`shooting_proposal_io.py` |
| 合并适配 | `merge_relaxed_mw_lmc_smc_history_v1.py`、`merge_relaxed_mw_lmc_smc_probe_v1.py`、`merge_relaxed_mw_lmc_smc_joint_y_v1.py` |
| 只读审计 | `audit_shooting_history_v2.py`、`audit_mw_center_response_v1.py` |

历史 shooting 需配对 IC、实际终点 CSV、provenance、来源与运行记录；只带走 proposal JSON 不足以重用。新的质量模型和新构建不能自动继承旧 Jacobian。`shooting_local_live_v1/` 明确绑定旧 trial10，旧构建例外不适用于新家族。

## 6. 数值采集、切片与旋转曲线

| 入口 | 功能 |
|---|---|
| `mw_isolated/mw_validator_v6_1.py` | 内部版本 v6.2，实际星/气旋转、总引力圆速度、密度/温度/金属图及 CSV。 |
| 两目录的 `data_collector_unified.py` | 盘形状、速度、SFR、BH、卫星时间序列；使用顶部变量，无 argparse CLI。 |
| `erosita_xray/mwcgm_velocity_diagnostic_v1.py` | 按来源/温度选 MW-CGM，在指定厚度区域诊断速度与横风。 |
| `erosita_xray/diagnose_prejet_cgm_crosswind_v1.py` | 对比落入组与同宿主同物理的无卫星控制组。 |
| `chazhao.py` | CGM 径向流、温度与质量预算。 |
| `123.py` | 粒子质量分辨率检查，不是无用占位文件。 |
| `check_snapshot_metallicity_v1.py` | 金属及可选 APEC 发射率核对。 |

~~~bash
python mw_isolated/mw_validator_v6_1.py \
  --snapshot-dir "$DATA/new_orbit/production/output" \
  --output-dir "$DATA/rotation" --start 10 --stop 10

python erosita_xray/mwcgm_velocity_diagnostic_v1.py \
  "$DATA/new_orbit/production/output/snapshot_010.hdf5" \
  --reference-ic "$DATA/new_orbit/orbit_000.hdf5" \
  --output-dir "$DATA/cgm_velocity" --fov 100 --slab-half-width 5
~~~

速度诊断的 `--fov` 是 y/z **半宽**，上例对应200 kpc全宽。旋转曲线区分运动学速度和引力圆速度；诊断 Plummer 平滑长度不自动等于 GIZMO softening。

采集器需设置 `snapshot_dir`、`snap_nums`、`output_file`、插入时间和模型 ID 选择，不接受不存在的 `--snapshot-dir`。正式 12D 仍使用固定原始恒星身份的 validator。

## 7. 44/200 kpc 可视化与 video

用户主脚本为 `erosita_xray/erosita_bubble_xray_yt_v10_log.py`。导出包提供可移植调用入口，定义和资源说明见 [VISUALIZATION.md](VISUALIZATION.md)。

~~~bash
python tools/render_v10.py \
  --snapshot-dir "$DATA/new_orbit/production/output" \
  --output-dir "$DATA/figures" --fovs 44 200 --indices 10 \
  --reference-ic "$DATA/new_orbit/orbit_000.hdf5" \
  --apec-data-dir erosita_xray
~~~

这里的 `--fovs` 是全宽。多帧选择可用 `--start`、`--stop`、`--stride`；video 参数见可视化文档。普通投影、厚度切片和观察者全天图不是同一物理量，绝对亮度比较需统一定义。

## 8. 角动量方向与喷流

`erosita_xray/azaxis.py` 在新的 IC 副本中设置 BH/Sink 比角动量向量：

~~~bash
python erosita_xray/azaxis.py \
  --input "$DATA/new_orbit/production/output/snapshot_final.hdf5" \
  --output "$DATA/jet/jet_000.hdf5" \
  --target-format cpp --axis 0 0 1 --report "$DATA/jet/axis_report.json"
~~~

它修改 BH/Sink 字段，**不旋转全部粒子，也不自动测量恒星盘角动量**。`--axis 0 0 1` 是固定 z 轴示例。

喷流注入由 GIZMO 源码、Config 和参数共同决定。仅有 azaxis 和参数模板不构成固定供给喷流复现；自然吸积与固定外部供给不同。历史 fixed-supply C++ port 的构建、stage3 准备及一次性 runner 只记录当时实验，不能直接在 Mac 执行其旧启动器。

## 9. 全天图

按研究定义选择：

- `allsky_xray_yt_v3_disk_preserved.py`：保留盘分支，默认 disk mask=0。
- `allsky_xray_yt_v4_disk_masked_audit.py`：盘掩膜和发射率重建审计分支，默认掩膜及网格范围不同，不能静默替代 v3。
- 其他版本保留作历史工具；统一能段、亮度单位、吸收和观察者设置后才能比较。

~~~bash
python erosita_xray/allsky_xray_yt_v3_disk_preserved.py \
  --snapshot-dir "$DATA/allsky" \
  --output-dir "$DATA/allsky_figures" --grid-dir "$DATA/allsky_grids" \
  --apec-data-dir erosita_xray --bh-id 90000000 --start 1 --stop 30
~~~

详细定义见 [VISUALIZATION.md](VISUALIZATION.md)。工作流不自动替用户选择最终全天图口径。

## 10. Mac 迁移边界

- Python 和 HDF5 可迁移；不复制 Linux .venv、ELF 可执行文件作为 Mac 程序。
- GIZMO 需本机编译，核对所选物理模块、Config、MPI/HDF5/GSL 等依赖，记录新 SHA；不改历史 hash 假装旧构建。
- campaign 的 stage_safety 读 Linux `/proc`；旧运行脚本另有 tmux、固定路径、Linux 动态库和一次性锁。不能简单替换路径就声称完成移植。
- 参数模板只按实际环境修改路径；不自动更改物理值。冷却表、TREECOOL、APEC 表和观察资料需随所选版本保留或明确下载。
- 精确复现还需要初态、真正末帧、sidecar、参数、Config、构建来源、观测口径、随机种子。代码仓库不能替代这些模拟数据。
- 历史 test_*.py 多为合成 I/O/数学检查，不是科学收敛证明，也不是被取消的短模拟烟测。
