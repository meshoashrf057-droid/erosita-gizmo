# TG19：LMC、SMC 独立弛豫，再与 MW snapshot_007 合并

这是一套可独立使用的完整脚本，不需要另外导入旧的 add_lmc_smc 脚本。所有 `.py` 文件请放在同一目录。它不修改你的原脚本、MW 快照或卫星演化快照。

当前流程：LMC 在 `lmcre`、SMC 在 `smcre` 各自演化至 **Time=0.6136 代码时间（约0.6000 Gyr）**；读取两个末态，保留演化后的粒子状态；通过原脚本的 K13 终点反演方法赋予位置、速度；与指定的 MW `snapshot_007.hdf5` 合并，使用 **flag=2** 进入落入阶段。

这是 **TG19 参数启发的独立弛豫模型**。两颗卫星没有在此阶段相互作用，不是 TG19 的 6 Gyr 二体预演化末态，也不保证再现其中的冷气体质量、潮汐桥或 Stream。

**2026-10-11 合并器 v1.3 / validator v3.2（本次实际数据审查）：**原始add脚本的6D修正与合并器代数一致，不存在移植时漏掉上轮offset的证据。本次CSV确认SMC示踪范围跨进气体ID区，新生恒星持续加入；新版固定reference IC中的原生老恒星复合身份，并修复时间曲线与类别柱状图共用横轴的问题。先重新诊断已有trial00/01，再生成trial02；不要继续用本次上传的旧SMC残差。

**2026-10-10 合并器 v1.2：**增加当前初态局部shooting响应、共同限步、残差配对校验及 `next` 命令；配套validator v3.1只增加中心审计，不改变已有12D测量。见第6节直接从trial01生成trial02。本次没有实际读取你的trial01 HDF5，不能声称已经数值校准成功。

**2026-10-09 合并器 v1.1 修复：**孤立非周期模型不再因粒子坐标小于 0 或大于 Header/BoxSize 而停止。保留全部粒子的原坐标、质量和状态，不做周期折回、裁剪、平移或扩大 BoxSize；在终端及 `.orbit.json` 的 `coordinate_extent_diagnostics` 中记录越出名义盒的粒子数、质量占比、MW/LMC/SMC 来源和坐标范围。NaN/Inf 仍被拒绝。旧版因 BoxSize 检查失败的运行可只替换 `merge_relaxed_mw_lmc_smc.py`，然后重跑原合并命令；共用模块、弛豫 IC、manifest 和已有快照无需重建。

该处理适用于无 `BOX_PERIODIC`（或旧名 `PERIODIC`）、无特殊盒边界的孤立模拟。`GRAVITY_NOT_PERIODIC` 只控制引力，不会关闭周期流体边界；若快照 Config 明确声明周期/特殊盒边界，本合并器会指出配置不符。缺少 Config 时沿用本项目已声明的孤立非周期设定。源码 `begrun.c` 在无盒相关编译项时允许忽略 BoxSize；后处理工具仍可能用 Header/BoxSize 定义读取域，因此分析越出名义盒的粒子时需使用覆盖实际坐标的读取域。

## 1. 文件

| 文件 | 用途 |
|---|---|
| `make_lmc_relax_ic.py` | 创建 LMC IC、参数文件和 flag0 启动脚本 |
| `make_smc_relax_ic.py` | 创建 SMC IC、参数文件和 flag0 启动脚本 |
| `merge_relaxed_mw_lmc_smc.py` | 合并、轨道初猜、累计 shooting、flag2 读入检查 |
| `satellite_relax_common.py` | HDF5、单位、参数文件、粒子身份等共用实现 |
| `tg19_backend.py` | 从你提供的卫星脚本提取的结构采样和轨道算法 |
| `shooting_local.py` | 当前初态附近的静态正向响应、局部修正及共同限步 |
| `validation_dashboard_sat_6d_gaswake_v3.py` | v3.2：固定原生老星身份、正确示踪保留率、legacy对照、CSV配对及图轴修复 |
| `test_pipeline.py` / `test_shooting_local.py` / `test_validator_audit.py` | 合成数据与数值算法测试；不运行 GIZMO |
| `requirements.txt` | Python 依赖 |

最低 Python 3.10。使用你现有的 `.venv` 即可；若缺依赖：

```bash
python -m pip install -r requirements.txt
```

下列示例假定你将文件放在 `/home/xingdi/gizmo-runs/mw_isolated/` 并进入此目录，实际 MW 参数是 `./mw_isolated.param`、MW 末态是 `./output/snapshot_007.hdf5`。如果它们实际位于其他目录，修改对应参数即可。GIZMO 参数文件不支持路径中的空白字符，本脚本也不接受这种输出路径。

## 2. 时间约定：务必区分 Gyr 与 TimeMax

你当前三个 snapshot_007 的 Header/Time 均为 0.6136。保持这个代码时间，不需要为了 shooting 重做弛豫。下列生成示例已使用 `--relax-code 0.6136`。

在 `UnitLength_in_cm=3.08568e21`、`UnitVelocity_in_cm_per_s=1e5` 下：

- `--relax-gyr 0.6136`：生成 `TimeMax≈0.627535692619`。
- `--relax-code 0.6136`：生成 `TimeMax=0.6136`，物理时长约 0.6000 Gyr。

这两个参数互斥。不必要求三个快照的 Header/Time 相同；合并器会统一时钟并保留恒星实际年龄。

## 3. 创建并运行 LMC、SMC 弛豫

```bash
python make_lmc_relax_ic.py \
  --base-param ./mw_isolated.param \
  --gizmo-executable ~/gizmo-public/GIZMO \
  --run-dir ./lmcre \
  --relax-code 0.6136 \
  --gas-particle-mass 1e5 \
  --dm-particle-mass 5e6 \
  --star-particle-mass 5e5 \
  --metallicity 0.2

python make_smc_relax_ic.py \
  --base-param ./mw_isolated.param \
  --gizmo-executable ~/gizmo-public/GIZMO \
  --run-dir ./smcre \
  --relax-code 0.6136 \
  --gas-particle-mass 1e5 \
  --dm-particle-mass 5e6 \
  --star-particle-mass 5e5 \
  --metallicity 0.1
```

每个目录生成：

- `lmc_ic.hdf5` 或 `smc_ic.hdf5`；
- `lmc.param` 或 `smc.param`；
- `run_relax.sh`；
- `relax_manifest.json`，记录物理时间、代码时间、模型、单位、GIZMO 二进制校验值等；
- `output/`。

脚本继承你真实使用的 `base-param`：软化长度、邻居数、恒星形成、反馈等运行参数保持原配置。只修改本阶段的输入、输出、时间、初始金属丰度和显式记录的恒星年龄等必要项。**它不会自动编辑 Config.sh，也不会重新编译 GIZMO。**

继续使用你目前的冷却、低温冷却、分子处理、恒星形成与显式力学反馈配置；不要求开启 `GALSF_EFFECTIVE_EQS`。三个系统应使用同一个编译好的 GIZMO。弛豫和落入阶段保持 AGN 吸积/喷流关闭。本套脚本针对当前纯流体加显式恒星反馈的配置；不作为任意 MHD、辐射输运、CR、CHIMES 模型的通用拼接器。

启动：

```bash
bash ./lmcre/run_relax.sh
bash ./smcre/run_relax.sh
```

默认每次 6 个 MPI 进程。可使用 `NPROC=3 bash ./lmcre/run_relax.sh` 修改进程数。示例按顺序运行，避免同时抢占同一台机器的内存和 CPU。启动脚本在 `base-param` 所在目录执行，从而沿用该目录的冷却表等相对路径；输出采用绝对路径。

这些启动脚本使用 **flag0**。每个卫星自身采用一个统一初始 Z：LMC 0.2 Zsun，SMC 0.1 Zsun；气体和既有恒星均受该全局初值影响。初始逐元素比例由你当前 GIZMO 的初始化逻辑确定，随后允许富集演化。它没有声称重建观测到的空间金属梯度。

**恒星年龄不是固定单龄。** `--stellar-age-gyr` 对应原代码 `InitStellarAge`，在这份源码里是均匀分布 `[0, 2×InitStellarAge]` 的均值。默认继承 `base-param`；如果其中没有该参数，则显式写入代码原默认 10 Gyr，并在终端提示。这个默认分布可包含超过宇宙年龄的恒星，不能当作观测星族模型。你若要采用一个经选择的近似平均年龄，请生成时加 `--stellar-age-gyr 数值`；不能把该参数误当作所有恒星的共同年龄。不同年龄分布会改变显式反馈，需作为模型假设记录。

生成器不会覆盖非空 `lmcre`、`smcre`。若这些目录已经用于旧实验，应选择新的运行目录，避免混入旧输出。

## 4. 合并两个卫星末态与 MW snapshot_007

两个卫星完成演化后：

```bash
python merge_relaxed_mw_lmc_smc.py merge \
  --mw-snapshot ./output/snapshot_007.hdf5 \
  --lmc-run-dir ./lmcre \
  --smc-run-dir ./smcre \
  --base-param ./mw_isolated.param \
  --gizmo-executable ~/gizmo-public/GIZMO \
  --output ./mw_lmc_smc_relaxed_trial00_000.hdf5 \
  --lookback-gyr 1.0 \
  --trial-label trial00
```

若需要将 MW 原 IC 中的来源 ID 元数据一并写入，可添加：

```text
--host-reference-ic ./mw_live_cgm_ism_isolated_v3.hdf5
```

该参数只补来源元数据，不用原 IC 覆盖弛豫后的物理字段。MW 原 IC 的分析参数保留为历史说明；原来的 EQS/关闭显式反馈建议不会成为当前运行要求。

LMC、SMC 默认从各自 `output/` 中选择 **Header/Time 最大的单文件 snapshot**，并检查它是否已经达到 manifest 中的目标时刻。不会假定末态一定叫 006 或 007。必要时可显式指定 `--lmc-snapshot`、`--smc-snapshot`，但仍必须达到计划的弛豫末态。

输出为：

- `mw_lmc_smc_relaxed_trial00_000.hdf5`；
- `mw_lmc_smc_relaxed_trial00_000.orbit.json`；
- `mw_lmc_smc_relaxed_trial00_000_run/smoke/`：flag2 短程读入检查；
- `mw_lmc_smc_relaxed_trial00_000_run/production/`：flag2 正式落入运行。

文件名末尾保留 `_000.hdf5`：你这份 GIZMO 源码的 flag2 启动会用最后一个下划线后的数字推断下一帧编号。因此新输出通常从 `snapshot_001.hdf5` 开始。这里的 000 是输入编号，不是要求重新 flag0 初始化。

合并时：

1. MW 坐标和速度保持不动；卫星用老恒星核心确定中心和速度，剔除各自独立运行的整体漂移，再赋予目标位置、速度。中心定义在 JSON 中记录。
2. 卫星所有组分作相同刚体旋转和平移。默认盘法向沿名义初始轨道角动量；这是继承旧脚本的简化，不是观测盘面姿态。可用 `--lmc-disk-normal=nx,ny,nz`、`--smc-disk-normal=nx,ny,nz` 明确设置。shooting 迭代不会因修正初始位置/速度而重新改变盘面。
3. 质量、气体内能、电子丰度、完整 11 列金属丰度、恒星形成时间等逐粒子保留。密度和光滑长度作为读入猜测保留；重力势、重力加速度清零为占位值，由 GIZMO 在新环境重算。可识别的矢量随盘旋转，未知矢量/张量拒绝猜测处理。
4. 默认沿用 MW `Header/Time`。恒星字段是形成时刻，不是年龄，采用 `tform_new=tform_old+tmerge-tsource`，保证 `tmerge-tform_new=tsource-tform_old`。可用 `--stage2-start-code 0` 重置总时钟，三者都会同步平移恒星时间。
5. 保留 parent/child/generation IDs。允许分裂产生的重复父 ID，但要求同一 PartType 内三元组身份唯一。卫星沿用原脚本的 1亿、2亿预留 ID 区间。气体转成恒星后，来源身份仍通过实际 ID 追踪，不假定它们只在原老恒星 ID 段。
6. 不把缺失的物理字段随意补零。三者必须是同一套物理配置的真实演化输出，具有兼容数据结构。部分分析标签允许补 0 表示未分类。

**“保留快照字段”不等于二进制 restart。** flag2 会重新建立力、邻居、时间步及部分内部子网格状态；新势场/介质也会产生物理响应。快照未输出或源码明文重初始化的内部状态，HDF5 拼接器无法恢复。

## 5. 先检查 flag2 读入，再正式运行

```bash
bash ./mw_lmc_smc_relaxed_trial00_000_run/smoke/run.sh

python merge_relaxed_mw_lmc_smc.py verify \
  --reference-ic ./mw_lmc_smc_relaxed_trial00_000.hdf5 \
  --snapshot ./mw_lmc_smc_relaxed_trial00_000_run/smoke/output/snapshot_001.hdf5
```

smoke 参数在极接近初始时刻输出第一帧。verify 按复合粒子 ID 对齐，而不依赖粒子顺序，核对金属丰度、恒星形成时间及 BH 质量。如果发生变化，会列出数量和最大差异；要结合日志区分错误初始化、真实反馈和转星，不会武断地一律归因于 I/O。若你的源码改变了输出编号，请选择 smoke 的第一帧；超过初始时刻 `1e-6` 代码时间的快照会被拒绝作为纯读入检查。

通过后，从同一合并 IC 启动正式演化：

```bash
bash ./mw_lmc_smc_relaxed_trial00_000_run/production/run.sh
```

**不要将这个合并文件再次用 flag0 启动。** production 仍然保留 base-param 的 `InitMetallicity` 等参数文本，但在你提供源码的 flag2 路径里不应该覆盖读入的 Z 和恒星形成时间；因此无需删除这些参数，且应以上述实际检查确认。

已有输出的启动目录会阻止误重复启动。长任务中断后的继续运行属于 GIZMO restart 流程，不能直接再次执行这里的“全新运行”脚本覆盖原输出。

## 6. 轨道与累计 shooting

保留原卫星脚本的 K13 目标位置/速度、双卫星近似引力、Chandrasekhar 摩擦和 MW 反冲选项。不同之处是使用 **三个真实弛豫快照的质量分布** 构建轨道初猜，而非继续假定卫星保持最初的解析剖面。质量包含输入文件中所有保留粒子；没有自动剔除不束缚外流。如果反馈造成明显质量流失或大范围外流，应先检查卫星末态，再解释该近似轨道。

默认 `--lookback-gyr 1.0` 指合并后正向演化约 1 Gyr 到目标终点；不是把卫星的独立弛豫时间加到轨道上。可显式设为 1.5，但需要重新反演和校准。

原 K13 目标保持不变：

| 系统 | 目标位置 kpc | 目标速度 km/s |
|---|---|---|
| LMC | (-0.8, -41.5, -26.9) | (-57, -226, 221) |
| SMC | (15.3, -36.9, -43.3) | (19, -153, 153) |

上述数值和原始参考坐标约定均继承你提供的脚本；不是本次重新做的观测坐标拟合。MW 的盘方向应与该 Galactocentric 坐标约定一致。

静态近似势的数值前后向闭合不是 live GIZMO 的物理轨道通过。

### 本次 trial01 审计

你贴出的终点误差：LMC 从 28.676 kpc / 124.525 km/s 降到 20.593 / 81.325；SMC 从 29.981 / 74.603 降到 13.517 / 45.038。12维按各分量 sigma 缩放后的范数由31.901降至24.145。修正有作用，但每个分量必须在3 sigma内才通过；LMC x由−14.61变为−16.79 sigma，仍有局部恶化。这个加权范数不是考虑完整观测协方差的统计置信度。

旧代码残差符号与累计 offset 正确，不是“没有读取上次修正”。旧更新相当于
`x_new = x_base + B(y_target - alpha*r) - B(y_target)`，其中 B 为固定静态模型的反向轨道，r为live模拟值减目标值。它总在固定参考终点计算响应，不是live系统的Newton法。

v1.2 默认 `--shooting-method local-defect`：先从当前 `x_base=nominal+previous` 正向计算静态终点 `S(x_base)`，再用
`increment = B(S(x_base)-alpha*r) - B(S(x_base))`。
减去同一数值基点可避免将数值闭合误差加入修正。首次校准基点为nominal时两种方法近似相同；因此不能倒推说trial01未通过就是旧实现出错。改进主要用于后续迭代。

四个三维块的增量采用同一个缩放系数限步，默认每颗卫星位置不超过20 kpc、速度不超过30 km/s；保留12维修正方向。这是可调数值步长，不是观测约束。随后只在静态模型上预测加权残差变化，必要时减半步长。**模型预测下降不保证下一轮GIZMO下降，更不保证trial02通过。** `--shooting-method legacy` 可准确复现旧更新；该模式不做上述限步。local-defect不允许同时附加手动offset，以免预测与最终IC不一致。

### 这次实际 CSV 揭示的示踪问题

`magellanic_validation_summary.json` 给出 LMC STAR范围120000000–120004999、SMC范围210000000–220019933。原add脚本的SMC原生恒星从210000000开始，气体从220000000开始。旧合并器取弛豫后全部PartType4的min/max写作STAR范围；一旦有气体转星，就把两个ID子区间连在一起。

实际历史中LMC一直5000颗；SMC参考初态2048颗（由计数/比例反推），首帧2106、终点3938，`tracer_fraction=1.9228515625`。这是示踪**数量**增长92.3%，不是质量增长92.3%。旧`fraction>=0.999`检查还会让新生星掩盖原示踪星缺失。仅凭CSV无法求出正确SMC中心，必须重读已有HDF5。

v3.2默认 `--stellar-tracers original-stars`：在明确的原生恒星ID段内，从reference IC冻结具体粒子身份，运行时只匹配这个集合；有child/generation字段时一起匹配，不能用后来的新星补数。原始恒星缺失会降低保留比例，重复或不完整身份字段会明确报错。原生老恒星是本模型选定的轨道中心示踪群体，并不声称它等于全体系质量中心或唯一观测中心。

`--stellar-tracers legacy`可复现旧按范围选星的6D，便于对照；原模式与legacy模式请输出到不同目录。原模式的初始孔径质量也改用与运行相同的收缩核心和孔径设置，因此初始retention不再由不同中心定义造成偏移。新生恒星仍完整保留在真实模拟中，参与引力与反馈，只是不再加入这组固定轨道示踪粒子。

合并器v1.3写出的STAR范围恢复为原生老星范围；完整弛豫恒星范围另存ALL_STAR元数据。旧trial00/01 IC不用改写，新validator能识别本项目的ID布局并直接读取它们。MW速度定义暂时保留原validator的内10kpc质量加权方法；它与插入器内5kpc中位数不完全一致，这不是此次移植新产生的问题。

本次history还显示 LMC–SMC在0.7822 Gyr输出帧间距9.518 kpc，0.8800 Gyr已变为18.417 kpc且相对径向速度反号。输出较稀疏，不能把9.518当作真正最近距离；近遇过程说明轨道响应可能强非线性。当前图像的速度轴拥挤则是独立绘图错误：旧sharex='col'把0–1 Gyr时间与六分量类别0–5共轴；新版将其分开，不改变数值残差。

### 用已有 trial00/01 继续，不用重做三个星系的弛豫

把本包中的merger、validator、shooting_local三个文件放在原脚本目录；保持原 `tg19_backend.py`、`satellite_relax_common.py`、GIZMO二进制和base-param物理设置。先在 `~/gizmo-runs/mw_isolated` 对已有两轮统一重新诊断（仅后处理）：

```bash
for trial in trial00 trial01; do
  python validation_dashboard_sat_6d_gaswake_v3.py \
    --snapshot-dir "./mw_lmc_smc_relaxed_${trial}_000_run/production/output" \
    --reference-ic "./mw_lmc_smc_relaxed_${trial}_000.hdf5" \
    --stellar-tracers original-stars \
    --output-dir "./mw_lmc_smc_relaxed_${trial}_000_run/production/output/magellanic_validation_fixed"
done
```

检查新的原生星计数、tracer_fraction和old-minus-legacy对照。终端会输出 `||r/sigma||_12D`；在同一新口径下优先从分数较小且核心追踪稳定的一轮继续。下面示例假定trial01仍较好，若trial00更好则base和CSV一起改为trial00。新旧口径的残差不能直接用来声称迭代改善；只有在同一新口径下比较trial00与trial01才有意义。旧CSV留在原目录，校准时改用`magellanic_validation_fixed`中的新CSV。

诊断同时生成 `magellanic_present_6d.provenance.json`，绑定CSV内容、参考IC、目标时刻和中心测量方法。它能检查残差与reference IC配对，但不能替你证明production目录确实由该IC运行，仍需选对目录。目标时刻未覆盖的最近帧表只能查看，不可用于shooting；单个恰好在目标时间的快照可正确用于终点测量。

生成下一轮：

```bash
python merge_relaxed_mw_lmc_smc.py next \
  --base-param ./mw_isolated.param \
  --shooting-base-ic ./mw_lmc_smc_relaxed_trial01_000.hdf5 \
  --shooting-residual-csv ./mw_lmc_smc_relaxed_trial01_000_run/production/output/magellanic_validation_fixed/magellanic_present_6d.csv \
  --shooting-method local-defect \
  --shooting-alpha 0.3 \
  --shooting-max-position-step-kpc 20 \
  --shooting-max-velocity-step-kms 30 \
  --require-shooting-provenance \
  --output ./mw_lmc_smc_relaxed_trial02_000.hdf5 \
  --trial-label trial02
```

`next` 从trial01 IC内的记录找回三个源快照、卫星弛豫目录、盘面、目标、lookback、起止时间、摩擦/反冲参数；不用再猜LMC_RUN和SMC_RUN路径。它读取相同三个弛豫末态重新合并，继承trial01累计修正，再添加本次增量。它不会从trial01的最终snapshot_011接着演化。旧文件路径若已经搬家，可继续用普通 `merge` 命令显式指定新路径，源内容和模型必须一致。不要手改弛豫manifest里的结构数据。

生成后检查终端的每颗卫星增量、模型预测和 `.orbit.json` 的 `shooting_diagnostics`，再按原流程读入与演化：

```bash
bash ./mw_lmc_smc_relaxed_trial02_000_run/smoke/run.sh
python merge_relaxed_mw_lmc_smc.py verify \
  --reference-ic ./mw_lmc_smc_relaxed_trial02_000.hdf5 \
  --snapshot ./mw_lmc_smc_relaxed_trial02_000_run/smoke/output/snapshot_001.hdf5
bash ./mw_lmc_smc_relaxed_trial02_000_run/production/run.sh
python validation_dashboard_sat_6d_gaswake_v3.py \
  --snapshot-dir ./mw_lmc_smc_relaxed_trial02_000_run/production/output \
  --reference-ic ./mw_lmc_smc_relaxed_trial02_000.hdf5
```

每轮都用本轮IC和本轮真实终点、相同固定示踪定义下的CSV。若之前已经使用v1.2旧示踪流程，重新测量同一base后仍可保留其实际初态及offset继续；脚本记录测量方法的切换，不会把旧残差再次相加。不要再把trial00残差加一遍：累计offset已包含其影响。改变源快照、盘面、目标、lookback或近似引力模型时，不能沿用旧offset。

保存的名义参考曲线不含累计修正；早期 `d6` 大不代表初态没赋好。shooting约束的是今天的6D终点，过去1 Gyr的名义静态轨道不是观测真值。

若后续live加权误差持续停滞/增大，应回到误差较小的base、降低alpha或通过受控初态扰动估计live响应。trial00/01仅约束12维响应中的一个方向，不能从两次结果推断完整Jacobian，更不应通过放宽sigma人为让验收通过。

## 7. 实际采用的初始结构

数值直接来自你上传的 `add_lmc_smc_gaswake_orbit_v3.py` 内部 **v6 TG19** 分支，不是另造一个同名模型。

| 参数 | LMC | SMC |
|---|---:|---:|
| 截断 NFW 暗晕质量 Msun | 1.3e11 | 1.3e10 |
| 暗晕 rs / Rtr kpc | 13 / 75 | 3.8 / 28 |
| 恒星质量 Msun | 2.5e9 | 8.0e8 |
| 恒星 Rd / zd kpc | 2.4 / 0.48 | 1.2 / 0.25 |
| 恒星 Rmax / zmax kpc | 9.6 / 2.4 | 4.8 / 1.25 |
| 冷气体盘质量 Msun | 2.0e9 | 2.0e9 |
| 气体 Rd / Rmax kpc | 9.5 / 38 | 4.7 / 19 |
| 气体初始温度 K | 1e4 | 1e4 |
| 生成器原默认单粒子质量（本次next直接保留快照实际质量）Msun | 1e5 / 1e6 / 1e5 | 1e5 / 1e6 / 1e5 |
| 独立初始 Z/Zsun | 0.2 | 0.1 |
| 初始气体 / DM / 恒星粒子数 | 20000 / 130000 / 25000 | 20000 / 13000 / 8000 |

卫星不含 BH，不额外叠加 C25 热 corona。恒星盘使用原分支的双指数垂直分布；冷气体盘高度由原脚本的近似静水平衡求出，并受数值上下限约束。采样采用对称粒子对，初始平均整体速度近零，但这不等于体系在显式冷却/反馈下已经稳定。

经过独立弛豫的冷却、恒星形成和反馈后，**实际冷气体质量可能明显偏离 2e9 Msun**。合并器保留这个演化结果，并在 JSON 中报告气体质量、半径分位数、质量加权 Z；若有 Temperature 字段，还报告 T<2e4 K 的气体质量。不会为了“增强冲压”把消耗的气体补回，也不会重新缩放卫星质量。

## 8. 验证范围与解释边界

交付前已运行：

- 所有 Python 文件的语法编译和命令行入口检查；
- 两个卫星按默认完整分辨率生成 IC，并检查粒子数；
- 当前31项合成HDF5/数值测试覆盖字段保留、累计offset、next继承、CSV/IC错配、原生星元数据、非线性重新定心、12维耦合、限步/回退、单位和DF/反冲正反积分；validator回归覆盖固定示踪、防止新星掩盖旧星缺失、单帧终点、legacy一致性、孔径中心一致及时间轴/类别轴分离；
- 模拟 flag0 覆盖 Z 的错误，确认 verify 能识别；
- 启动脚本 `bash -n` 检查。

你可重复执行：

```bash
python -m unittest -v test_pipeline test_shooting_local test_validator_audit
```

**未在交付环境运行真实 GIZMO，也未读取你的实际 MW snapshot_007、LMC/SMC 末态。** 合成测试不能证明星系稳定性、轨道校准或气泡科学结论。因此随包提供实际 flag2 smoke 检查，遇到本地编译的额外字段会明确报出，不会悄悄补成 0。

独立弛豫后拼接适合你当前“不重建二体历史、研究卫星带来的 CGM 扰动”的受控实验，但演化到指定时间本身不是稳定性证明。应检查末段气体质量、盘厚、流出速度、温度和 SFR 是否仍快速变化；拼接瞬间加入 MW 外势和环境介质仍可能引入暂态。最好用同一 MW 末态的无卫星运行作为对照，以区别卫星扰动、MW 自发演化和反馈的贡献。

旧 MW 若确实在错误的零金属条件下演化过，这套合并器不会修复已经发生的冷却历史；应选择你认可的正确 MW 弛豫末态。

## 9. 代码来源

- 用户附件 `add_lmc_smc_gaswake_orbit_v3.py`：SHA256 `db8af7a36519779d727f72f004e6ef394161102cf12b0afac38582610d657bac`。提取结构、采样、K13 轨道和 CSV/累计 offset 读取函数；仅在卫星引力接口增加实测质量剖面支持。旧 IC writer、旧 EQS/反馈政策没有复用。
- 本次重新上传的 `add_lmc_smc_gaswake_orbit_v3(2).py` 与上述卫星源文件字节一致；`make_ic_live_mw_cgm_isolated_v3(2).py` 与上述 MW 源文件字节一致，已核对 SHA256。
- 用户附件 `make_ic_live_mw_cgm_isolated_v3.py`：对照其单位、粒子类型、MW BH ID 和卫星预留 ID 布局；不重建或覆盖已经弛豫的 MW。
- 用户提供的 `read_ic.c`、`init.c`、`begrun.c`、`allvars.h` 和手册：用于核对本地版本的 flag0/flag2、初始 Z、恒星形成时刻、输入 Header 和输出编号逻辑。

本包保留原脚本中的 TG19/K13 归属说明，但并未将这次独立弛豫流程宣称为对 Oprea 或 TG19 论文完整实验的复现。
