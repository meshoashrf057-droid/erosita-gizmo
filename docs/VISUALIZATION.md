# 可视化入口与版本说明

本仓库保留用户原有绘图程序。新增 tools/render_v10.py 只负责选择快照、传递参数、加载兼容字段和管理输出；密度、温度、X-ray 图与视频仍调用原 erosita_bubble_xray_yt_v10_log.py 的函数。它对应近期使用的 v10_log 风格，不自动替换色标或绘图算法。

以下命令从仓库根目录执行。把 /Volumes/SimulationData/... 换成 Mac 上实际的数据路径；含空格的路径加引号。大快照、图片、视频和运行缓存应存放在代码仓库外。

## 44 / 200 kpc 图

先查看参数：

~~~bash
python tools/render_v10.py --help
~~~

单帧、两个视场：

~~~bash
python tools/render_v10.py \
  --snapshot-dir /Volumes/SimulationData/output \
  --output-dir /Volumes/SimulationData/figures_snapshot010 \
  --indices 10 --fovs 44 200
~~~

最后两帧需要显式填写编号，例如 9 与 10；本入口没有隐含的“最新两帧”规则：

~~~bash
python tools/render_v10.py \
  --snapshot-dir /Volumes/SimulationData/output \
  --output-dir /Volumes/SimulationData/figures_snapshot009_010 \
  --indices 9 10 --fovs 44 200 \
  --reference-ic /Volumes/SimulationData/orbit_000.hdf5
~~~

连续范围与视频：

~~~bash
python tools/render_v10.py \
  --snapshot-dir /Volumes/SimulationData/output \
  --output-dir /Volumes/SimulationData/figures_sequence \
  --start 0 --stop 10 --stride 1 --fovs 44 200 \
  --make-video --fps 8
~~~

--start / --stop 是包含端点的快照编号范围；--stride 对范围内存在的文件逐项抽取。--indices 支持任意单个或多个编号，不能同时设置 --start、--stop 或非默认 stride。不提供这些选择参数时处理目录内全部符合命名规则的文件。当前仅支持完整的单文件 snapshot_NNN.hdf5，不支持拆分成 snapshot_NNN.0.hdf5 等的快照。

每帧生成密度、温度、X-ray 三张独立图；两个视场分别保存在 fov44/ 与 fov200/。视频也按物理量分别生成，共两个视场 × 三种物理量；默认文件名如 fov44/density_morphology_v10_log.mp4。视频帧按快照编号排序，每帧停留时间固定为 1/fps，不按物理时间差重新采样。视频沿用原程序的 ffmpeg/libx264 编码；没有 ffmpeg 时保留 PNG、提示跳过视频，并在 manifest 中记录未生成。

## 默认科学定义

| 项目 | 新入口保留的 v10_log 默认值 |
| --- | --- |
| 视线方向 | x 轴，图面为 y-z；不会重新旋转整个快照 |
| 44 / 200 kpc | 图像的**完整宽度**，不是半径或半宽 |
| MW 中心 | 原函数优先使用 MW BH ParticleID；失败时依次使用其他 BH / 宿主恒星 / 气体回退中心 |
| 银盘遮罩 | 0 kpc，保留银盘；包装器不提供遮盘参数 |
| 密度 | 穿过 MW 中心的切片；原自动对数色标 |
| 温度 | 穿过 MW 中心的切片；固定 1e5–3e8 K 对数色标 |
| X-ray | APEC 0.6–1.0 keV、本征发射率沿完整深度积分；原自动对数色标 |
| X-ray 归一化 | 没有前景吸收与仪器响应，也未除以 4π；不是直接可与观测计数率等同的全天图 |
| 图像采样 | 1200 × 1200，保存 DPI 180 |

--reference-ic 可补充 MW_BH_ID 和源粒子身份元数据，快照自身元数据优先。这里的中心确定只用于 MW 图像居中，**不是 LMC/SMC 的 12D 轨道测量**。每帧实际采用的中心和回退方法写入 render_manifest.json。

默认标注快照的绝对物理时间。若要显示某一阶段开始后的 Myr，可以额外传入该阶段真实的 code-time 起点：

~~~bash
python tools/render_v10.py \
  --snapshot-dir /Volumes/SimulationData/output \
  --output-dir /Volumes/SimulationData/figures_elapsed \
  --indices 10 --fovs 44 200 \
  --time-origin-code 0.6136272158015191
~~~

上述起点只是曾用落入阶段的例子，不能直接套用到其他落入或 jet 阶段；程序按数据集时间单位换算，不硬编码“一 code-time = 一 Gyr”。

支持的显式显示调整包括：

- --buff-size、--dpi。
- --temperature-mode slice|slab；slab 使用原程序的密度加权温度投影，--slab-half-depth 指 x 方向半厚度，单位 kpc。
- --temperature-zlim MIN MAX，单位 K。
- --density-color-scale auto|fixed 与 --density-zlim MIN MAX，单位 g/cm³。
- --xray-color-scale auto|fixed 与 --xray-zlim MIN MAX，单位 erg/(s·cm²)。
- --apec-data-dir、--make-video、--fps、--video-name。

提供密度或 X-ray 的 zlim 时，也要显式选择 fixed 才会使用该范围。默认 auto 会逐帧选择色标范围，因此形态视频中的颜色不能直接当作固定表面亮度标尺。若要跨快照定量对比，使用同一物理定义与固定色标；这不会改变快照物理量。

## 兼容加载与输入保护

包装器经根目录的 gizmo_compat.load_gizmo 加载数据，要求存在可用的原生 KernelMaxRadius 或 SmoothingLength，关闭邻居搜索重建核长。单帧临时缓存放在输出目录的 .yt-frame-.../yt_cache/，帧结束后清除；Matplotlib 缓存放在输出目录的 .mpl-cache/。程序以只读方式打开输入 HDF5，不把兼容字段写回原快照。

输出 manifest 记录原绘图源文件 SHA256、所选快照、实际中心、图像参数、输出列表、起止时间，以及输入文件处理前后的大小/mtime。大小/mtime 检查用于发现绘图期间输入变化，**不代替整个快照的哈希归档核验**。请选择已经写完的快照。

同一输出目录已存在 manifest 或所选 PNG 时，程序拒绝覆盖；另选一个输出目录再执行。保留失败的 manifest 便于查因，不会启动或续跑 GIZMO。

## 保留的其他版本

| 文件 | 用途与区别 |
| --- | --- |
| erosita_xray/erosita_bubble_xray_yt_v10_log.py | 原 v10_log 算法与色标来源；直接入口使用原加载流程。跨机器优先使用上述兼容包装器。 |
| erosita_xray/erosita_bubble_xray_yt_v10_1_native.py | 文件内部版本为 v10.3，原生兼容加载入口；默认 200 kpc / 800 像素，密度、温度与 X-ray 默认采用 yt 色标策略，**与 v10_log 的默认显示不同**。支持 --no-panel 和固定注入面板参数。 |
| erosita_xray/allsky_xray_yt_v3_disk_preserved.py | 保留银盘的全天图流程，默认遮罩 0 kpc、网格半径 40 kpc、256³、0.6–1.0 keV；太阳偏移 [-8.29, 0, 0] kpc，默认 offplane-max 显示色标。 |
| erosita_xray/allsky_xray_yt_v4_disk_masked_audit.py | 带遮盘审计的全天图流程，默认遮罩半厚度 5 kpc、网格半径 30 kpc、256³；提供 --emissivity-reconstruction luminosity|normalized，默认 luminosity。与 v3 的遮罩、体积和重建选择要区分。 |
| 其他 allsky / logluminosity / v2 文件 | 历史方法保留用于复现；不能仅因版本号较旧就认作相同的科学定义。 |

使用 native 入口前先查看其独立帮助，不把新包装器的 --fovs、--indices 参数直接传给原脚本：

~~~bash
python erosita_xray/erosita_bubble_xray_yt_v10_1_native.py --help
python erosita_xray/allsky_xray_yt_v3_disk_preserved.py --help
python erosita_xray/allsky_xray_yt_v4_disk_masked_audit.py --help
~~~

v3 保留银盘的全天图示例：

~~~bash
python erosita_xray/allsky_xray_yt_v3_disk_preserved.py \
  --snapshot-dir /Volumes/SimulationData/allsky \
  --output-dir /Volumes/SimulationData/allsky_figures \
  --grid-dir /Volumes/SimulationData/allsky_grids \
  --apec-data-dir erosita_xray \
  --bh-id 90000000 --start 1 --stop 30
~~~

这里的 BH ID 同样必须匹配实际 MW。全天图具有独立的 --grid-radius-kpc、--grid-n、--nside、--n-los、--los-min-kpc、--los-max-kpc、--sun-offset-kpc、能段和显示色标选项。--reuse-grid / --reuse-allsky 用于已存在的中间产物；变更输入或物理参数时应使用新的输出/网格目录，避免混用缓存。

## 依赖与移植范围

- 保持仓库目录层次：根目录的 gizmo_compat.py、gizmo_units.py、gizmo_schema.py 等共享模块不能从 erosita_xray/ 中单独拆走。
- 绘图主要依赖 NumPy、SciPy、h5py、yt、unyt、Matplotlib 和 Pillow；全天图另需 healpy。以仓库环境安装说明为准。
- X-ray 需要 erosita_xray/apec_emissivity_v3.h5；--apec-data-dir 指包含该文件的目录。这是发射率表，不是模拟快照。
- 视频需要系统 ffmpeg，编码器需支持 libx264。
- 这批入口没有专用字体文件路径依赖；不同系统的 Matplotlib / 字体版本仍可能产生版式差异。
- TREECOOL 与 spcool_tables/ 是模拟冷却相关资源，不替代 APEC 绘图表。
- firelib.py 中部分旧 FIRE 示例函数依赖外部 Z*.dat 或可选 trident；pressure_at_surface.py 带旧机器的 FIRE 数据路径，不能作为通用命令直接执行。这些并非上述 v10_log / allsky 主入口的必要数据。
- Mac 上不需要沿用 WSL 的绝对路径。原始的一次性运行/归档脚本保留了历史路径与冻结哈希，属于可审阅的复现记录；修改自己的运行配置后再使用，不能直接当作新模拟启动器。

导出时仅做语法与 --help 检查，没有为迁移重新运行真实绘图、修改科学参数或读取大型快照数组。Mac 上的首次完整绘图仍需在安装环境和数据就绪后验证。
