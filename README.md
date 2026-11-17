# eROSITA / GIZMO 科研流程代码

MW–LMC–SMC 初值、弛豫与合并、轨道校准、数值诊断、旋转曲线、yt 可视化、allsky 以及固定供给喷流移植的 Python 工具。整理日期：2026-11-17。

原科研脚本按原目录层级复制，保留不同版本，没有修改模型物理或观测目标。新增 tools/ 和 docs/ 用于路径迁移与使用说明。

## 在 Mac 上开始

```bash
git clone https://github.com/meshoashrf057-droid/erosita-gizmo.git
cd erosita-gizmo
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tools/doctor.py
python tools/verify_export.py
python tools/render_v10.py --help
```

Mac 系统依赖及替代安装方式见 [Mac 安装说明](docs/MAC_SETUP.md)。doctor 与 verify_export 不启动 GIZMO、生成初值或绘制科学图。正式模拟流程不要求额外烟测。

## 入口说明

- [完整流程](docs/WORKFLOW.md)：初值、弛豫、合并、落入、12D、旋转曲线、结构诊断、喷流准备。
- [可视化与 allsky](docs/VISUALIZATION.md)：用户原 v10_log、44/200 kpc、视频，以及全天图各版本区别。
- [数据和模型](docs/DATA_AND_MODELS.md)：需另行迁移的快照、源树和历史轨道证据。

| 目录 | 内容 |
|---|---|
| 根目录 gizmo_*.py | 旧 C / 新 C++ HDF5 字段、单位、初值和 yt 兼容层 |
| mw_isolated/ | MW 初值、合并、旋转曲线、原始恒星 12D、shooting |
| erosita_xray/ | 用户可视化、数据收集、角动量设置、全天图和气体诊断 |
| mw_isolated/cgm3_n3000_gas19_20261116/code/ | 当前低气体版本生成器及当时流程参考 |
| mw_isolated/*/scripts/、shooting_local_live_v1/ | 原实验流程、数学与证据工具；不能当作 Mac 自动启动器 |
| jet_reference/ | 固定外部供给喷流的移植补丁、准备脚本、编译配置 |
| reference_configs/ | 原模型参数与配置参考 |
| resources/spcool_tables/ | 原冷却表 |
| tools/、docs/ | 可移植绘图入口、环境检查、说明 |
| provenance/、environment/ | 来源 SHA、源 Linux 包版本和可选 conda 环境 |

## 版本和迁移边界

本轮新增 gas19/recovery_20261117_02、gate_endpoint_resume02 和 review_tools 内的恢复及 alpha 校准参考入口，实际保存于 mw_isolated/cgm3_n3000_gas19_20261116 下；这些脚本绑定原 Linux 运行路径与证据，不能直接作为 Mac 启动器。

当前低气体 LMC/SMC 使用 gas19/code 内的生成器，显式设置气体总质量。顶层原生成器仍保留旧模型，不能自动替代。12D 优先使用 mw_isolated/validation_dashboard_sat_6d_gaswake_v3.py；erosita_xray 同名文件较老。

原 v10_log 与 v10_1_native 色标不完全相同。新增 render_v10 调用用户原 v10_log，只适配加载与路径。不同 allsky 的盘掩膜和光度定义也不自动互换。

历史一次性 launcher 保留当时路径、SHA、tmux 和 /proc 约束，只供溯源；Mac 上应采用通用入口与新运行目录。少数无 CLI 的旧脚本需要先修改文件顶部的数据目录，具体见说明。

仓库不包含模拟快照、restart、运行输出、Linux 虚拟环境或 GIZMO 可执行文件。GIZMO 完整源树需单独迁移并在 Mac 原生重新编译。喷流固定供给需要 jet_reference 中的补丁和相应 Config，不能只复制 erosita.param。

已在现有 WSL 环境检查语法、文件完整性和可用入口；没有在真实 Mac 上完成科学运行验证。未新增开源许可，原作者及第三方资源的权利和署名保留。
