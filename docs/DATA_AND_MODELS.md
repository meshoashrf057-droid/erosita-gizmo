# 数据与模型迁移

代码以外还需单独复制：要分析的 HDF5 快照、对应初始 IC、orbit sidecar 和来源身份元数据。建议数据放到 ~/Research/erosita-data/，新输出用独立目录，均不加入 Git。

若重做 shooting，必须带配对初始 IC、真正终点、残差 CSV 和 provenance；不能只用旧 proposal JSON 或旧 Jacobian。低气体新模型不能继承旧模型的轨道通过结论。

中断恢复还需要完整匹配的原生 restart 分片、参数与二进制；不应未经验证把 Linux restart 交给 Mac 新编译版本恢复。

## 小型资源

- erosita_xray/apec_emissivity_v3.h5：用户原 X 射线发射率表。
- resources/spcool_tables/：原 MW 冷却表，按 source_manifest 保存 SHA。
- reference_configs/*/TREECOOL：各原目录的冷却输入。
- firelib.py 的部分旧 FIRE 功能另需 Z*.dat 等表，当前项目缺少这些资源，未伪造。
- trident 属可选旧分析依赖，需要按其说明独立配置离子表；基础 yt/allsky 不强制安装。

GIZMO 运行时须将冷却资源放到参数/编译设置实际要求的位置。reference_configs 中的文件只是参考；InitCondFile、OutputDir、表路径、二进制路径都需根据 Mac 数据位置设置。

## GIZMO 与喷流

本仓库没有完整 GIZMO 源树或 Linux 二进制。应另行带走对应原源树及修改记录，再用 Mac 编译器、MPI、HDF5、GSL 等构建。旧 Makefile 的 Ubuntu 路径只供参考；记录新二进制自己的 SHA，不能让它冒充旧 SHA。

reference_configs/infall/Config.sh 是原落入配置。jet_reference/fixed_supply_cpp_20261115 包含固定外部供给移植补丁、准备代码、Config/Makefile 和审查记录；reference_configs/jet_fixed_supply_dt001/stage3.param 是 MaxSizeTimestep=0.001 的实际喷流参数参考。SINK_DEBUG_FIX_MDOT_MBH 对应该实验模型；迁移不能悄悄改成自然吸积或混用落入/喷流二进制。

当前 gas19 初始 LMC/SMC 气体质量为 1.10e9 / 0.79e9 Msun，入口在 mw_isolated/cgm3_n3000_gas19_20261116/code/。其它物理参数保留原设置。本次代码整理没有启动模拟。

provenance/source_manifest.json 记录每份原脚本/资源的 SHA，历史路径只用于溯源。
