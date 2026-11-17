实测双 y 联合候选 v1 — 2026-10-15

用途
====
根据 shooting_response_audit_04_09.json 中 trial06、08、09 的实测终点，生成一个新的 trial10 初始条件。
这是候选实验，不是已通过的轨道。没有在此环境运行用户的 GIZMO 演化。
只增加 LMC 与 SMC 的整体初始 y 坐标，速度、内部结构、粒子字段和原物理设置保持原流程。

数据与结论
==========
trial06：norm=6.14257055，Ly/sigma=-3.99110674，Sy/sigma=+3.06717188。
trial08：norm=5.98275963，Ly/sigma=-3.49261815，Sy/sigma=+3.97128951。
trial09：norm=6.74929056，Ly/sigma=-4.20161389，Sy/sigma=+3.52020301。
08、09 均从 06 分支；08 仅 Ly+1 kpc，09 仅 Sy-1 kpc。历史记录来源核对通过。
记录核对不等于在这里重新读取了大型 HDF5；应用补丁时会检查用户本地基准 IC、CSV 的实际 SHA256。

终点 y 对初始 y 的实测割线响应 (kpc/kpc)：
                  初始 LMC_y      初始 SMC_y
终点 LMC_y          +0.94347569      +0.39842111
终点 SMC_y          +1.58999880      -0.79670932
SMC 列 = (r06-r09)/1 kpc，不能忘记 trial09 的步长为负。

全 12 分量模型：z(a,b)=z06 + J@[a,b]。
a、b 是相对 trial06 的 LMC、SMC 初始 y 总偏移，单位 kpc。
用全部 12 个残差求最小最大绝对分量，解约 (1.12841012,2.71635085)，取整为 (1.13,2.72)。
不只消除两个 y：若强行令两个 y 预测为零，将需约 (2.79,12.34) kpc，其他分量会恶化。

应用基准选择目前实测 norm 最小的 trial08，因此本次只新增：
LMC dpos=[0,0.13,0] kpc，dvel=[0,0,0] km/s；
SMC dpos=[0,2.72,0] kpc，dvel=[0,0,0] km/s。
不能以 trial08 为基准再加 LMC_y=1.13；那会重复加上 probe 已有的 1 kpc。

预测（未通过 live 验证）
======================
LMC z=[-2.85837523,-2.85523517,-0.72486913,-0.64881948,-0.41038990,0.15143688]
SMC z=[+1.99246809,+2.85658014,-0.14963818,-0.18996159,-1.56756600,1.59270503]
norm=5.88581859，max|z|=2.85837523。
目标是降低最差分量，所以总 norm 的预测下降较小。
SMC +2.72 是从 -1 kpc 测量反向外推，联合非线性响应和模拟噪声均未标定。
预测最近的 3sigma 边界为 LMC x，余量仅约 0.05615 kpc。不能保证通过，不能据此放宽容差。

安装与生成
==========
将 shooting_joint_y_v1.zip 放进 ~/gizmo-runs/mw_isolated，执行：

cd ~/gizmo-runs/mw_isolated
unzip -n shooting_joint_y_v1.zip
bash build_trial10_joint_y.sh

build 脚本只生成 IC 与运行文件，不启动 GIZMO。完整生成指令写在该脚本中。
新 merger 文件名是 merge_relaxed_mw_lmc_smc_joint_y_v1.py，原 merger/backend 不替换。
需要保留原目录的 tg19_backend.py、satellite_relax_common.py、shooting_local.py，以及同一套弛豫源快照与原始 base-param。
需要 trial08 IC、trial08 的 fixed 残差 CSV 及对应 provenance；不能换成 trial09 或其他验证策略。
遇到指纹不匹配，应查文件版本/路径；不要编辑 proposal 的指纹以强行通过。

成功生成后核对打印增量应为 Ly+0.13、Sy+2.72，全部速度增量零。
按原流程：

bash ./mw_lmc_smc_relaxed_trial10_000_run/smoke/run.sh
python merge_relaxed_mw_lmc_smc_joint_y_v1.py verify --help

smoke 中的实际快照命名应沿用原有近零时刻检查方式。以下 production 和 smoke 均使用既有 flag2 流程：

bash ./mw_lmc_smc_relaxed_trial10_000_run/production/run.sh

production 完成后：

bash validate_trial10_joint_y.sh

将 shooting_response_audit_04_10.json 提供回来。该命令使用 actual parent 08→10，不按 trial09→10 错算增量。
若同名审计输出已存在，导出器会停止，需在命令中选择新的输出名；不会覆盖旧历史。

实现与验证
==========
新增 measured-joint-y 模式独立于单坐标 live-probe，不改变 live-probe 原有的 2 kpc 限制。
联合模式每个 y 增量以及相对 trial06 总偏移均限 3 kpc；候选固定，不再乘 alpha，不叠加手动 offset。
核对模型 key、base IC / CSV SHA256、基准初态与残差、sigma、两个 probe 的增量符号与初态、响应矩阵和预测。
重新从同样三个弛豫末态合并，继承 trial08 累计位移与旋转，不从 trial08 的落入终态接着演化。
静态终点仅为兼容原审计记录而计算，分别存储；预测残差采用实测双列模型，等待 live 比较。
此补丁不修改金属丰度、年龄、反馈、星系结构、质量分辨率、目标时刻或观测容差。

验证完成：10 个单元测试；一次合成 HDF5 合并测试（含非零旧累计修正），确认仅两个 y 平移、所有速度及其他粒子数据完全保持，alpha 不重复应用，原 IC 不变；历史导出兼容检查通过。
这些是数值/I/O 检查，未运行 live GIZMO，不能替代轨道科学验证。
