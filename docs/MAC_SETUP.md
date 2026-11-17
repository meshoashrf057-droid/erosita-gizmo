# macOS 环境准备

这份导出用于迁移 Python 初态工具、读取器和后处理流程。以下命令从仓库根目录执行，适用于准备 Python 3.12 环境；**本次未在真实 Mac 上实测**。依赖能导入不等于全部历史脚本已完成 Mac 适配，也不证明科学结果与原环境一致。

## 1. 建立独立 Python 环境

先安装适用于本机的 [Homebrew](https://brew.sh/)，然后执行：

```sh
brew install python@3.12 ffmpeg coreutils
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Homebrew 提供 [Python 3.12](https://formulae.brew.sh/formula/python@3.12)、[ffmpeg](https://formulae.brew.sh/formula/ffmpeg) 和 [coreutils](https://formulae.brew.sh/formula/coreutils)。`ffmpeg` 用于动画编码；`coreutils` 提供历史脚本可能调用的 `sha256sum`。

如果终端找不到 `python3.12`，可用下列命令创建环境：

```sh
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv
```

让当前终端能找到仓库根兼容模块：

```sh
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

若 `command -v sha256sum` 未找到命令，按 coreutils 的安装说明在当前终端补充路径：

```sh
export PATH="$(brew --prefix coreutils)/libexec/gnubin:$PATH"
```

这里的路径设置只影响当前终端，不要求改写全局 shell 配置。后续打开新终端时，先回到仓库根目录并激活 `.venv`。

## 2. 运行环境检查

```sh
python tools/doctor.py
python tools/doctor.py --help
```

`doctor.py` 检查 Python 版本、固定清单中的第三方包导入、`ffmpeg`/`sha256sum` 是否位于 PATH、五个根兼容模块是否存在且语法可解析，以及默认 APEC 文件的 HDF5 标识。它不会导入项目科研脚本、读取快照数组、生成科学图像或启动 GIZMO。第三方包在独立 Python 子进程中导入，临时缓存位于临时目录。

返回码 0 表示没有 FAIL；WARN 和 SKIP 仍需要按用途判断。依赖缺失或加载失败会返回非零。可用 `--json` 输出结构化报告；它不会自动安装依赖或修复文件。

## 3. 数据与 APEC 表

本仓库包含 `erosita_xray/apec_emissivity_v3.h5`。doctor 默认只检查文件存在及 HDF5 标识，不验证发射率数值或替你更改各绘图脚本的表路径。

大体积初态、快照和重启数据需另行搬运。建议把数据放在仓库外，由你选择目录；例如在仓库同级建立 `erosita-data/`，再把实际数据放入其中。为相应脚本显式传入快照、参考 IC、输出目录和 APEC 路径，保留原始输入文件。

若要检查自选表：

```sh
python tools/doctor.py --apec-table ../erosita-data/apec_emissivity_v3.h5
```

带 `--apec-data-dir` 的脚本接收的是**包含表的目录**，而 doctor 的 `--apec-table` 接收的是**表文件路径**。doctor 选项不会替其他脚本设置参数。其余脚本可能使用 yt 的数据目录设置，运行前查看对应源码或文档。只有已经确认以 main guard 保护入口的脚本，才适合直接执行 `--help`；部分历史科研脚本导入或运行时会立即做计算。

根兼容模块应与 `mw_isolated/`、`erosita_xray/` 保持同一级，避免只拷贝单个绘图脚本后丢失其导入依赖。仓库中的冷却表不替代模拟所需的初态及物理配置。

## 4. yt、healpy 与 Apple Silicon

yt 官方建议使用隔离环境，并说明 HDF5 等读取器可能需要额外依赖；本仓库的 requirements 同时列出相关科学 Python 包。[yt 安装说明](https://yt-project.org/doc/installing.html)

healpy 含编译扩展，安装路径取决于 Python 版本、CPU 架构和可用 wheel。先按上面的 requirements 安装；若 pip 尝试源码编译并失败，按官方说明选择合适的二进制或独立 conda 环境，避免在同一个环境里混用架构或反复混装 pip/conda 核心库。官方文档同时给出 macOS、pip 和 conda 路径，并列出源码编译限制。[healpy 安装说明](https://healpy.readthedocs.io/en/stable/install.html)

Apple Silicon 上应使终端、Python 和编译依赖的架构保持一致。可检查：

```sh
uname -m
python -c "import platform; print(platform.machine())"
```

本指南不保证每个包版本都有本机架构 wheel；以当前安装输出为准。可选依赖见 `requirements-optional.txt`，仅在所选脚本需要时安装。

## 5. GIZMO 必须单独处理

**原 Ubuntu/WSL 的 Linux ELF GIZMO 二进制不能直接在 macOS 上运行。** Python 环境迁移不等于模拟可执行文件迁移。若要在 Mac 进行模拟，需要取得相应 GIZMO 源码，在本机重新编译并核对原模型的编译选项、MPI、HDF5、GSL 等实际依赖；仓库中的 Python 工具不能替代这一步，也不自动启动任何模拟。

已有本机编译结果后，可只检查路径及文件类型：

```sh
python tools/doctor.py --gizmo ../gizmo-build/GIZMO
```

这个命令**不会执行 GIZMO**，也不验证动态链接、MPI、多进程启动、物理配置或数值一致性。仅做 Python 后处理时，可以省略 `--gizmo`。

历史 campaign 启动器可能包含 WSL 路径、tmux、Linux 进程管理及一次性锁定逻辑；不要把它们当作 Mac 一键启动入口。模拟迁移应另设独立运行目录并明确选择模型及输入。

## 6. 本次验证边界

此次交付可以检查文本语法、导出文件布局和安全检查器的命令行入口。Mac 端依赖安装、真实快照读入、图像复现以及 GIZMO 本机编译仍需在目标 Mac 上验证。保留原环境及必要输入、配置与校验记录，便于比较结果。
