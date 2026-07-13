# 环境、连接与文档修正说明

> 日期：2026-07-13  
> 范围：W0 连接层修正、G1-03 真图验收策略、易误导文档的覆盖说明  
> 权威状态仍以 `PROGRESS.md` 与对应 `tasks/` 为准；本文解释**为什么**这样改，不是新的任务授权。

---

## 1. 背景：代理常把「有环境」说成「没环境」

### 1.1 实际环境（本轮在 WSL 内核实）

| 项 | 事实 |
|---|---|
| 执行位置 | **已在 WSL2 Ubuntu 24.04 内**（Grok CLI / bash） |
| 判定依据 | `/proc/version` 含 Microsoft；存在 `WSLInterop`；`python3` 3.12.3；`fcntl` 可用；`/opt/ros/jazzy` 存在；`carla` 可 import |
| 工程根 | `/mnt/e/autonomous driving` ↔ `E:\autonomous driving` |
| CARLA 安装 | Windows 构建：`E:\CARLA_0.9.16` ↔ `/mnt/e/CARLA_0.9.16` |

因此：**不是「没装 WSL」**，而是执行上下文或诊断逻辑判错。

### 1.2 常见误判来源

| 错误现象 | 真实原因 | 正确结论 |
|---|---|---|
| Windows 上跑 Python 报 `No module named fcntl` | 用了 Windows/Anaconda/embedded 解释器跑 Linux runtime | 换到 **WSL 内** `python3`，不是重装 WSL |
| PowerShell 找不到 `sdf` | 在 Windows 找 Linux 入口 | 在仓库根用 `python3 scripts/sdf.py …` |
| `doctor` 调 `wsl.exe` 失败 | 进程**已经在** WSL 内，却按「Windows 外层」套娃查 WSL | 已在 guest 时走本机 `/opt/ros/jazzy` |
| `process_state=NOT_RUNNING` / `RPC_HANDSHAKE_FAILED` | CARLA Server 未就绪或 host 错 | 启 Server / 修 host，**不是**环境没装 |
| 历史文档写「WSL 未安装」 | G0-01 **当日**盘点快照 | 不得再复制到新断点；见 `HOST_INVENTORY.md` 顶部覆盖说明 |

### 1.3 代理在真实 CARLA 任务前应做的最小自检

```bash
grep -qi microsoft /proc/version && echo IN_WSL=1
python3 -c 'import fcntl, carla; print("fcntl+carla OK")'
test -d /opt/ros/jazzy && echo ROS_JAZZY=1
cd "/mnt/e/autonomous driving"
python3 scripts/sdf.py sim status
python3 scripts/sdf.py sim preflight
```

**规则**：在输出上述客观证据之前，不得写「未安装 WSL / 环境不存在」。

---

## 2. CARLA Host：为什么不能写死 `172.30.80.1`

### 2.1 历史 IP 是什么

G0 证据与部分旧 README 里的 `172.30.80.1` 是**当时** WSL NAT 默认网关的一次采样。  
它在**那一天**能连上 CARLA，但：

- WSL 重启后网关可能变；
- 镜像网络（mirrored）下 Windows CARLA 常在 **`127.0.0.1:2000`** 可达；
- 本机还存在 **Mihomo/透明代理** 路由 `198.18.0.0/15`，metric 很低，容易抢默认路由。

### 2.2 本机实测（修复前后）

| Host | TCP:2000 | RPC handshake（Server 运行时） | 说明 |
|---|---|---|---|
| `127.0.0.1` | 通 | **成功** | 镜像网络下的可靠路径 |
| `192.168.5.1` 等 | 常拒绝 | 失败 | 非 CARLA 宿主机 |
| `198.18.0.2` | **假连通** | 失败 | 代理/TUN，不能当 Windows host |
| `172.30.80.1` | 可能通 | 视网络模式 | 历史 NAT 路径，非永久 |

另外：`get_client_version()` 是**本地 API 版本**，不能证明 Server 可达。  
`rpc_reachable` 必须在 **server RPC**（如 `get_server_version` / `get_world`）成功后才置真。

### 2.3 现行解析策略（`runtime/carla_connection.py`）

统一入口：

```text
python3 scripts/sdf.py sim status|preflight|ensure
```

候选顺序（摘要）：

1. 显式 `CARLA_HOST` / CLI  
2. WSL/Windows 下优先 **`127.0.0.1`（loopback）**  
3. 全部 default gateway 中**非代理**优先  
4. `198.18.0.0/15` 等代理风格殿后  
5. READY **仅**当 RPC handshake 成功  

`source safedrive_foundry/config/runtime/carla_environment.sh` 只是兼容导出环境变量，**不是**业务前置（见 `START_TASK.md`）。

### 2.4 路径约定

| 场景 | 路径 |
|---|---|
| Windows 可执行文件 | `E:\CARLA_0.9.16\CarlaUE4.exe` |
| WSL 访问同一安装 | `/mnt/e/CARLA_0.9.16/CarlaUE4.exe` |
| 官方 OpenDRIVE | `/mnt/e/CARLA_0.9.16/CarlaUE4/Content/Carla/Maps/OpenDrive/*.xodr` |
| ensure 配置 | `safedrive_foundry/config/runtime/carla_start.toml`（含 `windows_executable` + `wsl_path`） |

当前安装是 **Windows 构建**（`CarlaUE4.exe`）。Server 进程在 Windows；客户端在 WSL。  
`sdf sim ensure` 从 WSL 经 PowerShell 启动该 exe。用户接受「怎么方便怎么来」：不要求再装一份 Linux CARLA，除非以后有 `CarlaUE4.sh`。

---

## 3. G1-03：真图怎么验收，以及 CARLA Fatal Error

### 3.1 错误做法（已踩坑）

为导出三张地图而连续：

```text
client.load_world('Town01') → load_world('Town03') → …
```

在本机可导致 **超时 / Fatal Error**，用户关闭 Server。  
因此：**离线地图任务不要依赖连续切图导出**。

### 3.2 正确做法（已采用）

CARLA 0.9.16 安装包自带官方 OpenDRIVE：

```text
Content/Carla/Maps/OpenDrive/Town01.xodr
Content/Carla/Maps/OpenDrive/Town03.xodr
Content/Carla/Maps/OpenDrive/Town10HD.xodr
```

复制到工程：

```text
safedrive_foundry/classic_stack/map/fixtures/carla/
docs/architecture/evidence/g1-03/carla-opendrive/
```

并登记 SHA-256（`manifest.json`）。

交叉验证：曾对 Town01 成功调用 `to_opendrive()`，与官方文件 **同字节/同 hash**，说明官方 xodr 可作为真源。

### 3.3 两套地图各干什么

| 类型 | 路径 | 用途 |
|---|---|---|
| **验收真源** | `fixtures/carla/*.xodr` | 真实 CARLA 拓扑；`load_carla_map()` |
| **单元回归** | `fixtures/Town0*.xodr`（小合成图） | 固定节点 ID 的路线/行为单测；`load_map_fixture()` |

任务完成标准「至少三张 CARLA 地图」由 **官方 xodr** 满足；合成图不冒充真图。

### 3.4 G1-03 状态

- 已完成：`tasks/G1/G1-03_*.md` → `COMPLETED`  
- 指针：`PROGRESS.md` 推荐下一任务 **G1-04**（不自动开始）  
- 测试：`test_g1_02_connection` + `test_g1_03_map_route_behavior` 合计 20/20（完成时）

---

## 4. 文档修正原则

### 4.1 两类文档

| 类型 | 处理 |
|---|---|
| **现行说明**（README、G1_02 连接节、复测命令、排障） | **改成**动态 host / `sdf sim` / WSL 内权威环境 |
| **G0 冻结证据**（某日 doctor JSON、HOST 原始表、某次 echo 日志） | **保留原文**，顶部或旁注写「历史采样，非永久真理」 |

禁止：把 G0-01「WSL 未安装」或 G0-04「127.0.0.1 不可达」再抄进新任务断点当现状。

### 4.2 已改动的主要文件

| 文件 | 修正要点 |
|---|---|
| `safedrive_foundry/README.md` | 去掉写死 gateway；统一 `sdf sim`；路径双写 |
| `docs/runtime/G1_02_SCENARIO_RUNTIME.md` | 主入口改为 preflight，shell 仅为兼容 |
| `docs/environment/CARLA_ROS_CONNECTIVITY.md` | 现行连接表 + 历史 IP 标注 |
| `docs/environment/HOST_INVENTORY.md` | 顶部「状态覆盖」：当前已有 WSL/Jazzy/0.9.16 |
| `docs/environment/G0_05_DETERMINISM.md` | 复测命令不写死 host |
| `docs/environment/CARLA_SERVER_BASELINE.md` | WSL 路径、OpenDRIVE、ensure |
| `docs/environment/WSL_ROS2_BASELINE.md` | 工程根 `/mnt/e/autonomous driving` |
| `docs/project/TASK_CATALOG_AUDIT.md` | 审计当日 vs 现行进度分层 |
| `docs/project/CODEX_GROK_TROUBLESHOOTING.md` | IN_WSL 禁误判、代理路由、`load_world` fatal |
| `docs/architecture/G1_03_MAP_ROUTE_BEHAVIOR.md` | 真图来源与验收说明 |
| `config/runtime/carla_environment.sh` | 注释标明非主路径 |

### 4.3 未改动的

- G0 证据目录内具体 `doctor.json` / 某次 `172.30.80.1` 实测行（冻结历史）  
- G0 任务文件与冻结配置的无关回写  

---

## 5. 给后续代理 / 协作者的一页清单

1. **代码与测试**：默认在 WSL 仓库根执行。  
2. **真实 CARLA**：先 `python3 scripts/sdf.py sim preflight`，期望 `READY`。  
3. **Host**：读 preflight 的 `host` / `host_source`；禁止从旧文档抄 IP。  
4. **路径**：`E:\…` 与 `/mnt/e/…` 是同一棵树。  
5. **离线地图**：优先官方 OpenDRIVE；避免连续 `load_world` 仅为了导出。  
6. **状态**：只信 `PROGRESS.md` + 当前 `tasks/GX-XX_*.md`，不信过期审计段落。  
7. **WSL 判断失败**：先跑 §1.3 自检，再谈阻塞类型（`BLOCKED_EXTERNAL` vs 真无环境）。  

---

## 6. 相关代码与入口

| 组件 | 路径 |
|---|---|
| 连接解析 | `safedrive_foundry/runtime/carla_connection.py` |
| CLI | `scripts/sdf.py` → `safedrive_carla_bridge.cli` |
| Doctor（IN_WSL） | `…/doctor.py` |
| 启动规格 | `safedrive_foundry/config/runtime/carla_start.toml` |
| 真图加载 | `classic_stack.map.load_carla_map` |
| 合成图加载 | `classic_stack.map.load_map_fixture` |

---

## 7. 一句话总结

> **环境一直在（WSL2 + ROS Jazzy + CARLA 0.9.16）；错的是「写死网关 / 用错解释器 / 用历史盘点当现状 / 用切图导出当唯一真图来源」。现行约定是：WSL 内跑 `sdf sim`，动态 host，官方 xodr 做离线地图，G0 历史 IP 只当考古材料。**
