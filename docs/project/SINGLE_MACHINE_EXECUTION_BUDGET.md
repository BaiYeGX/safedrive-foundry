# 单机执行与资源预算规范

> 适用范围：G2～G8。本文中的显存、频率和时延均为准入目标，必须由任务实测确认；未测量的数据不得写成已达标结果。

## 1. 固定机器与共享资源事实

唯一目标机器：

~~~text
GPU: RTX 4080 Desktop 16GB
CPU: Intel i5-13600KF, 14 cores / 20 threads
CARLA Server: Windows
ROS 2 / Python / training / runtime: WSL2 Ubuntu
~~~

Windows CARLA 与 WSL CUDA 共享同一张物理 GPU。资源报告必须记录整卡占用，不能只记录 PyTorch allocated memory。没有服务器、第二张 GPU 或远程训练服务作为必需条件。

## 2. 工作负载互斥规则

| Profile | 允许的 GPU 主任务 | 必须关闭 |
|---|---|---|
| vla_train | VLA LoRA/QLoRA/SFT | CARLA、World 训练、Agent 本地模型 |
| world_train | 结构化 World Model 训练 | CARLA、VLA 训练、Agent 本地模型 |
| online_eval | CARLA Low/No Rendering + VLA + World | 所有训练优化器、Agent 本地模型 |
| data_collect | CARLA + 轻量采集/编码 | VLA/World 训练 |
| regression | CARLA + 单个发布配置 | 其他发布配置和训练 |
| agent_offline | 小型本地模型或受控 API | CARLA 高画质、训练任务 |

总控器必须拒绝冲突 Profile，而不是等待 OOM 后恢复。

## 3. GPU 准入目标

### 3.1 在线闭环

建议初始预算：

| 占用 | 目标上限 |
|---|---:|
| Windows CARLA Low/No Rendering | 实测后控制在约 4～5GB |
| 量化 VLA Fast/Slow | 约 5～6GB |
| 结构化 World Model | 约 2～3GB |
| CUDA context、缓存和安全余量 | 至少约 2GB |
| 整卡稳定峰值 | 不高于约 14～14.5GB |

这些是 admission target，不是硬件现状声明。如果 CARLA 实测超过预算，依次降低渲染质量、图像分辨率、历史帧数、VLA 候选数和 World batch；不能牺牲 Safety/Classic 进程。

### 3.2 VLA 训练

- 0.5B～2B 小型语言/推理模块；
- 冻结或部分冻结视觉编码器；
- LoRA/QLoRA、4-bit/8-bit 权重、BF16/FP16；
- gradient checkpointing 和梯度累积；
- 单前视 224～320 像素级输入起步；
- 4～8 帧短历史，micro-batch 1～2 起步；
- 训练显存稳定峰值目标不高于约 14.5GB；
- 先完成 20～100 step resource smoke，再批准正式训练。

### 3.3 World Model 训练

- 仅训练 object/vector/BEV latent dynamics，不做像素视频 diffusion；
- Actor 数、候选 K、horizon 和 mode 数配置化；
- 从 1～3 秒和小 K 起步，再扩展到 5 秒；
- mixed precision、gradient accumulation 和变长 mask；
- 训练显存稳定峰值目标不高于约 12～14GB；
- actor/occupancy/ranking 小样本过拟合通过后才正式训练。

## 4. CPU、线程与进程预算

必须独立的进程包括 CARLA client/runtime、ROS 2 executor、VLA/World inference worker、data writer 和 resource monitor。

WSL DataLoader/编码 worker 初始总数建议不超过 4～6，再通过实测增加。不得占满 20 线程导致 CARLA tick、ROS callback 或 20ms 控制周期抖动。性能报告必须同时给出 CPU、内存、磁盘写入和 deadline miss。

## 5. 在线流水与降级

~~~text
frame t: encode → VLA candidates
frame t: hard precheck
frame t: World batch scoring
frame t: final Safety
control: continuously track last fresh accepted trajectory
~~~

VLA 与 World 使用只读 feature cache；不得由两个模块分别复制完整视觉 backbone 到显存。先以串行调度确保稳定，再验证流水。

降级顺序：

1. 关闭 Slow VLA；
2. 降低 World candidate batch/mode/horizon；
3. 跳过 World，进入 VLA+Safety；
4. VLA stale/unavailable，进入 Classic+Safety；
5. 无合法轨迹，进入 Minimal Risk/Emergency。

Safety 状态监控和 MPC/PID 不等待上述 GPU 工作。

## 6. 数据与磁盘预算

- 图像、结构化 Actor、轨迹和事件分别分片，避免重复保存相同帧；
- 默认使用压缩 Parquet/DuckDB manifest，视频仅用于代表性证据；
- 每次采集前估算每小时数据量、剩余磁盘和 shard 数；
- 达到冻结 quota 后停止采集并报告，不自动删除旧数据；
- Regression、Evidence 和冻结数据只读；
- World action-branch 数据优先保存结构化未来，像素视频不是核心训练输入。

## 7. 阶段资源门禁

### G2

Safety、QP、RATO 和故障注入主要使用 CPU。必须证明学习模块/GPU 全失效时 Classic+Safety 仍运行。RATO P95/P99 超预算时默认关闭二级修复。

### G3

先做非语言多候选资源基线，再批准 VLA。Fast、Slow、候选 K、历史帧和分辨率分别消融。训练和 CARLA 不并发。

### G4

场景搜索串行 CARLA rollout 起步；只有状态隔离、磁盘和恢复可靠后才增加有限并发。单机不追求大规模并行吞吐。

### G5

World 必须批量评价候选而不是逐候选重复编码。Active CARLA 是异步研发验证，不参与当前帧控制。

### G6

一次只训练一个 adapter。后训练前后模型串行评测，避免同时加载多个 checkpoint。

### G7

Agent 默认离线；优先受控 API 或小型量化模型。无 Agent 时确定性 workflow 必须运行。

### G8

四发布配置串行加载和测试，不构建组合常驻进程。长稳测试只运行当前配置。

## 8. 每个任务必须报告的资源字段

~~~text
hardware_id
profile
CARLA quality/rendering mode
CPU utilization / P50 / P95 / P99
system RAM peak
Windows CARLA VRAM
WSL CUDA allocated/reserved
whole-GPU peak
model precision/quantization
latency P50/P95/P99
deadline miss
disk read/write and artifact size
OOM/thermal/disconnect/recovery
~~~

没有使用某项资源时写明“不适用”，不得省略到无法判断。

## 9. 任务启动阅读索引

- G2：读取第 1、2、4、5、7、8 节；
- G3：读取第 1～5、7、8 节；
- G4：读取第 1、2、4、6～8 节；
- G5：读取第 1～5、7、8 节；
- G6：读取第 1～4、6～8 节；
- G7：读取第 1、2、4、7、8 节；
- G8：读取全文并冻结实际资源结果。

