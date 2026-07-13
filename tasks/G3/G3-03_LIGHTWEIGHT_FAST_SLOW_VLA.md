# G3-03：轻量 Fast/Slow VLA 与并行轨迹头

**状态**：PENDING
**依赖**：G3-02

## 启动读取清单

启动本任务前，除 START_TASK.md、PROGRESS.md、ROADMAP.md 对应阶段和本任务全文外，按顺序读取：

1. docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md 第 3～7、12 节；
2. docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md 第 1～5、7、8 节；
3. docs/project/CLAIMS.md C1 与统一指标；
4. G3-01 数据契约和 G3-02 基线接口、预算、checkpoint 与断点；

只读取列出的章节和直接依赖最终产物；引用文档继续列出必读项时，按其任务索引继续读取。

## 目标

在 RTX 4080 16GB 上实现单前视短视频+Ego+Route+可选语言的轻量 VLA：Fast 目标 10Hz 输出 4～8 条 3～5 秒轨迹；Slow 仅在 OOD/分歧/复杂交互时以 1～2Hz 目标触发且不阻塞控制。

## 实现范围与边界

- 冻结/轻量微调时序视觉编码器、小型语言模块、独立并行轨迹头；
- 输出轨迹概率/有效期/动力学元数据和结构化行为/关键 Actor/风险时域/意图；
- 大型模型只作离线教师，不自回归逐点生成轨迹；
- Slow 超时沿用新鲜安全轨迹或走冻结回退；

## 完成标准与验证

- 合成/真实 batch 前后向、小样本过拟合、保存恢复和 OOM 降级通过。
- CandidateSet/Validator 无损解析，非法轨迹拒绝。
- 报告参数、量化、显存、吞吐、时延分位和 trigger 成本。
- 关闭语言仍能运行 Fast；不直接输出控制或把自由文本当证据。

## 允许修改与交付物

本节所列实现组件路径均相对 safedrive_foundry/；tests/、docs/、tasks/ 与 PROGRESS.md 相对仓库根目录。

允许修改 `driving_vla/model、adapter、config、tests/g3、docs` 及本任务和 `PROGRESS.md`。交付实现/配置、对应测试、原始运行、可追溯报告和精确断点；涉及真实 CARLA 时先执行 `sdf sim preflight`。当前任务通过后停止，不自动开始下一任务。

## 断点记录

尚未开始。恢复时先核对直接依赖的最终接口、版本、冻结协议和外部状态。
