# SafeDrive Foundry runtime

已有 Runtime、独立 Classic/VLA、Guard、Safety、MPC/PID 与 collector 保留。
本周只做 C3 常规微调、C4 四维后果辅助联合微调、C5 三臂开发闭环、C6 交付。
C3 已完成并验证；C4 尚未启动。C3 的实际入口是 [`scripts/h6_cora_sft.py`](../scripts/h6_cora_sft.py)，
权威运行摘要位于 `generated/h6/cora/c3-vla-sft-20260909-final-v3/`。

## 模块与入口

| 模块 | 职责 |
|---|---|
| driving_vla/ | VLA forward、候选合同与 adapter |
| classic_stack/ | Classic planning/control |
| data_pipeline/ | 已有数据、World、评估，优先复用 |
| safety_kernel/ | 硬验证、repair、MRM/fallback |
| runtime/ | connection、registry、唯一 tick owner |
| ros_ws/ | ROS 2 tick/status bridge |
| config/ | 机器可读路径与运行配置 |

[START_TASK](../START_TASK.md) 是唯一执行任务，[ROADMAP](../ROADMAP.md) 是完整阶段，
[WORLD_MODEL](../docs/WORLD_MODEL.md) 与 [COUNTERFACTUAL_DATA](../docs/COUNTERFACTUAL_DATA.md)
规定模型/监督；[ENVIRONMENT](../docs/ENVIRONMENT.md) 与 [RESOURCES](../docs/RESOURCES.md)
规定运行预算，[EVIDENCE](../docs/EVIDENCE.md) 保存实际结果。

## 不变量

Observable → 独立 Expert + nominal VLA → per-candidate Guard → World rank/defer →
Safety → MPC/PID。World 仅看 PASS/REVIEW，无生成候选、底盘或 tick 权限。
h 来自当前信息、在候选特有输入之前提取，World-only 梯度更新 LoRA。
真实未来和 source 元数据不进 feature；轨迹风格捷径单独诊断。
C5 不做正式校准，selector 标 UNCALIBRATED；旧 readiness 不自动证明新模型可用。

正式 tick 只有 ScenarioRuntime；ROS bridge bring-up 互斥。需要真实 CARLA 时：

```bash
python scripts/sdf.py sim preflight --json
```

只有 READY 才继续；不硬编码 host，不新建 tick master。C3 训练命令已经实现并实际测通，
新行为有直接测试，模型实测和单元测试证据分开；当前下一入口是 C4。

当前 C4 的 World 为冻结候选 b + 共享 h 的残差 R；训练按驾驶梯度相容门控，
在线仍只有原两条候选。b 系数、R、LoRA 与归一化必须一起绑定；新配方尚未实现。
