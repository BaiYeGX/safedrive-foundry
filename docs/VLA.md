# VLA-V1 设计与当前修复合同

本文只保留当前 VLA 实施所需的模型、接口、训练和验收规则。项目目标与完成定义见
`PROJECT.md`，当前唯一任务见根目录 `START_TASK.md`。

## 1. 已有基线

当前真实能力：

```text
CARLA RGB + ego + coarse navigation
  → SimLingo / InternVL2-1B real CUDA forward
  → pred_route + pred_speed_wps
  → PathManager
  → constrained MPC
  → CARLA
```

该 K1 pure VLA 基线状态为 `MEASURED_WITH_LIMITS`。它证明模型真实看图、产生路径与
速度并由 CARLA 执行，但不证明 K2、World 增益或道路安全。

冻结的上游契约：

- camera：1024×512、FOV 110°、位姿 `(-1.5, 0, 2.0)`；
- BGRA → BGR → OpenCV JPEG → RGB → 官方底部裁剪；
- coarse targets：上游 RoutePlanner 7.5m 语义；
- path 与 speed head 分离；
- 地图中心线不作为 MPC 跟踪参考。

## 2. 当前缺口

旧 V1 residual 接口不能作为真实 K2：

1. 两条候选空间位置重合；
2. 只改速度，没有重新积分/参数化位置；
3. `x/y/yaw/v/a/kappa` 不完全一致；
4. 位置型 oracle 无法区分两候选；
5. candidate 0/1 尚无正式强制执行证据。

因此当前状态是 `G3-04 / CURRENT / REPAIR_REQUIRED`。

## 3. 最短 K2 实现

第一版允许：

- `k0`：上游 nominal；
- `k1`：conservative 时间分支；
- 相同空间几何可以作为起步，但必须沿路径按新速度重新时间参数化，使每个相同时刻
  的 `x/y/v/a` 真正不同且一致；
- 若 G4A pilot 证明仅纵向差异没有选择空间，再训练空间 residual 或双轨迹头。

禁止：

- 复制轨迹只改 candidate_id/probability；
- 只改标量 speed、不改变 T10 位置；
- 随机噪声伪装模型候选；
- 为每条候选重复运行视觉主干；
- 为追求 K2 改用 3B/7B 模型。

## 4. VLA-V1 结构

输入：

- 当前前视图像；
- 当前 ego；
- 至多 4 个低维 ego history 时刻；
- route/navigation；
- Observable identity/freshness。

建议结构：

```text
frozen SimLingo visual/language backbone
  + small ego/history adapter
  + anchored K2 trajectory head
  + probability/margin head
```

优先 heads-only；确有必要时少量 LoRA。VLA-V2 FAST/REASON、复杂 grounding、
ensemble OOD、K4 和 3 秒时域均为 optional。

## 5. 数据与训练

- 训练/验证/回归 split 冻结，Regression 不得进入训练；
- 正式 runtime 输入禁止 CARLA future、真实 TTC 和隐藏意图；
- 记录 base revision、checkpoint hash、data/config/seed、precision 和许可；
- 先做小样本过拟合与 20–100 step resource smoke；
- 一次只训练一个配置；训练和 CARLA 不并发；
- NaN/OOM/中断必须可恢复并登记。

最小损失：

```text
L = L_traj + L_speed + L_dyn + L_diversity + L_probability
```

`L_dyn` 保证运动学一致，`L_diversity` 防止候选坍塌，但二者不能替代 Contract Guard。

## 6. 当前完成标准

G3-04R 通过必须同时满足：

1. `K=2/T=10/dt=0.25s/horizon=2.5s`；
2. 固定输入可复现；
3. candidate 0/1 通过预登记差异度；
4. `x/y/yaw/v/a/kappa` 一致；
5. candidate 0/1 可分别进入同一 PathManager/MPC；
6. collapse、margin、概率和谱系可查询；
7. K1 baseline 不回归；
8. 小训练/适配 smoke 无 OOM；
9. 对应单元测试通过。

建议验证：

```bash
source /home/sdf/.venvs/sdf/bin/activate
python -m unittest discover -s tests/g3 -t . -v
```

需要 CARLA 时必须先：

```bash
python scripts/sdf.py sim preflight --json
```

## 7. 故障与降级

- K2 schema/finite/time/kinematics/diversity 不通过：拒绝进入 G4；
- VLA timeout/unavailable：核心 run 停车并记失败，不伪造候选；
- World timeout/unavailable：执行 VLA 原始 top-1；
- Safety 扩展启用时，复用同一 CandidateSet，不改变核心 K2。

## 8. 模型谱系

研究主路径：

```text
SimLingo code: simlingo-main/
checkpoint: models/simlingo/
runtime venv: /home/sdf/.venvs/sdf
```

InternVL2-1B 只作为主路径真实不可修复时的干净后备；切换后必须重做加载、资源、
K1/K2 和闭环证据，不能继承 SimLingo 结果。
