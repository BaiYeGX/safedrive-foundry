# 项目成功口径（SIL 作品向）

> **决策状态**：`ACTIVE`  
> **适用**：G3～G8 任务解释、验收与简历表述  
> **上位冲突**：用户本轮指令 > `START_TASK.md` §5 > 本文件 > 各任务正文 > 长期愿景  

## 1. 项目是什么 / 不是什么

| 是 | 不是 |
|---|---|
| 单机 CARLA–ROS 2 **纯软件在环（SIL）** 作品/研究平台 | 实车、公共道路、量产 L4 |
| 必须**真实接入**轻量 VLA + 动作条件 World Model | 只写文档或只跑 Classic 冒充学习闭环 |
| **稳定可跑、可演示、可复现证据** 为第一目标 | 必须全面超过 baseline / 论文 SOTA |
| 效果一般或负结果可接受，但必须诚实记录 | 伪造指标、假闭环、把 World 当安全真值 |

## 2. 发布完成的最小定义

以下同时满足即可宣称本项目主线完成（允许 `COMPLETED_WITH_LIMITS`）：

1. **VLA** 从 Observable 输入稳定产出 `PolicyCandidateSet`，经 G2 Safety → MPC/PID 可控。  
2. **World-V0** 对预筛后的 K2 候选做动作条件预测/软排序，**可开关**。  
3. 演示配置 **`VLA + World + Safety`** 在固定场景/seed 下连续跑通，不二次 tick、可恢复、显存可控。  
4. World 超时/OOM/NaN 时**确定性降级**为 `VLA+Safety`，控制环不阻塞。  
5. 有 Evidence：原始 run、配置 hash、限制清单；**正/负收益均可**。  
6. **不上实车**；CARLA 结果不得写成真实道路安全证明。

## 3. 指标与门禁怎么用

| 机制 | 用途 |
|---|---|
| F0（G3-03） | checkpoint 能否真加载、稳定前向；失败则停训练 |
| G4A oracle best-of-K | **科学标注**选择空间强弱（C2/C4），不是“不准做 World”的否决票 |
| G5 实现 | **本项目必做**（作品完整性）；弱选择空间时仍接入，标记 `WEAK_SELECTION_SPACE` / 负收益 |
| G6 一轮监督 | 推荐做；做不出正收益记负结论仍可进入 G8 |
| G4B / G7 | Optional，默认 `OPTIONAL_NOT_RUN` |
| CLAIMS C1～C5 | 有证据或诚实负结论即可，不要求全正 |

## 4. 与旧门禁文案的关系

旧文中 “G5 可 `SKIPPED_BY_GATE`、不阻塞 G6/G8” 仍适用于**严格科学发表路径**。  
**本仓库当前作品路径**覆盖为：G5 **必须实现并接入演示配置**；选择空间不足时：

- 仍训练/接入 World-V0；  
- C2 记负结论或 “无稳定净收益”；  
- 默认演示开关可为 World on + 一键 off 对照；  
- **禁止**因无增益而删除 World 模块或只留空接口。

## 5. 本机资产与 Python 环境

权威登记：`docs/project/LOCAL_ASSETS.md`。

- 路径：SimLingo 代码/权重、InternVL2-1B、CARLA 0.9.16  
- **G3+ VLA 必须**使用 `/home/sdf/.venvs/sdf`（PyTorch `2.12.1+cu126` + CUDA；含 carla）  
- **禁止**用系统 `/usr/bin/python3` 或 `/home/sdf/.venvs/carla_ros` 跑 SimLingo  
- 系统 Python 受 PEP 668 保护；勿 `--break-system-packages`  
- CARLA 未启动时 preflight 失败为预期；live 前再 `sdf sim preflight` 至 READY

## 6. 单任务执行提醒

- 一次只做一个 `GX-XX`。  
- 稳定/契约/降级 > 刷榜。  
- 未实测数字不得写成 VERIFIED。  
- live CARLA 前先 `sdf sim preflight`。  
