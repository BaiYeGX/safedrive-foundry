# G5-05：VLA+World+Safety 闭环验收（作品主展示）

**状态**：PENDING  
**依赖**：G5-01～G5-04  
**阶段角色**：必做（G5 关闭）  
**一句话**：固定 VLA/Safety/场景/seed，做 World 开/关 A/B；**模块必须在，增益可负**。

## 启动读取清单

1. `docs/project/PROJECT_SUCCESS_PROFILE.md`；  
2. `docs/project/CLAIMS.md` **C2**（主）；  
3. `docs/project/SDF_WORLD_MODEL_DESIGN.md` A/B 与负结果；  
4. `docs/project/SDF_VLA_WORLD_SYSTEM_ARCHITECTURE.md`；  
5. `docs/project/G2_G8_INDUSTRIAL_ARCHITECTURE.md` 第 1～7、9～14 节；  
6. `docs/project/SINGLE_MACHINE_EXECUTION_BUDGET.md` 第 1、2、4、5、7、8 节；  
7. G5-01～G5-04 产物、模型卡、降级矩阵。

## 项目成功口径（本任务）

**本阶段最小成功（作品）：**

1. 配置 `VLA+World+Safety` 稳定跑通固定演示场景；  
2. 配置 `VLA+Safety` 同协议对照可跑；  
3. 降级路径验证通过；  
4. C2 写清：正收益 / 无稳定净收益 / 负收益（皆可）。

**禁止：** 无 World 二进制或仅空接口却宣称“已集成 World”。

## 目标

- 预测质量、ranking/regret、闭环安全与舒适、资源一并报告；  
- 模型卡绑定默认开关与证据状态。

## 实现范围与边界

### 必做

- 冻结数据/模型/场景/seed；  
- 常规 + 至少一类长尾或故障切片；  
- 扩大 N/M/K 仅在有稳定净收益后允许。

### 明确不做

- 不因负收益删除 World 代码路径；  
- 不自动开始 G6。

## 完成标准与验证

### 最小通过

- 演示脚本或清单：一键/有限步骤启动 VLA+World+Safety；  
- A/B 表与原始 run 链接；  
- Evidence 自检。

### 建议验证命令

```text
sdf sim preflight
python3 -m unittest discover -s tests/g5 -t . -v
# 演示/A-B 脚本（实现后写入断点）
```

## 允许修改

G5 小型缺陷、`validation/g5`、`registry`、`artifacts/g5`、`tests/g5`、`reports`、本任务、`PROGRESS.md`。

## 断点记录

尚未开始。
