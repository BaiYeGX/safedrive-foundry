# G3-01：VLA/VLT 数据契约与采集

**状态**：PENDING  
**依赖**：G2-06

## 目标与范围

建立前视图像、Ego 历史、导航/路线、语言条件、Classic 专家候选、执行结果与监督标签的帧级数据契约、采集器和版本登记。明确 policy input、privileged label 和 evaluation-only 三类字段。

允许修改 `safedrive_foundry/data_pipeline/vla/**`、`safedrive_foundry/data_pipeline/collector/**`、`safedrive_foundry/data_pipeline/schema/**`、`safedrive_foundry/registry/**`、`safedrive_foundry/config/**`、`tests/g3/**` 和 `docs/architecture/**`；不训练模型。

## 完成标准与验证

- `run_id+carla_frame` 对齐图像、状态、路线、候选和结果；缺帧不静默插值。
- Parquet/DuckDB schema、压缩、分片、hash、重放和容量门禁可用。
- 固定 seed 小数据集重复生成摘要一致；中断不重复已完成分片。
- 数据卡明确 Oracle 来源、许可、分辨率、频率和失败比例。

## 交付物

- 本任务范围内的实现或配置、对应测试/场景、运行记录和可追溯报告；具体对象以本文件的范围和完成标准为准。

## 资源与自动化边界

- 仅使用本任务允许修改路径、已登记输入和版本化配置，不启动后续任务。涉及真实 CARLA 时先执行 `sdf sim preflight`；不得自行解析 WSL gateway、创建第二套 `carla.Client`/tick master 或由业务节点直接调用 `world.tick()`。Agent 不执行任意 shell，学习模块不得修改 Safety 硬约束，MCP 保留到 G7-01。
## 断点记录

最后状态：PENDING；中断记录数据版本、分片、场景、失败样本和恢复命令。
