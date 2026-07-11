# G3-01：VLA数据管线与特权审计
**状态**：PENDING  
**依赖**：G2-04

## 目标
建立Vision–Language–Action/Trajectory数据采集、质量、版本、划分和特权信息审计。

## 范围
采集前视图像、Ego历史、导航、专家轨迹、行为、风险Actor、规则和执行结果；按Town/Route/场景/天气隔离。

## 完成标准
- CARLA隐藏未来、真实TTC、碰撞标签和测试专家轨迹不进入正式Policy输入。
- 语言事实由真值模板生成并校验，禁止幻觉标签。
- Parquet/DuckDB/Registry schema稳定。
- ID/OOD/Regression划分可追溯且无泄漏。

## 验证方法
运行数据审计、重复/缺失/时间对齐检查和故意泄漏测试。

## 断点记录
最后状态：PENDING；恢复时读取数据版本、审计报告和未处理分片。

