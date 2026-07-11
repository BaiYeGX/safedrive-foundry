# G3-04：VLA闭环与不确定性基线
**状态**：PENDING  
**依赖**：G3-03、G2-04

## 目标
将VLA接入CARLA闭环，比较Classic、Raw VLA、VLA+Validator和VLA+Safety，并建立选择性驾驶基线。

## 范围
完成5–10Hz轨迹chunk、MPC跟踪、Shadow对照、轨迹分歧、embedding OOD、温度校准和grounding检查。

## 完成标准
- 正式推理无特权信息。
- 闭环失败、超时、接管和不确定性完整记录。
- Risk–Coverage、OOD Recall、Selective Success、误/漏接管可计算。
- G3门禁与Evidence Bundle完成。

## 验证方法
ID与未见Town/天气/参数场景闭环，多seed比较open-loop与closed-loop差异。

## 断点记录
最后状态：PENDING；恢复时读取策略版本、最近run和失败切片。

