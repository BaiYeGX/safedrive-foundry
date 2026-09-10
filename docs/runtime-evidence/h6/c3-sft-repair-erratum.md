# C3 SFT 勘误与修复记录

## 状态

`c3-vla-sft-20260909-final-v3`、更早的 `c3-repair-20260910T014948Z`、
`c3-repair-20260910T091736Z` 和 `c3-repair-20260910T093810Z` 都是历史运行。它们保留在
本机 `generated/h6/cora/` 供追溯，但不再属于 C3 的活动验收依据。旧报告中的
“90.77% 驾驶能力提升”、旧 manifest、旧 `VERIFIED` 状态和由它们导出的比较均已撤回。

修复版权威运行是：

```text
generated/h6/cora/c3-repair-20260910T100651Z/
run_id = c3-repair-20260910T100651Z
verify.status = VERIFIED
verify.content_sha256 = 29f9c56e4bfff8780c1a6a3f902872f0f81b4b56e271bdaa4ff5d521341f8af3
manifest_sha256 = 54281a1de6207b4e0c553ee8456de40f43131a6ef30b434e7bd1064da776c2fd
model_sha256 = ec8943723d266ee9f5f56f45d153a163b22616960bfccb741965ea5daa700d28
```

## 无效原因

初版适配没有把导航/参考路线先按当前位置投影到有序路线并裁去已经经过的前缀，导致
部分监督坐标可以落在车辆后方。初版把这个结果当成完整原生专家路线，因而把一个不符合
部署时序语义的 target 带进了 route loss 和主指标。这个问题不是训练波动，不能用重复训练
或扩大容差修正，所以旧结果不再可比。

## 修复内容

- 从冻结 C2 dev release 复用 train 158 / validation 53 身份和物理去重，不重跑 release、
  不重扫旧 Hash。每个样本独立记录当前输入、专家监督、离线后果和审计元数据。
- 对有序 native expert reference path 做当前位置投影，清理连续重复点，裁去已通过前缀，
  保留合法转弯、自交和负 map-x 坐标；投影歧义或距离超过 2.5 m 时拒绝，不以负 x 作为
  删除规则。route 只在真实支持区间按 1 m 采样 20 点（0…19 m），不外推、不拼接导航。
- 将导航折线限定为部署输入/诊断来源，不把导航自动当成专家 label。route label 绑定
  `native_expert_reference_path_projected`、`classic-frenet-st@h1` 和 route revision；speed
  label 独立绑定 canonical expert proposal 的 10 点、0.25 s 合同。
- 对碰撞、控制回退、越界和提前终止保留失败事实；不能证明正确的区间只作 mask。3 个
  collision terminal train root 的 speed 监督全部无效，route 仍来自独立且审计过的 native
  reference path，不能混成一个“全样本失败”标签。
- 当前速度只读 observable snapshot；211 个 anchor 零速度及其与 history 的差异均保留。
  采集 timeline 的 0.05 s dt 和 branch preroll 不能证明与 anchor 对齐，因此 P-SPEED 为
  `N/A`，没有用执行未来速度替换输入。
- 训练只调用原生可微 forward/head，显式白名单更新 LoRA 与 route/speed head，冻结 base、
  vision projection 和 query embeddings。checkpoint 原子保存 optimizer、RNG、游标、root
  暴露和日志边界；158 roots 每轮为 39 个四样本窗口加一个两样本尾窗口。
- `verify` 同时检查代码/配置/manifest/model identity、逐 root 完整性、逐头 mask、独立
  指标重算、训练摘要与 JSONL 日志一致性、checkpoint/adapter 指纹、资源账本和历史失败。
  预测、重复 root、manifest、metrics、checkpoint identity、日志删尾、资源字段篡改都会
  失败；缺失的 failure sidecar 在 CLI 层直接拒绝。

## 修复版实测边界

修复版 M1 从原始 M0 开始，以 seed 17 完成 200/200 updates、790 个样本暴露；LoRA 和驾驶
head 均发生有限变化，冻结参数指纹不变。53 个 validation root 的 M0/M1 结果为：route
P-ADE `0.210514 → 0.232211 m`（下降 `-0.021698 m`，相对 `-10.3069%`，95% CI
`[-0.159394, 0.103600]`）、route P-FDE `0.735295 → 0.806690 m`（`-0.071395 m`，
`[-0.625551, 0.471091]`）、speed P-WP `2.178557 → 0.336781 m`（`+1.841776 m`，
`84.5411%`，`[1.570158, 2.128422]`）。P-VALID 为 53/53，P-FAIL 为 0/53，P-SPEED 为
N/A。这个单 seed 开发结果显示 route 分支退化、speed waypoint 分支改善，不能包装成
全面收益；工程验收独立于正收益。

修复版的 200 更新优化及历史尝试的保守总账上界为 `8.622364703124443 h`，观测整卡
allocated/reserved 峰值为 `9.889132/10.845703 GiB`，低于 10 h 优化和 14.5 GiB 峰值约束。
C4 尚未执行；C4 的 M2 必须从原始 M0 起点开始，不能从 M1 续训。
