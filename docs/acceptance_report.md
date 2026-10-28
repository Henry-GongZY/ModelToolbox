# 验收报告 — ModelToolbox

日期：2026-10-28 · 验收人：自主 Agent（总控提示词阶段H）· 全部结论基于本机真实执行

环境：macOS (arm64, Darwin 27) / Python 3.9.6 / onnx 1.19.1 / numpy 2.0.2 / torch 2.8.0 (CPU) / Node 25（仅浏览器自动化用）
测试命令：`.venv/bin/python -m pytest tests/ -v` → **92 passed, 0 failed**（单元 + 集成 + HTTP API）

---

## 1. 端到端场景验收（10 项）

| # | 场景 | 结果 | 证据 |
|---|---|---|---|
| 1 | 单 ONNX 加载、展示、节点属性 | **PASS** | 浏览器实测 `fixtures/linear.onnx`：3 节点/6 边，元信息（opset/指纹/参数量）渲染；点击节点显示 op/attrs/shape/dtype/params（截图记录于执行日志） |
| 2 | 两个相同 ONNX 对比 | **PASS** | UI：linear vs linear 全部 unchanged + 出现 3 节点折叠组；API 测试 `test_fold_roundtrip` |
| 3 | 算子/属性/shape/dtype/连接变化 diff | **PASS** | UI：op_replaced 红框 M 徽标 + 字段级 `op_type: Relu→LeakyRelu`；pytest `test_attr_shape_dtype_param_changes_isolated`、`test_connection_changed` 覆盖每类且互不误报 |
| 4 | 节点名全变、拓扑不变 | **PASS** | UI + `test_end_to_end_real_fixtures`：renamed_linear 全 unchanged，拓扑指纹一致 |
| 5 | 公共子图折叠/展开 | **PASS** | UI 勾选折叠 → "Conv×2 + Relu×1" 摘要节点，取消恢复；`test_apply_folding_all_states_no_loss` 断言任意折叠状态节点/边无丢失无重复 |
| 6 | PyTorch 层级概览/详细切换 | **PASS** | UI 实测 `/tmp/pt_demo`（嵌套+重复模块+自定义 leaf 类）：10 模块、FX 捕获、leaf 标记、树图切换；trace 失败降级路径由 `test_fx_failure_degrades` 覆盖（degraded=True + 可读诊断） |
| 7 | 标准 JSON/CSV profile + 耗时/占比/热力图 | **PASS** | UI 导入 `profile_linear.json` → 3 映射/0 未映射，热力图按耗时着色（relu 最浅、conv_2 最深）；CSV 等价性有单测；单位换算/占比/calls/坏输入 18 项单测 |
| 8 | diff 与热力图层同时开启 | **PASS** | UI：红框 M 徽标（diff 通道）+ 蓝色填充（热力通道）同节点叠加可区分，图层独立开关 |
| 9 | 坏 ONNX / 坏 profile / 无法 trace 的模型恢复 | **PASS** | UI + API：bad_not_onnx.bin → 422 "not a parseable ONNX model (protobuf decode failed…)"；二进制 profile → 422 UTF-8 报错；动态控制流模块 → 模块树 + FX 诊断降级；应用不崩溃 |
| 10 | 1k/5k 节点压测 | **PASS** | 见 §3；数据文件 `docs/stress_results.json` |

## 2. 核心功能状态

| 功能 | 状态 | 说明 |
|---|---|---|
| ONNX 浏览 | **PASS** | 元信息/图/详情/搜索/缩放/平移/适合视图/嵌套子图标记 |
| 模型对比（双模型并排） | **PASS** | 内部数据结构按模型列表设计（`STORE.models` 字典），UI 当前双模型 |
| 匹配与歧义 | **PASS** | 3 级锚点 + 邻域传播 + tie 推迟 + 对称组歧义标注；8 类场景测试 |
| 子图折叠 | **PASS** | 仅高置信度（默认 ≥0.9）匹配的连通分量；无丢失有测试保证；UI 同步折叠两侧 |
| PyTorch 层级 | **PASS** | 子进程隔离 + 超时 kill + leaf 规则 + FX 尽力捕获 + 降级诊断 |
| profile 热力图 | **PASS** | JSON/CSV 适配、5 级映射可靠度、4 类记录状态、双图层同开 |
| 大图性能 | **PASS** | 5k：全管线 10s 内、交互 320ms、视口裁剪 |
| 报告导出 | **PASS** | diff JSON/CSV/MD/HTML、profile JSON/CSV/MD；回读测试（JSON 解析、CSV 表头断言） |

## 3. 压测数据（真实测量，本机）

构建方式：链式 DAG + 每 25 步分支汇合（`scripts/stress_test.py`，可复现）；B 侧为全部改名版本。

| 阶段 | 1k 节点 | 5k 节点 |
|---|---|---|
| ONNX 构建 | 0.015s | 0.026s |
| 导入 GraphIR | 0.082s | 1.264s |
| JSON 序列化 | 0.052s (0.86MB) | 0.30s (4.4MB) |
| 匹配（全部改名） | 0.065s（1078 全配对） | 1.327s（5398 全配对） |
| Diff 构建 | 0.034s | 0.214s |
| 单图布局 | 0.098s | 2.086s |
| 对齐双图布局 | 0.187s | 4.546s |
| **HTTP /api/diff 端到端** | **0.646s (2.0MB)** | **10.146s (10.3MB)** |
| 浏览器加载（UI 实测） | — | 8.0s（后端 4.1s + 前端解析） |
| 浏览器缩放交互 | — | ~320ms |
| 视口裁剪后实际绘制 | — | 141 节点 / 271 边（全长 5398/10399） |

压测过程中发现并修复的真实问题（详见执行日志）：压测脚本 rename 断链 bug、matcher O(N²) 传播与歧义过早判定、layout O(N²) 排序。

## 4. 测试清单汇总

- 单元/集成：92 项全部通过（`tests/`：importer 12、matcher 10、diff 8、layout 8、profile 18、folding 10、pytorch 10、server 15、helpers/conftest）
- 覆盖规定的 8 类匹配场景（相同/改名/增删/属性/连接/对称歧义/权重不同/不可判定）
- UI 验收：浏览器自动化真实操作（非截图摆拍），关键交互均有断言或截图
- 压测：真实计时，环境与构造方法已记录

**NOT RUN / INCONCLUSIVE 项**：
- 真实厂商 profile 格式（如某板端工具导出格式）：**NOT RUN** — 仅实现并验证文档化标准格式（见 `docs/profile_format.md`）
- 真实大型生产模型（>100MB ONNX）：**NOT RUN** — 全部 fixture 为合成数据（已在文档标注）；大 initializer 元信息路径有单测但未用真实大模型验证
- 多模型（>2）UI：**NOT IMPLEMENTED** — 后端数据结构不写死两模型（dict 存储、按 model_id 索引），UI 仅双模型并排（符合 PRD 第一版边界）
- 权重数值逐值比对模式：**NOT IMPLEMENTED**（设计上仅指纹级别比对；结构相同数值不同的模型正确判为结构相同，有测试）

## 5. 已知限制（按影响排序）

1. **已验证**：匹配器对"无锚点且互相对称"的大块区域输出歧义而非强行配对——正确但用户需自行判断（UI 有候选列表与说明）。
2. **已验证**：5k HTTP 响应 10.3MB，前端解析约 4s——可用，未来可增量加载。
3. **未验证风险**：真实训练框架导出的复杂控制流 ONNX（Loop/Scan 嵌套较深时）UI 只显示子图存在性标记，不展开交互浏览（子图数据已解析保留）。
4. **未验证风险**：PyTorch FX trace 对动态控制流天然受限（已降级处理并明示）。
5. 双模型视图的同步缩放以"最后操作侧"为准，极端快速操作下可能有 1 帧差异（未量化）。

## 6. 安全边界（已实现）

- PyTorch 源码仅在子进程加载（`start_new_session` + 硬超时 kill 进程组 + stderr 尾部捕获）；UI 明示"子进程隔离不是沙箱"。
- initializer 只序列化元信息，权重不进入 IR/前端/导出。
- 服务绑定 127.0.0.1；无遥测；模型与 profile 不外发。
- 报告/导出不含密钥与环境变量。

## 7. 如何增加新 profile adapter

实现 `backend/modeltoolbox/profile_adapter.py` 中的同款接口：`load_xxx(path) -> ProfileData`（记录字段见 `docs/profile_format.md`），在 `server._profile_load` 挂接格式嗅探；映射逻辑无需改动（`profile_mapper.map_profile` 只依赖 ProfileData）。

## 8. 启动与测试命令（供另一位开发者）

```bash
python3 -m venv .venv && .venv/bin/pip install -e . pytest onnx numpy
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # 可选
.venv/bin/python scripts/make_fixtures.py
.venv/bin/python -m modeltoolbox.server --port 8765     # → http://127.0.0.1:8765
.venv/bin/python -m pytest tests/ -v                    # 92 项
.venv/bin/python scripts/stress_test.py                 # 压测
```
