# 架构 — ModelToolbox

日期: 2026-10-28 · 状态: 活跃

## 1. 总体形态

本地单机工具，两部分：

- **Python 后端**（`backend/modeltoolbox/`，包名 `modeltoolbox`）：ONNX/PyTorch 解析、Graph IR、匹配、Diff、折叠、profile 适配/映射、布局、导出、HTTP API。核心逻辑**不依赖前端**，全部可单元测试。
- **前端**（`frontend/`，零构建 vanilla JS ES modules + SVG）：图交互、diff/热力图图层、详情面板。

**运行方式**: `python -m modeltoolbox.server --port 8765`（stdlib `http.server` + ThreadingHTTPServer，无第三方 web 依赖），浏览器打开 `http://localhost:8765`。

## 2. 关键技术决策（含理由）

| 决策 | 选择 | 理由 |
|---|---|---|
| ONNX 解析 | `onnx` 官方包 (1.19) | 标准、可靠；手写 protobuf 解析作为降级路径不必要（环境有网络）。 |
| 后端框架 | stdlib `http.server`，无 FastAPI | 减少依赖面；API 简单（JSON in/out），stdlib 可测试性足够。 |
| 前端框架 | vanilla JS ES modules，无构建步骤 | 空仓库、追求"可启动、可测试"的可靠性上限；React/Vite 收益主要是 DX，代价是构建链风险。类型用 JSDoc 注释约束关键数据形状。 |
| 渲染 | SVG + 视口裁剪（culling） | 节点需 DOM 级 hover/点击/样式切换；配合"只渲染可见节点"满足 5k 规模（阶段G实测验证，若不达标再换 Canvas/WebGL）。 |
| 布局 | 自研分层布局（Sugiyama-lite：拓扑分层 → barycenter 排序 → 坐标分配） | 双模型对齐布局需要与 match ID 深度耦合，通用图库反而难定制；规模可测。 |
| PyTorch 导入 | 子进程 worker（`pytorch_worker.py`）+ JSON IPC | Python 源码属可执行代码：隔离、超时、异常完整捕获（安全模型见 §5）。 |
| 存储/会话 | 后端内存 session + 惰性重解析 | 本地工具，无持久化需求；模型指纹记录在 IR 供 profile 关联。 |

## 3. 核心模块（强制分离）

```
backend/modeltoolbox/
  ir.py               # Graph IR: dataclass + to_json/from_json，schema_version
  onnx_importer.py    # ONNX ModelProto -> GraphIR（shape inference 部分失败容忍）
  pytorch_worker.py   # 子进程内运行: torch 脚本解析 nn.Module 层级 / FX 图
  pytorch_importer.py # 父进程侧: 子进程管理、超时、诊断、降级
  matcher.py          # 多阶段锚点+传播匹配，置信度/歧义标注（纯算法）
  diff_engine.py      # Diff IR 构造: 差异分类、字段级变化、解释文本（纯算法）
  folding.py          # 公共连通区域折叠/展开，group_id 管理
  profile_adapter.py  # 标准 JSON/CSV -> ProfileRecord 列表（单位/校验）
  profile_mapper.py   # ProfileRecord -> 节点映射（可靠度分级），状态追踪
  layout.py           # 单图分层布局 + 双图对齐布局（基于 match_id）
  export.py           # diff/profile 导出 JSON/CSV/HTML/MD
  server.py           # HTTP API + 内存 session
```

数据流：`文件 → Importer → GraphIR(JSON) → [Matcher → DiffIR] → Layout → 前端渲染`；
`profile 文件 → Adapter → ProfileRecord[] → Mapper → 节点性能标注 → 热力图图层`。

## 4. Graph IR 约定（schema_version = 1）

- 顶层：`schema_version, model_id, name, producer, ir_version, opset, metadata, inputs, outputs, nodes, edges, value_info, fingerprints, warnings`。
- ID 约束（严格区分）：
  - `model_id`：加载实例身份（会话内唯一）。
  - `node_id`：模型内唯一（ONNX 节点名缺失时合成 `n{index}`）。
  - `edge_id`：`e{src}->{dst}->{tensor}` 形式。
  - `match_id`：跨模型对应关系，仅由 Matcher 生成。
  - `group_id`：折叠组，仅由 Folding 层生成（多模型公共组统一）。
- 形状未知 → `null`（数组内维度 unknown 也是 `null`），不猜测。
- initializer 默认只带元信息（name/dtype/shape/字节数），不序列化权重值。
- ONNX `If/Loop` 嵌套子图：解析为节点上的 `subgraphs` 字段（属性摘要 + 子图节点列表），不静默丢弃；UI 显示存在性标记。

## 5. 匹配与 Diff

匹配分阶段（每阶段产出证据记录 `evidence[]` 与 confidence ∈ [0,1]）：

1. **锚点**：图输入/输出（唯一）、唯一 op type 签名（op+domain+规范化属性+参数形状）、可靠唯一节点名、唯一 tensor 名。
2. **传播**：从锚点沿拓扑邻域扩散（前驱/后继一致 + op/shape 兼容打分），迭代到不动点。
3. **歧义处理**：多候选且分数并列 → 不强行分配，标记 `ambiguous`，candidates 列表入 diff 记录；UI 显式呈现。

Diff 分类：`unchanged / added / removed / op_changed / attr_changed / shape_changed / dtype_changed / param_changed / connection_changed / ambiguous`。DiffIR 版本化（`diff_schema_version=1`），含字段级 before/after、match 依据、边级增删记录、人读说明。

结构相同但权重数值不同 → 默认结构相同（权重比对仅显式独立模式，本版仅做 fingerprint 提示不做逐值比对）。

## 6. 安全边界

- PyTorch 源码 = 可执行代码。仅在**子进程**中 import/trace，`timeout` 硬超时 + 进程 kill，异常捕获回传；worker 不开网络。子进程隔离**不是**沙箱，UI 明示"仅导入你信任的本地代码"。
- profile / 模型文件不外发；无任何遥测。
- 密钥/私有路径不写入导出报告（对路径做 basename 化可选）。

## 7. Profile 语义

- 内部统一微秒（us）浮点，适配器记录原始单位；占比默认 = 该记录 duration / **已映射且参与统计的记录总和**；若 profile 提供端到端总耗时，另行展示并标注口径差异。
- 映射可靠度：fingerprint 匹配 > node id/index > 精确节点名 > tensor 名 > op_type+局部拓扑（最后一级必须局部拓扑佐证，禁止纯 op type 顺序分配）。
- 记录状态：`mapped / unmapped / ambiguous / group_level / kernel_fused`；kernel 融合记录在 UI 标注"覆盖多节点"，绝不伪装成单节点精确耗时。

## 8. 前端图层模型

- 基础层：节点/边（灰 `#8a8f98` 系）。
- Diff 图层（开关）：红系边框 + 角标图标 + 类型缩写（A/D/M/±）+ 图例；不依赖颜色单独传达。
- 热力图图层（开关）：节点填充色 = 顺序色带（viridis 风格 5 档），独立图例（耗时/占比两种模式）+ 热点列表。两层同开：边框红系 + 填充热力，互不混用通道。
- 布局：双图各自分层但按 match_id 行对齐；折叠组渲染为摘要节点，展开恢复原节点/边。
