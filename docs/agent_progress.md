# Agent 进度板

最后更新: 2026-10-28（全部阶段完成）

## 状态图例

TODO = 未开始 · DOING = 进行中 · DONE = 完成（有真实测试/验证）· BLOCKED = 受客观因素阻断

## 任务板（最终状态）

| # | 任务 | 优先级 | 状态 |
|---|---|---|---|
| A1 | 环境勘察 + venv + 依赖基线 | P0 | DONE |
| A2 | docs: PRD/architecture/roadmap/progress/log | P0 | DONE |
| B1 | `ir.py` Graph IR + 序列化 + 单测 | P0 | DONE |
| B2 | `onnx_importer.py` + shape 容错 + 坏文件诊断 + 单测 | P0 | DONE |
| B3 | 合成 fixtures 生成脚本（16 个文件） | P0 | DONE |
| B4 | HTTP API server + 会话管理 + 15 项 API 测试 | P0 | DONE |
| B5 | 前端: 单模型视图（图/详情/搜索/缩放/适合视图） | P0 | DONE |
| C1 | `matcher.py` 锚点+前沿传播 + 置信度/歧义 | P0 | DONE |
| C2 | `diff_engine.py` Diff IR + 10 类差异 + 往返序列化 | P0 | DONE |
| C3 | 8 类差异场景自动化测试 | P0 | DONE |
| D1 | `layout.py` 单图布局 + 双图对齐布局（伪节点跟随） | P0 | DONE |
| D2 | 前端: 双模型 diff UI（图例/导航/同步/仅看差异/详情/导出） | P0 | DONE |
| E1 | `folding.py` 公共子图折叠/展开（无丢失有测试保证） | P1 | DONE |
| E2 | 折叠 UI（摘要节点/勾选折叠/同步两侧） | P1 | DONE |
| E3 | `pytorch_importer.py` 子进程隔离 + leaf 规则 + FX 降级 | P1 | DONE |
| E4 | PyTorch UI（层级树/概览-详细切换/失败降级提示） | P1 | DONE |
| F1 | `profile_adapter.py` 标准 JSON/CSV + 单位/校验 | P1 | DONE |
| F2 | `profile_mapper.py` 5 级可靠度 + 4 类状态 | P1 | DONE |
| F3 | 热力图图层（单模型 + 双模型、与 diff 图层同开） | P1 | DONE |
| G1 | 1k/5k 合成 DAG 压测（docs/stress_results.json） | P2 | DONE |
| G2 | 导出 diff/profile JSON/CSV/MD/HTML + 回读测试 | P2 | DONE |
| G3 | 健壮性: 坏输入 422/会话删除/错误态/空态 | P2 | DONE |
| H1 | 10 场景端到端验收 + `acceptance_report.md` | P0 | DONE |
| H2 | README 启动说明 + profile 格式文档 | P0 | DONE |

**测试总计：92 项 pytest 全部通过；浏览器 E2E 场景 10/10 PASS（详见 acceptance_report.md）。**

## 阻塞项

（无）

## 假设记录（最终）

- 全部 fixture 为合成数据（文档已标注）；未验证真实厂商 profile 格式与大生产模型。
- UI 第一版为双模型并排；后端数据结构不写死两模型。
- 权重数值比对未实现（结构指纹级别），符合"结构相同数值不同→结构相同"的设计。
