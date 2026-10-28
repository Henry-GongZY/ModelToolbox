# ModelToolbox — 本地模型结构差异分析工作台

本地优先的研发工具：像 Beyond Compare 一样对比 ONNX 模型计算图，保留 Netron 式算子级查看，并把逐层 profile 数据映射为热力图。**无遥测、无上传、离线可用。**

![mode](docs/screenshot-none.png) <!-- 占位：截图见 docs/acceptance_report.md -->

## 功能

- **单模型浏览**：ONNX 元信息、计算图（缩放/平移/适合视图）、节点详情（op/domain/属性/shape/dtype/参数/嵌套子图）、搜索定位。
- **双模型对比**：多阶段可解释匹配（锚点+拓扑传播，名称只是证据之一）、差异分类（新增/删除/算子/属性/shape/dtype/参数/连接变化）、匹配歧义显式标注（不强行配对）、差异摘要+导航+同步定位、仅看差异、公共子图折叠/展开、字段级变化详情。
- **热力图图层**：标准 JSON/CSV profile 导入，节点耗时/占比着色，独立图例与开关，与 Diff 图层可同时开启。
- **PyTorch 层级**：子进程隔离加载本地 `nn.Module` 层级，leaf 模块规则，FX 算子图捕获，失败可读降级。
- **导出**：diff → JSON/CSV/Markdown/HTML；profile → JSON/CSV/Markdown。

## 启动

```bash
# 1) 环境（Python ≥ 3.9）
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/pip install pytest onnx numpy        # 核心依赖
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu  # 可选：PyTorch 层级

# 2) 生成合成示例 fixtures（可复现，全部合成数据）
.venv/bin/python scripts/make_fixtures.py

# 3) 启动本地服务
.venv/bin/python -m modeltoolbox.server --port 8765

# 4) 浏览器打开
open http://127.0.0.1:8765
```

服务只绑定 `127.0.0.1`。模型/profile 通过**本地路径**加载（输入框填路径），不会离开本机。

## 运行测试

```bash
.venv/bin/python -m pytest tests/ -v      # 全量 92 项（核心引擎 + API 集成）
.venv/bin/python scripts/stress_test.py   # 1k/5k 压测（结果写 docs/stress_results.json）
```

## 快速上手路径示例

- 单模型：`fixtures/linear.onnx`
- 对比（算子替换）：A=`fixtures/linear.onnx`，B=`fixtures/op_replaced.onnx`
- 对比（全部改名，拓扑一致）：B=`fixtures/renamed_linear.onnx`
- 对比（匹配歧义）：B=`fixtures/ambiguous_diamond.onnx`（对称分支，显示 `?` 歧义标记）
- Profile：`fixtures/profile_linear.json` 或 `.csv`

## 目录结构

```
backend/modeltoolbox/   # Python 后端（IR/导入/匹配/diff/折叠/profile/布局/导出/服务）
frontend/               # 零构建 vanilla JS + SVG 前端
tests/                  # pytest（单元 + 集成 + API）
scripts/                # fixture 生成、压测
docs/                   # PRD/架构/路线图/进度/验收报告/profile 格式说明
fixtures/               # 合成 ONNX 与 profile 样例
```

## 安全边界

- PyTorch 源码 = 可执行代码：仅在**子进程**中加载（硬超时 + 进程组 kill），**不是沙箱**，只导入你信任的本地项目。
- 大型 initializer 只传元信息，权重数据不进入 IR/前端。
- 不上传任何文件；无网络依赖（除安装依赖时）。

## 已知限制

见 `docs/acceptance_report.md`（真实测试结果与未完成项）。
