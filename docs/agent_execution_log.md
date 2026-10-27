# Agent 执行日志

真实命令与观察，按时间追加。

## 2026-10-28

- 工作区勘察：`/Users/shinano/Opensource/ModelToolbox` 为空目录，非 git 仓库。判定：全新项目，采用参考架构（docs/architecture.md）。
- 环境：Python 3.9.6 (/usr/bin/python3)、Node v25.2.1、网络可用（pypi.org HTTP 200）。
- `python3 -m venv .venv`；安装 `pytest 8.4.2, onnx 1.19.1, numpy 2.0.2`（成功）。
- 后台安装 CPU torch → **torch 2.8.0 安装成功**（E3 解除阻塞）。
- 决策记录：前端零构建 vanilla JS + SVG 裁剪渲染；后端 stdlib http.server；自研分层布局与匹配。理由见 architecture.md。
- fixtures 一律合成生成，文档将标注为合成数据。

## 阶段B–F/G-partial 执行记录（2026-10-28 续）

- B: `ir.py`（含 build_edges 修复：initializer 不再误报悬空）+ `onnx_importer.py` + `make_fixtures.py`（15 个合成 fixture）+ 12 测试通过。修复：dtype 提取（DataType.keys() 返回名字非值）、重名节点后缀、负 dim→None、node_removed fixture 断言。
- C: `matcher.py`（锚点3级+传播+歧义）+ `diff_engine.py`（10 类差异+边diff+往返序列化）。修复：IO 锚点要求 op 一致防过度自信、参数签名去名化（w1/w2 假唯一锚点 bug）、B 侧歧义对称上报、params 位置比较（重命名容忍）。10+8 测试通过（含 8 类规定场景）。
- D: `layout.py` Sugiyama-lite + `layout_diff_pair`（matched pair 共享 y 硬保证）。8 测试通过；1k 节点布局 0.65s。
- F: `profile_adapter.py`（JSON/CSV、单位换算、逐行报错）+ `profile_mapper.py`（5级可靠度、4类状态、指纹警告）。18 测试通过。修复：二进制文件 UTF-8 解码错误→可读 422。
- E: `folding.py`（匹配图连通分量、min_size/min_confidence、折叠视图无丢失有测试保证）+ `pytorch_worker.py`/`pytorch_importer.py`（子进程隔离、超时 kill、leaf 规则、FX 降级）。10+10 测试通过。发现：eval() 后 `if self.training` 可被 trace，改用数据依赖控制流测降级。
- server/export: 15 API 测试通过。修复：FRONTEND_DIR 层级、坏 profile 422。
- **累计 92/92 测试通过**（pytest tests/）。
- subagent 并行派发被平台拦截（用户确认并发不足），全程自主顺序执行。

## 阶段G–H 执行记录（2026-10-28 续）

### 前端实现与浏览器 E2E（真实执行）
- 前端：index.html / styles.css / api.js / graph.js（SVG+视口裁剪+缩放平移）/ app.js（三模式）。
- 修复：fitView 改用 box bbox（x 输入伪节点在视口外）；热力填充用内联样式（SVG attribute 被 CSS 覆盖）；深色填充文字转白；compare 默认开 diff 图层；折叠视图保留伪 IO 节点；对齐布局伪节点跟随邻居 y。
- 浏览器实测：单模型加载+详情 ✓；profile 导入+热力图着色 ✓（conv_2 深蓝=844us 最大）；双模型 diff（Relu→LeakyRelu 红框 M 徽标、字段级详情、置信度警告）✓；差异导航定位双侧 ✓；同模型折叠组摘要节点 ✓；ambiguous_diamond 歧义（黄虚线框 + ? 徽标 + "存在多个等价候选"文案 + 仅看差异变暗）✓；PyTorch 层级（/tmp/pt_demo：10 模块、FX 捕获、leaf 规则、概览/详细切换）✓；坏文件可读报错 ✓。
- 发现并修复 fixture 缺陷：ambiguous_diamond 原 Relu 有唯一同名 → 名字锚点合理消歧未触发歧义路径；改为空名 Relu（真实导出常见）后端到端 ambiguous=4。
- 双模型热力图补齐：server 将 diff 两侧注册为可映射模型；compare 侧栏 profile A/B 导入；diff 边框 + 热力填充独立通道同开可区分（截图验证）。

### 压测（scripts/stress_test.py，结果 docs/stress_results.json）
第一轮暴露真实问题：
1. stress 脚本 rename_all 用节点名映射表改 tensor 名 → B 侧图断链成孤岛（匹配器对坏图诚实输出全歧义）——重写 rename_all。
2. matcher 传播 O(N²) 全对打分（5k 141s）+ 歧义过早永久化（1k 假歧义 234 个）——重写为增量前沿传播（候选=已匹配邻居伙伴的邻域），tie 推迟到收敛，剩余对称组诚实标歧义。
3. layout barycenter 用 list.index O(N²)（5k 13.5s）→ pos dict O(1)；layout_diff_pair rebuild 每节点线性扫 pairs → 预建映射（8s→4.2s）。
最终真实数据（本机 macOS arm64 / Python 3.9 / CPU）：
- 1k：import 0.08s / match 0.065s（1078 全配对 0 歧义）/ diff 0.03s / layout 0.1s / **HTTP diff 端到端 0.65s（2.0MB）**
- 5k：import 1.26s / match 1.33s（5398 全配对 0 歧义）/ diff 0.21s / layout 2.1s / pair layout 4.5s / **HTTP 10.1s（10.3MB）**
- 浏览器 5k：加载 8s（后端 4.1s + 前端解析渲染），缩放交互 ~320ms，**视口裁剪生效：10399 边仅绘制 271 条、5398 节点仅绘制 141 个**。
- 92/92 测试保持全绿。

## Netron 风格重构（2026-10-28，用户反馈驱动）

- 用户要求：节点详情就地显示（点击展开/再点收起）；加载方式改为文件选择/拖拽（去路径输入框）；整体参考 Netron 视觉。
- 后端：新增 `/api/model/upload`、`/api/profile/upload`（base64 上传）、`/api/model/nodes`（按需节点详情）、`/api/diff` 支持 data_a/data_b 上传字节；layout edge_points 附带 tensor 名；diff 响应附轻量 nodes_a/nodes_b 摘要。测试 92→94。
- 前端重写：Netron 式菜单栏（打开模型…/对比…/PyTorch…/导出 + 图层开关）；文件选择器 + 全窗口拖拽（拖入两个文件自动对比）；节点 = op 名 + 输出 tensor 名 + 边 tensor 标签；**点击节点就地展开详情卡（HTML overlay，含 tensors/attributes/weights/字段级变化/性能），再点或点空白收起**；右侧属性抽屉；对比对话框；图例浮窗。
- 浏览器实测：文件链路加载（DataTransfer 注入 File 走完整 base64→API→渲染）✓；节点展开显示 tensors+weights ✓；对比模式 B 侧 LeakyRelu 展开显示字段级变化与置信度（按需 /api/model/nodes 拉取）✓；再次点击收起 ✓；空白点击收起当前侧 ✓。
