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

## 无工具栏交互重构（2026-10-28，用户反馈驱动）

用户要求完全去掉工具栏，重新设计交互。落地模型（全部浏览器实测）：
- 空状态画布欢迎卡（打开模型/对比两个模型/PyTorch 项目）+ 快捷键提示；加载后自动消失。
- ⌘/Ctrl+K 命令面板：全部动作（打开/对比/导出/适合视图/属性）+ 节点模糊搜索，回车跳转并就地展开详情卡（browse/compare A·B/pytorch 三个上下文）。
- 右键上下文菜单：画布空白（打开/对比/适合视图 F/模型属性/加载 Profile/图层面板/导出）、节点（展开收起/另一侧定位/复制信息）、PyTorch（概览详细/返回画布）。
- 图层开关合并进右下角"图层与图例"浮窗（热力/Tensor名/同步/仅看差异 + 图例色带，可关，右键可再开）。
- 快捷键：O 打开、D 对比、F 适合视图、Esc 收起/关闭、⌘K 面板。
- 常驻 UI 仅剩：底部状态栏 + 图层面板浮窗（可关）+ 差异导航浮窗（对比模式）。
- 修复：命令面板跳转后自动展开详情卡；CSS 旧 #legend 块清理。
- 94/94 测试保持通过。commit: canvas-first UI。

## UNet fixtures（2026-10-28，用户需求）

- 新增 `fixtures/pytorch/unet_models.py`（UNet5：5 层 512×512；UNet5Stem：前后 3×3 conv + 全局残差 y=stem_out(unet(stem_in(x)))+x）与 `scripts/make_unet_fixtures.py`（torch.onnx.export → unet5_512.onnx / unet5_512_stem.onnx，各 1.5MB、65/69 节点）。
- 过程中发现并修复三个真实缺陷：
  1. **matcher 缺前缀包装名字证据**：B 侧 torch 导出带 `/unet` 前缀（/up1/up/Constant vs /unet/up1/up/Constant），8 个 Constant 落入对称歧义。新增 `_name_relation_bonus`（路径边界对齐的后缀关系 +0.6）→ UNet 65/65 全匹配、0 歧义。
  2. **confidence 传播早期锁定**：中位数 0.77（早期邻居少），折叠失败。修复：全部配对完成后用最终状态重算（max 不降级）→ 中位 0.92。
  3. **folding 邻接索引 bug**：`p2[1]`（A 侧 id）误用于 B 侧边检查，应为 `p2[2]`——小图同名测试掩盖了它，UNet 前缀名暴露。修复 + 新增异名回归测试。
- 结果：diff = 64 unchanged + 1 connection_changed（/inc/conv1 上游变为 stem_act，诚实语义）+ 4 added（stem_in/act/out/Add）；折叠 @0.9 出 57 节点组。浏览器实测：拖入两文件自动对比、折叠勾选、差异导航 1/5 跳转。
- 修复 UI 残留 `setStatus is not defined`（重写遗留）与 fitView 对又长又扁图的过缩（scale<0.45 时适配高度锚定左上）。
- 测试 101/101。

## UNet 双工具结构对比（2026-11-01，用户指出结构问题）

用户对比本项目与 Netron 渲染同一 unet5_512.onnx，指出 Constant/Concat 全堆在顶部一排结构不合理。根因：rank 用"最长路径（尽早放置）"，纯供料侧链（Resize 的 scales 构建 Constant→Concat）被放在 rank 1-2，而消费者 Resize 在主干 10-46 层 → 顶部杂带 + 超长垂边。

修复（对标 Netron 的布局语义）：
1. `_sink_feeders` 下沉 pass：每个节点 sink 到 min(后继 rank)-1（可行性保持：边距 ≥1；主干节点 ceiling==自身 rank 不动），迭代到不动点。效果：UNet 中 4 组 scales 侧链全部贴到各自 Resize 上层（rank 27→28 等）。
2. 大跨度 skip 边外凸绕行（span>420 时控制点外推 min(280, 70+span/5)，按 dx 方向选侧），替代穿主干的直曲线。
3. 布局方向自适应（前一轮）：深窄图自动转垂直。
4. shape 符号维度显示 `?`。

验证：新增侧链下沉回归测试（含错误图结构修正）；102→103 全部通过；浏览器复验 UNet 顶部杂带消失、侧链贴邻、skip 弧线绕行。
- push 状态：commit 4d71eb8 本地完成，`main` 领先 origin/main 2 个提交；推送时 GitHub 443 直连与本地代理(7897，已关)均不可达（网络窗口期变化）。网络恢复后直接 `git push` 即可。

## 主链居中 + 通道化（2026-11-01，第二轮结构对标）

用户继续指出"全部堆在一列，Netron 有自然的左右分野"。三个根因与修复：
1. 主干每层 slot=0 → 贴最左列：垂直布局加**主链居中**后处理（每层按 successor 链选贯通节点，slot 重排到层中间，其余节点保持相对序填空）。
2. 无左侧通道：`x_origin=320`（垂直时全部 box 右移），前端 fitView 对 vertical 以世界 0 为左界 → 视口含跳线通道。
3. skip 弧线方向随目标摆动会撞侧链：固定**走左侧通道**（dir=-1），右侧留给短侧链。
另：伪 IO 节点跟随 anchor 列；fitView 最小缩放 0.55。
浏览器复验（decoder 区域）：主干垂直居中偏左、右侧 scales 短支、左侧长 skip 弧线通道——与 Netron 第二张图同构。103/103 测试通过。

## 内嵌 Netron 官方 viewer（2026-11-01，用户要求"完全照搬"）

用户要求渲染逻辑完全照搬 Netron（开源 MIT）。方案从"模仿"升级为"直接 vendor 官方 viewer"：
- `frontend/netron/`：从 pip 包 netron 9.3.4 复制官方 viewer（index/view/grapher/base/onnx 等 18 个文件，按 import 闭包取最小集），保留 MIT 归属（NETRON-LICENSE.md）。
- 后端：`/netron/` 静态路由（路径遍历防护）+ index.html 注入 `<meta name="file" content="/netron/data/<id>.onnx">`（模仿 netron.server 的注入方式）+ `/netron/data/<id>.onnx` 返回会话内原始模型字节；STORE.models 增加 raw 字段（path/upload 双入口）。
- 前端：browse 单模型画布默认切换为 Netron viewer（iframe），图层面板提供「Netron 官方渲染（画布）」开关；自研渲染保留（对比模式的双模型 diff/折叠是 Netron 没有的能力）。
- 修复路由 bug：前缀剥离偏移（[8:]→[7:]）、默认页归一路径、data 端点剥 .onnx 后缀、坏文件 OSError→ModelLoadError。
- 测试 103→106（+netron 路由测试）。浏览器验证：UNet 在 ModelToolbox 内以官方 Netron 结构渲染，左右分野与 netron.app 一致。

## 修复 Netron viewer 集成三连（2026-11-01）

用户反馈"勾选 Netron 官方渲染没有任何显示差异"。排查发现实装只完成一半（patch 静默失败：applyBrowseRenderer 无调用点、toggle 无 handler）——此前用户看到的其实是自研布局的输出。补齐 3 个调用点后暴露后续 4 个真实缺陷，逐一修复：
1. preload require 链缺失模块（hdf5/flatbuffers/flexbuffers/tar/python 等）→ terminate 于欢迎页黑屏。修复：vendor 全量官方 js（28MB，按包整拷）。
2. GA 遥测 + ipinfo.io 外呼违反本地优先原则 → browser.js 去除 consent/telemetry 流程，base.js Telemetry 永久 _enabled=false（send no-op），记入 NETRON-LICENSE.md 修改声明。
3. meta file 用绝对路径被 viewer _url() 拼成 /netron/netron/data/... 404 → 改相对路径 data/<id>.onnx。
4. data 端点未剥 .onnx 后缀 → 404。
最终浏览器验证：bodyClass="default"，Netron 官方渲染 UNet 完整（深色主题/主干居中/权重形状行/tensor shape 边标签/skip 弧线），106/106 测试通过。
