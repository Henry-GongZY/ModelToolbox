# Profile 格式说明

ModelToolbox 只接受**文档化的标准格式**，不做任意厂商格式猜测。所有时间在内部统一为微秒（us，浮点）。

## 标准 JSON（schema_version 1）

```json
{
  "schema_version": 1,
  "model": "model.onnx",
  "device": "target-board-name",
  "unit": "us",
  "end_to_end_us": 1500.0,
  "model_fingerprint": "<sha256 hex，可选扩展字段>",
  "measurements": [
    {
      "node_name": "conv_1",
      "op_type": "Conv",
      "duration": 123.4,
      "calls": 1
    }
  ]
}
```

### 字段规则

| 字段 | 必填 | 说明 |
|---|---|---|
| `schema_version` | 是 | 仅支持 1；更大的值按 best-effort 解析并警告 |
| `model` | 否 | 模型文件名（展示用） |
| `device` | 否 | 设备名（展示用） |
| `unit` | 是* | 全局单位：`us`/`μs`/`ms`/`s`/`ns`（大小写不敏感）；缺省按 `us` |
| `end_to_end_us` | 否 | 端到端总耗时（**全局单位**）。与逐层占比是不同统计口径，UI 分开展示 |
| `model_fingerprint` | 否 | 扩展字段：模型内容 sha256。与加载模型不符时 UI 显式警告（不静默套用） |
| `measurements` | 是 | 测量记录数组，永不丢弃任何一条 |

### 每条测量记录

| 字段 | 必填 | 说明 |
|---|---|---|
| `node_name` | △ | 节点名（与 node_id/tensor/kernel_name 至少给一个） |
| `node_id` | △ | 扩展字段：Graph IR 的 node id（最可靠引用） |
| `tensor` | △ | 扩展字段：tensor 名（映射到其生产者节点） |
| `kernel_name` | △ | kernel 标识；通常意味着融合/覆盖多节点 |
| `op_type` | 否 | 仅在名称未命中时用于"唯一 op 回退"（多个同 op → 歧义，不猜） |
| `duration` | 是 | ≥0 数值，按 `unit`（或记录级 `unit` 覆盖）换算为 us |
| `calls` | 否 | 调用次数，默认 1，整数 ≥0 |
| `fused` | 否 | true 表示该记录覆盖多个节点（如融合 kernel），不映射到单节点 |
| `min`/`avg`/`max` | 否 | 附加统计量，原样保留在 raw 中并展示 |

### 占比（share）语义

- **默认占比 = 该节点 duration / 所有已映射记录 duration 之和**（只统计成功映射的记录）。
- `end_to_end_us` 不参与占比分母；两者同时存在时 UI 并排展示并标注口径。
- 未映射/歧义/融合记录不计入占比，但在"未映射记录"列表中完整保留（含原因与候选）。

## 标准 CSV

首行表头，列名固定：

```csv
node_name,op_type,duration_us,calls
conv_1,Conv,620.5,1
relu_1,Relu,35.2,1
```

- 必需列：`node_name`、`duration_us`（已是微秒）。
- 可选列：`op_type`、`calls`、`unit`（记录级单位覆盖）、`node_id`、`tensor`、`kernel_name`、`fused`。
- 未知列 → 警告并保留；缺必需列 / 行字段数不符 / 非数值 → `ProfileFormatError`（带行号）。

## 映射可靠度（从高到低）

1. `model_fingerprint` 与模型内容/结构指纹匹配（不匹配 → 全局警告，仍可映射但 UI 明示）
2. `node_id` 精确命中
3. `node_name` 精确命中（重名 → 歧义，不强行分配）
4. `tensor` 名 → 生产者节点
5. `op_type` 唯一回退（同 op 多实例 → 歧义 + 候选列表）

**禁止**按出现顺序分配耗时。每条记录的最终状态必为：`mapped / unmapped / ambiguous / fused_kernel` 之一，可导出追溯（`/api/export/profile`）。

## 示例文件

- `fixtures/profile_linear.json` / `fixtures/profile_linear.csv`（对应 `fixtures/linear.onnx`，合成数据）
