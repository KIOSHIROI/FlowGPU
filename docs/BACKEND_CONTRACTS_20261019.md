# 两块目标后端的代码生成契约（2026-10-19 官方实测）

这两条不是我推断的，是从 **11 次官方评测**里用单变量对照逼出来的。写成契约是因为
它们直接影响 FlowGPU 该发射什么形状的核；写在文档里而不是只写在比赛目录里，是因为
它们属于编译器能力边界，不属于某一道赛题。

---

## 契约一：昆仑芯的被掩码 load 不遵守 `other`

### 证据

Task144（`causal_conv3d_cat_pad`）的第 6 次与第 7 次提交**只差一处**：
第 6 次 load 不带掩码，第 7 次带掩码（地址在两次中都已夹紧到缓冲区内）。

| 尝试 | 网格 | 解码 | load 掩码 | 地址夹紧 | store 掩码 | 昆仑结果 |
|---|---|---|---|---|---|---|
| 6 | 一维扁平 | 扁平 | **无** | 有 | 有 | **7/7 通过** |
| 7 | 一维扁平 | 扁平 | **有** | 有 | 无 | **Failed** |

第 7 次仍然报错说明**夹紧地址不能救掩码**；第 6 次通过说明**去掉掩码就能救**。
第 6 次同时证明 **store 的掩码是好的**（它带了 `mask=valid` 并且通过）。

另外两次对照排除了别的东西：

- 尝试 1（三维网格 + plane-id 解码）与尝试 5（一维扁平 + 扁平解码）**报出逐例相同的
  错配数**（3414/9600、58997/207360、10579/51840、100144/887808）→ **网格与解码器都被排除**。
- 尝试 6（正确但 `<0.01×`）与尝试 8（正确且 `0.13×`）只差**地址是否仿射**：
  夹紧写成 `tl.minimum(tl.maximum(off, 0), N-1)` 是非仿射的，摧毁了向量化。

### 因此生成规则是

1. **不要依赖 `other` 填充被掩码的通道**。需要"范围外取零"时，用**不在核里读**的方式：
   先把输出整块置零，再只写有效元素；或用不依赖掩码的取值路径。
2. **地址必须对线程索引仿射**：写成 `base + constexpr 乘加`，不要写 `min/max` 夹紧。
   这条同时是性能要求和正确性要求。

**FlowGPU 当前状态**：**尚未实现**。当前发射器默认写
`tl.load(ptr + addr, mask=..., other=0)`。上面两条规则意味着需要一条
"平台不支持 masked-load `other`" 的能力位，并在该位为假时改发射形状。
目前只有本文档，没有代码。

---

## 契约二：昇腾的 `coreDim` 上限约束的是**总 program 数**

### 证据

Task144 尝试 9 的网格是 `(b*c, t_out, h_out)`，每维都不大，但官方直接拒绝：

```
Invalid_Argument(EE1003): KernelLaunch failed because value 78336 for parameter
coreDim is invalid. Expected value: less than or equal to 65535.
```

`78336 = 1 × 17 × 4608` 是**三轴乘积**。同一个错误在 Task155 的首次提交里也出现过
（`coreDim 65536`），那次是一维网格，所以看不出区别；三维网格这次说明**上限在总数**。

### 编译器里已修的部分

`flowgpu/compiler/backend/triton_backend.py` 原来的二维发射路径
（`grid_limited_body_2d`）是**逐轴**封顶到 `max_grid_dim` 的：

```python
(f"({capped_dim(rows, limit)}, {capped_dim(columns, limit)})")
```

两轴各自封顶到 65535，乘积可达 65535²，**根本不是上限**。这是一条潜伏的 bug：
只要是二维发射（matmul、transpose）且两块尺寸都很大，昇腾就会拒绝。

现在改为 `capped_grid_2d`：每轴封顶 `floor(sqrt(limit))`（对 65535 是 255），
于是乘积 ≤ 255² = 65025 ≤ 65535。选常数而不是
`limit // rows_launched` 的原因有两条，都写进了测试：

- 后者需要把行表达式在自己的上限里重复展开，生成的发射语句会变得不可审计；
- 后者在**退化形状**（行 tile 数为 0）上会 `ZeroDivisionError`，而常数不会。
  这是旧实现没有的失败模式，不能因为"形状不会退化"就引入。

配套测试（`test/test_platform_configs.py`）：

- `test_the_emitted_two_dimensional_grid_bounds_the_program_count` —— **直接 eval 发射出去的
  表达式**（而不是用 Python 的 `min` 复述一遍规则），在一组形状上断言
  `launched[0] * launched[1] <= 65535`；
- `test_a_degenerate_two_dimensional_launch_does_not_divide_by_zero`；
- 两条原有的逐轴上限测试改为断言乘积语义。

**边界**：`isqrt` 策略比必要值更保守——两轴同时接近上限时，本来可以发射到 65535 个
program，现在最多 65025 个。program 更多本身可能是更优的映射，但**先保证不被拒绝**，
是否要按实际乘积做更紧的封顶，需要设备侧证据才能定。

**未验证**：以上全部是 CPU 侧测试与官方拒绝日志的推断吻合，没有在昇腾设备上跑过修改后的
二维发射路径。

---

## 顺带记录：华为的性能（未解决）

华为从来不是正确性问题，而是**每种设计都比融合 baseline 慢约 14 倍**：

| 设计 | 华为 |
|---|---|
| 通用模块（三维网格，4096 元素/program） | 0.07× |
| 同上，`_MAX_BLOCK` 4096 → 8192 | 0.07× |
| 行粒度网格 + 运行时跨步循环 | 0.03× |
| 扁平一维，BLOCK 1024 | <0.01× |
| 三发射、掩码自由、仿射写偏移 | <0.01× |
| 三维网格 `(b*c,t_out,h_out)`、**零整数除法** + R 行分组 | 0.03× |

最后一行把"除法贵"这个假设也否掉了（去掉全部 `//`/`%` 后反而从 0.07 掉到 0.03）。
榜首在同一块芯片上是 **0.175×**，所以坎可达，但需要比我最好成绩再高 2.5 倍。
**没有昇腾设备就无法继续**，这里不做没有证据的猜测。

---

## 开放项：还有三条三维发射路径没有被封顶覆盖

修二维路径时顺手查了一遍所有多轴发射点，`program_id(2)` 出现在三处：

| 文件 | 用途 |
|---|---|
| `flowgpu/compiler/backend/cache_store_backend.py` | KV cache 写入，`kv_axis = tl.program_id(2)` |
| `flowgpu/compiler/backend/linear_decode_backend.py` | linear decode，`db = tl.program_id(2)` |
| `flowgpu/compiler/backend/cross_domain_backend.py` | 跨域归约分段，`segment = tl.program_id(2)` |

三处都**没有**调用 `grid_ceiling` / `max_grid_dim` / `grid_limited_*`。也就是说它们
无条件发射三维网格，与平台是否设了上限无关。

**是否可达**：`ascend` 平台把 `allowed_kernel_kinds` 限成 `("pointwise",)`，所以走不到；
但 **`ascend-910b` 没有任何 kind 限制**，上面三种核都能发射。Task155 在官方 harness 上
用的正是 `ascend-910b`。所以在 `ascend-910b` 上，这三条路径是**可触发的活 bug**：
一旦三维乘积超过 65535，就会得到与今天完全相同的 `Invalid_Argument(EE1003)`。

**为什么这一轮没修**：三条路径的网格语义各不相同（分段数、KV 轴长度、decode 块数），
封顶之后还要保证"每个 tile 恰好被访问一次"，需要逐条写跨步循环并逐条测试；
在一个已经跑了 1663 项回归的改动里顺手塞进去，风险高于收益。下一轮单独做，
每条配一个"访问恰好一次 + 乘积不超限"的测试，做法与 `capped_grid_2d` 一致。

**在那之前的行为**：大形状 + `ascend-910b` + 这三类核 = 运行期被设备拒绝，
而不是编译期报错。这是已知且已记录的缺口。

---

## 已内化进编译器（2026-10-19 18:20）

上面两条契约原先只存在于这份文档里。现在它们是可执行的东西：

### 1. 能力位（`flowgpu/compiler/backend/platform_configs.py`）

`PlatformConfig` 新增三个字段，随平台表一起声明，并像其它字段一样可被
`resolve_platform(...)` 覆盖（实验用）：

| 字段 | 含义 | cuda | ascend / ascend-910b | kunlunxin |
|---|---|---|---|---|
| `masked_load_other_honoured` | 带 `other=` 的被掩码 load 是否返回该填充值 | True | True | **False** |
| `block_ptr_reliable` | `make_block_ptr` + `boundary_check`/`padding_option` 是否可靠 | True | **False** | True（未测） |
| `vector_tile_2d` | 是否应让每 program 处理 2-D tile（显式 stride） | False | **True** | False |

每个取值旁都在源码里写了对应的单变量证据（第几次尝试、翻正了什么），
不是"文档说"。`__post_init__` 里加了布尔类型校验。

### 2. 源码审计器（`flowgpu/compiler/backend/contracts.py`）

```python
from flowgpu.compiler.backend.contracts import violations, assert_contracts, audit_pipeline

problems = violations(kernel.source, platform)   # 空列表 = 生成码在该平台契约内
assert_contracts(kernel.source, platform)        # 违规即抛 ValueError
report = audit_pipeline(pipeline, platform)      # 逐核审计，返回 {kernel: [问题]}
```

它按**括号配对**扫描 `tl.load(...)`，所以跨行书写的发射也会被查到——
按行扫描会漏掉续行上的 `other=`。这是刻意的：发射出来的
`tl.load` 经常换行。

触发规则只有两条，都来自实测否定结果：

- `masked_load_other_honoured=False` 且某次 `tl.load` 同时带 `mask=` 与 `other=`
  → 报"该目标的掩码 load 不遵守 `other`；请用分配来拥有零区，并保留 store 掩码"。
- `block_ptr_reliable=False` 且源码出现 `make_block_ptr` / `boundary_check` /
  `padding_option` → 报"该目标上不可靠；请改用 2-D `tl.arange` tile + 显式掩码"。

### 3. 测试（`test/test_backend_contracts.py`，12 条）

| 覆盖面 | 内容 |
|---|---|
| 能力位 | 昆仑为 False、昇腾 `block_ptr_reliable=False`、2-D tile 轴只对昇腾开启；布尔校验；可覆盖 |
| 审计器 | 掩码 load + `other=` 只在昆仑报；无 `other=` 处处干净；`block_ptr` 只在昇腾报；**跨行调用能被括号配对捕获** |
| 真实发射码 | `audit_pipeline` 能跑到每一个 lower 出来的核 |

全量回归：**1675 passed, 4 skipped**（此前 1663，净增 12）。

### 未做、且明确记下的部分

- 审计器目前**只是可调用的检查**，还没有被无条件塞进 `compile_source` 主路径。
  原因是它今天对真实发射码不产生告警（见测试注释），先让它可用、可测，
  等真正会触发它的 codegen 路径改动时再挂到入口。
- `vector_tile_2d` **还没有发射器真正消费它**。把它写成能力位是为了先固定结论；
  让发射器据此选择 2-D tile 是一次独立的 codegen 改造，不在本次范围。
  这条差距必须说清楚：**能力位是知识，不是实现。**

---

## 追查到后端实现：Intel XPU backend for Triton（2026-10-20）

用户给了 <https://github.com/intel/intel-xpu-backend-for-triton>。查这个仓库是因为两条线索
指向它：我们在 batch-8 里见过昆仑芯报 `OutOfResources: uni_sram`（`uni_sram` 是 Intel XPU
的 SLM/统一共享内存术语），以及 FlagTree 自己带 [Triton Intel XPU update 材料](https://raw.gitcode.com/flagos-ai/flagtree/raw/triton_v3.5.x/docs/meetups/10-25-2023/intel-xpu-update.pdf)。

**注意归属边界**：以下三条是这个后端的**已知行为**。它是否就是昆仑芯那条通路，我仍未直接证实
（证据是 `uni_sram` 报错术语 + FlagTree 打包关系）。所以这些是**强假设来源**，不是定论。

### 1. 掩码 load 变慢最多 3.5 倍，且有开关

[Issue #7755](https://github.com/intel/intel-xpu-backend-for-triton/issues/7755)：
3.8 起默认把普通 masked `tt.load` 降到 `__spirv_PredicatedLoadINTEL`，而 3.7 是
**"先按 mask 分支、再做普通 load"**。在 Arc Pro B60 上，掩码多的核慢 **1.29×**，
单个核（22 个 load）慢 **3.5×**。设 `TRITON_INTEL_PREDICATED_LOAD=0 fully` 恢复到 3.7 水平。

→ 对我们的意义：**掩码越少越好**，这与我在昆仑实测的"去掉 load 掩码才能得到正确结果、
而且掩码本身是性能负担"方向一致。

### 2. 带 `other` 的掩码 load 曾经是**类型错误**，会编译失败

[PR #7938](https://github.com/intel/intel-xpu-backend-for-triton/pull/7938)：
`reshape1DStridedLoad` 把一维 strided load 改写成携带硬件投递编码的二维 load 时，
**`other` 没有跟着 reshape**，于是

> `'tt.load' op failed to verify that other matches ptr type`

2026-09-04 才修。也就是说在某些版本上，**带 `other` 的掩码 load 不是"值不对"，而是编不过**。

### 3. 掩码 load + 整数整除，会让**掩码外的 lane 越界写**

[Issue #8203](https://github.com/intel/intel-xpu-backend-for-triton/issues/8203)：
带 `other=0` 的掩码向量 load 产生 phi；随后整数 `div/rem` 用这个 phi 做除数且**没有零保护**；
IGC 的 `igcscalarize` + SimplifyCFG 插入 `assume(mask)`，GVN 把 `PredicatedStore` 的掩码折成
`true`，于是 **"every masked-off lane of the last block stores out of bounds"**。

→ **这几乎逐字解释了我 10-19 看到的昆仑现象**：错配元素带 1e38 级别的垃圾值、错配比例与边界
占比吻合、而错误随 kernel 结构以确定性方式复现。当时的经验结论是"掩码 load 不可靠"；
现在有了机制解释：不是值不填，而是**掩码被折叠、越界写发生**。

### 4. 二维块投递（`ttig.block_io`）是**性能机制**，不只是写法

[PR #8112](https://github.com/intel/intel-xpu-backend-for-triton/pull/8112)：
`ttig.block_io` 让 load 走 **2D block message**（每 lane 一次 d64 投递），但要求 load 的
encoding 是**合并访问**的；若 `RemoveLayoutConversions` 把它重物化进消费者的 encoding，
这个 load 就丢掉合并编码、block_io tile 被密度检查拒绝。修好后 `gmixer_24_224` 从
1.94× 提到 **2.30×**（+18%）。

→ 这与我在华为上实测的「每 program 处理 2-D tile 才快」是**同一个机制的两种说法**：
在 Intel XPU 路线上，二维块投递是硬件通路。

### 因此对昆仑的两条新假设（未测）

1. **把拷贝核从一维扁平改成二维块（行 × W，无掩码）**，让后端能挂上 `block_io`、
   走 2D block message。现有 v8 是 `src_ptr + idx` 的一维稠密拷贝，已经合并但**没有二维形状**。
2. **彻底消除掩码**（v8 已做到 load 无掩码；但 store 仍有尾掩码）——按 #7755，
   尾掩码也是 predicated store 的开销来源。

**未验证**：本节全部是 Intel XPU 后端的**已知行为**，我们没有该硬件，也没有确认昆仑芯
确实走这条后端。它把"黑箱经验"变成了"有出处的假设"，但不构成对昆仑结果的事后解释。

# 追加：2026-10-20 第10批六次提交修正了昆仑契约

这一天在第10批交了 6 份 cross-chip 内核（Task146/147/149 通过并上榜，Task152/154 被判无效），
逐芯回执把昆仑芯的两条新契约和一条**旧契约的误报**同时确定下来。

## 1. 修正：`other=` 本身不是缺陷，"掩码不一致"才是

旧规则（2026-10-19 由 Task144 attempts 6/7 得出）是「昆仑上带 `other=` 的掩码 load 一律拒绝」。
今天的证据说明这条**过严**：

| 内核 | load 掩码 | store 掩码 | 昆仑读数 |
|---|---|---|---|
| Task147 rmsnorm 家族 | `m`（尾部） | `m` | **6.28×** ✅ |
| Task149 gemma_rmsnorm | `m` | `m` | 2.01× ✅ |
| Task146 append shared experts | `m` | `m` | 0.60× ✅ |
| Task144 attempts 6/7 | `keep` | `in_range`（更宽） | 失败/崩塌 ❌ |
| Task154 v2 patch tile | `in_image` | `in_patch`（更宽） | Correctness Failed ❌ |

所以真正的契约是 **load 的掩码必须与消费它的 store 掩码相同**（文本同一表达式）。
`contracts.violations` 已改为按此判定，并加了两条通道。

## 2. 新增：gather load 在昆仑上不可用

| 内核 | 国际通用A | 昆仑 |
|---|---|---|
| Task154 通用模块（按输出平坦索引，逐 lane 反解再 gather） | **1.81×** | **0.02×** |
| Task154 源序改写（load 用平坦源索引，置换挪到 store） | 1.24–1.78× | 0.05× |

50 倍的塌陷，且改写也只在 0.05× ——是**这一类**不可用，不是某一种写法。
判据：`tl.load` 的首参表达式中出现 `//` / `%`，或引用了由它们算出的变量（单跳污点）。

## 3. 新增：store 的值**循环携带**时在昆仑上不可用（不是"循环里有 store"）

| 内核 | 循环形式 | 昆仑 |
|---|---|---|
| Task152 kda_gate_chunk_cumsum（chunk 内 running accumulator + 每步一次 store） | 运行期上界 `range(0, n)` | **0.03×** |
| 同上 | `tl.static_range` 全展开 | **0.03×** |
| Task147/149 流式路径（`static_range` 里也 store，但每轮独立、无跨轮依赖） | `tl.static_range` | **6.28× / 2.01×** ✅ |

所以判据不是「循环里有 store」——那样会把 147/149 误判——而是**store 的值依赖同一循环里
被 `+=` 更新的变量**（一条依赖链）。检查器按循环体做两跳本地解析：收集循环体内的
augmented assignment，再用一次赋值传播，最后看 `tl.store` 的 value 参数是否引用它们。

## 4. 于是昆仑可用的集合只剩一种形状

> 合并访存 load（地址是平坦索引或常量乘加）+ 寄存器内累加/变换 + **一次仿射 store**，
> 且 load 与 store 用同一个掩码。

`PlatformConfig` 新增两个能力位：`gather_load_ok`、`store_inside_loop_ok`，昆仑芯均为 False
（`resolve_platform` 可显式覆盖，便于做对照实验）。

## 5. 验收：静态检查复现全部 6 次官方判定

拿今天真正提交过的 6 份内核源码跑 `violations(source, kunlunxin)`，与实际回执逐条对照：

| 内核 | 官方昆仑读数 | 静态检查 |
|---|---|---|
| 146 append shared experts | 0.60× ✅ | 0 flags ✅ |
| 147 dual rmsnorm（流式路径） | 6.28× ✅ | 0 flags ✅ |
| 149 gemma_rmsnorm | 2.01× ✅ | 0 flags ✅ |
| 152 chunk cumsum 昆仑槽 | 0.03× ❌ | 3 flags（掩码不一致 + 循环携带 + …） ✅ |
| 154 patchify 通用 | 0.02× ❌ | 4 flags ✅ |
| 154 patchify 昆仑槽 | 0.05× ❌ | 2 flags（gather load + …） ✅ |

**6/6 一致**——包括把 147/149 从"误报"里救回来这一步。这一步正是被误报逼出来的：
第一版检查只看「循环里有 tl.store」，把 147/149 判成违规，与官方 6.28×/2.01× 冲突。
**是官方数据纠正了检查规则，而不是检查规则去迁就实现。**

全量回归：`1691 passed, 4 skipped`（此前 1687，净 +4 条测试）。
