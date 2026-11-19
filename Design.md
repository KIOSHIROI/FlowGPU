# FlowGPU 语言与编译器设计

版本：v0.1.2（语言层）；当前语义修订日期：2026-10-02。
本文件描述当前实现。语法见 EBNF，使用与复现见 flowgpu/README.md，
实验证据与历史数据审计见 flowgpu/docs/EXPERIMENT_AUDIT.md。

## 1. 领域与目标

FlowGPU 面向 LLM 推理中的 GPU 算子子图，表达计算意图、张量类型与数据依赖。
核心原则是 Write Operators, Not Kernels。Agent 生成 DSL，编译器负责合法性检查、
融合、缓冲区规划与代码生成。当前实现是研究原型，支持 Triton 与 Python 参考后端。
不包含生产级推理运行时、完整模型加载或 CUDA 后端。

编译器验收目标：同一份高层程序，通过通用编译优化，在一类明确的推理子图和
未参与调参的形状上，取得正确、稳定、接近可靠手写基线的性能，同时减少人工干预。
Graph、完整调用、编译成本和临时内存分别检验。具体边界及当前缺口见
[编译器目标与验收](docs/COMPILER_ACCEPTANCE.md)；该目标尚未整体达成。

## 2. 问题定义

输入是无分支、无循环的算子流水线；输出是静态分析结果、生成代码和结构化诊断。
GPU 调度、tile 与 mask 由后端生成。源码生成不依赖 GPU，GPU 正确性需要设备验证。

## 3. 设计原则

显式声明输入，积极推导中间类型，一个计算对应一个 AST Call 和一个 IR 算子。
例如以下 P3 片段会推导 y/z 的 dtype 与 shape，MatMul 与 ReLU 可形成 epilogue：

```
input x : bf16[N,K]
input w : bf16[K,M]
y = matmul(x, w)
z = relu(y)
output z
```

与设备或算子语义有关的信息必须明示。RoPE 需要位置对应的 cos/sin 张量，
不能用身份变换冒充旋转。没有签名的算子保留并报错。

## 5. 名称与程序规则

文件包含一个或多个 pipeline。输入和赋值必须先定义后使用，每个名称只能绑定一次。
output 可出现多次，输出值保持活跃到整个程序结束。语句由语法识别，分号不属于语言。
+、-、* 分别是 add/sub/mul 的语法糖，* 优先，运算符左结合。
无数值字面量、隐式转换、广播、控制流、多结果赋值或维度算术。
关键字保留词的上下文名称约束仍有边界，详见 EBNF；不要使用保留词命名。

## 6. 表面语法

```
pipeline name {
    input x : bf16[N,D]
    y = relu(x)
    output y
}
```

## 7. 数据流 IR

IR 含 values、按定义排序的 nodes 和 producer-consumer edges。
嵌套表达式展开为 %0 等用户不可命名的临时值。input/output 是结构节点。
y = x 是别名，不产生算子或独立存储。SSA 图不包含基本块与 phi。
生命周期在源码节点坐标中表示，但对融合 kernel 保守扩展覆盖其所有成员。

## 8. 类型与布局

十种存储 dtype：bf16、fp16、fp32、int8、uint8、int32、int64、bool，外加值层引入的 fp8e4m3、fp8e5m2（v0.9）。
shape 是正整数或符号的非空序列，以语法相等进行约束。
整数零维在编译期拒绝，符号维度在运行前验证正值。
Triton 输入必须是同设备 CUDA 张量；pointwise/rowwise 与可选 output_gather 转置的外部输入支持非连续 strided 视图，通过实际 stride 映射逻辑坐标。默认 tiled 转置、GEMM、RoPE 等模板直接读取的输入仍必须连续；host 检查 dtype、shape、布局与设备。转置映射通过 `--transpose-mapping tiled|output_gather` 选择；CUDA 默认 tiled，国产设备白名单不会因映射选择自动扩大。
Python 参考输入是平坦 row-major 列表，host 检查符号维度与元素数量。

## 10. 算子语义与类型推导

| 算子 | 语义 | 约束 |
|---|---|---|
| add/sub/mul | 同坐标二元算术 | 两输入 dtype/shape 相同；浮点或整数 |
| relu | max(x,0) | 浮点或整数 |
| silu | x/(1+exp(-x)) | 浮点 |
| round | 就近取整，**四舍六入五成双**（与 torch.round 一致） | 浮点；输出同 dtype |
| cast_to/cast_like | 浮点→整数：**向零截断，并按目标位宽回绕**（同 torch `.to(int)` 与硬件 `.to`）；浮点→fp8e4m3：**就近舍入、ties-to-even、在 448 处饱和**（e4m3fn 无 inf）；舍入与饱和不由整数转换承担 | bool 目标拒绝；算术算子对 storage dtype 直接拒绝（须先加宽） |
| rms_norm | x / sqrt(mean(x*x)+1e-6) * w，乘权重后舍入 | 浮点；末轴规约，w 为 [D] 且同 dtype |
| rmsnorm_hf(x,w) | 归一化结果先舍入至输入 dtype，再乘 w 并舍入 | 同 rms_norm；epsilon 固定为 1e-6 |
| softmax | exp(x-max(x))/sum(exp(x-max(x))) | 浮点；末轴稳定规约 |
| matmul | 沿 K 轴求和 | 浮点；仅二维 [M,K] 与 [K,N] |
| transpose | 二维坐标交换 | 所有 dtype；仅二维 |
| rope(x,cos,sin) | x*cos + rotate_half(x)*sin | 浮点；三个输入同 shape/dtype，D 为偶数 |
| attention(q,k,v) | softmax(QKᵀ/√D)V；内部不物化低精度 scores/probs | 同浮点 dtype；q:[L,D]、k:[S,D]、v:[S,V]（2D）或 q:[B,H,L,D]、k:[B,G,S,D]、v:[B,G,S,V]（4D，G∥H 为 GQA，字面量时静态检查整除、符号维运行时检查） |
| causal_attention(q,k,v) | 同 attention，因果掩码：query i 只看 keys 0..i+(S−L)（kv-cache 偏移约定；S≥L 运行时检查）；语言无布尔字面量，故掩码约定用独立算子名表达（rms_norm/rmsnorm_hf 先例） | 同 attention |

RoPE 的 split-half 约定：D=2h，rotate_half(x)=[-x[h:],x[:h]]。
cos/sin 是完整 shape 的表；调用者按 position、theta、频率约定准备并复制半轴系数。
没有默认 position 或隐式 theta。静态奇数末维编译报错，符号奇数末维运行前报错。
不保证任意表都代表正交旋转；具体表值由调用者负责。
RoPE 在 GPU 上用 fp32 计算，再舍入到输出存储 dtype；每个 DSL 算子保持其 dtype 边界。
整数逐元素算术按 int32/int64 二进制补码宽度回绕；整数不能用于浮点专属算子。

## 11. Buffer 规划

Buffer 包含 size（符号字节乘积）、lifetime 和 storage。
输入由调用者提供；融合且无需物化的值没有全局 Buffer；其他值分配 slot。
复用要求完全相同 dtype/shape 与严格不相交的保守生命周期。
slot 复用节省分配容量，不等于消除了每次读写。

## 12. 可执行示例

### 12.1 Residual + RMSNorm

```
pipeline residual_norm {
    input x        : bf16[N,D]
    input residual : bf16[N,D]
    input weight   : bf16[D]

    y = add(x, residual)
    z = rms_norm(y, weight)

    output z
}
```

### 12.2 MLP

```
pipeline mlp {
    input x  : bf16[B,D]
    input w1 : bf16[D,H]
    input w2 : bf16[H,D]

    h  = matmul(x, w1)
    h2 = silu(h)
    y  = matmul(h2, w2)

    output y
}
```

### 12.3 QK Transformation

```
pipeline qk_transform {
    input q  : bf16[B,H,S,D]
    input k  : bf16[B,H,S,D]
    input qw : bf16[D]
    input kw : bf16[D]

    input rope_cos : bf16[B,H,S,D]
    input rope_sin : bf16[B,H,S,D]

    qn = rms_norm(q, qw)
    kn = rms_norm(k, kw)
    qr = rope(qn, rope_cos, rope_sin)
    kr = rope(kn, rope_cos, rope_sin)

    output qr
    output kr
}
```

QK 示例保留两个归一化 kernel 与两个配对旋转 kernel。RoPE 不是普通同索引逐元素算子。
`llama3_decoder_layer.fgpu` 是简化的未缩放单头 decoder 子图，历史文件名保留；
未实现多头/GQA、KV cache、因果遮罩或分数缩放，不能作为完整 LLaMA-3 的实现证明。

## 13. 编译流程

### 13.1 Lexer

flowgpu/compiler/grammar.lark 定义终结符，独立词法视图和解析器共用它。

### 13.2 Parser / AST

Lark LALR 解析；AST 保留名称与运算符源码位置。

### 13.3 类型与形状检查

注册表规则是纯类型函数，位置由调用者附加。输入、赋值、输出逐语句检查。

### 13.4 数据流 IR 与 13.5 依赖

AST 降为 DAG，显式展开嵌套与别名；边直接表示依赖，不重新从 AST 猜测。

### 13.6 融合

实现是保守启发式，不是任意 GPU 上的充要定理。
同索引 pointwise 区域可吸收到一个非 pointwise anchor 中，连接值必须单次使用，
所有外部操作数必须在整个 kernel 开始前已生成。两分支门控计算优先选择较晚的
合法生产者，否则保持物化。Add -> RMSNorm、MatMul -> SiLU/ReLU、逐元素链可以融合。
RoPE 自身是配对读取 anchor，RMSNorm -> RoPE 与 MatMul -> RoPE 当前保持分开。

默认 traffic 策略在合法放置之间比较局部全局内存读写字节：GEMM 左输入重复
ceil(N/BLOCK_N) 次，右输入重复 ceil(M/BLOCK_M) 次，prologue 叶节点继承该次数。
比较物化（叶节点读一次、结果写一次并按分块重复读）与融合（重复读取叶节点）。
该局部可加模型只在字面形状下作选择；符号形状保持原规则的 tie preference。
`--fusion-policy rule` 可复现旧放置；autotune 使用候选中最小 BM/BN 的保守重复上界。
这不是实测延迟最优：不预测 L2 命中、带宽、计算重复或 launch 延迟。

CUDA 的第二个放置阶段组合索引表达式：gather、广播、reshape、移轴、dtype 转换、
逐元素数学和末轴 sum/max 可在同一 kernel 求值。基础索引区域只保留一个外部结果、最多一个
归约；分支共享值只有在所有读者均在区域内时才消除，不克隆跨 kernel 的计算。
外部操作数须在最早成员之前就绪，归约结果的扩展广播不吸收归约，防止逐元素重算整行。
`--no-index-fusion` / API `index_fusion=False` 保留逐原语物化的同源码对照。

随后执行共享行列域的多输出归约放置：最多两个末轴归约按依赖顺序扫描，行统计只在
行 tile 内计算一次。完整域输出可在其最后一个消费者的归约扫描中写出；其他完整域输出
另作一次扫描。输出和跨域消费者仍需要全局缓冲，例如 LSE 权重供 D 维加权归约复用。
这个阶段不吸收 gather、计算结果的跨行移轴或有歧义的右对齐广播；外部操作数仍须在
区域最早成员之前就绪。`--no-reduction-fusion` / API `reduction_fusion=False` 仅关闭此阶段，
保留基础索引融合；两种开关均参与缓存键。JSON 的 `reduction_domain` 标识共享域。
该放置是保守区域合并，不依据实测延迟决定权重应物化还是重算。traffic 对两个扫描分别
统计输入读取，对缓存行统计计一次，对后续跨域广播继续计算重复读取；它不是 HBM 计数器。

`--cross-domain-strategy recompute` 为显式实验路径，默认仍是 `materialize`。它只合并闭合的
两区域：基域的末轴统计，以及在归约轴之前插入一个广播轴的加权 sum 区域。以一个基域行
和一个扩展轴 tile 为 program，先求 max、再求权重 sum、最后重算 tile 权重并作扩展轴输出的
加权 sum；基础统计每个扩展轴 tile 重算一次，权重向量在 tile 内广播到多个输出列。
只写最终行结果，基域结果由扩展轴 tile 0 唯一写出。其他外部消费者、完整权重导出、
晚到外部操作数、计算结果移轴或不匹配的域继续保持物化，不以题名匹配。
`--indexed-columns` / `--indexed-rows` 指定正二次幂 tile 上限；物化路径对应归约列和输出行，
重算路径对应 R 与插入的 D 轴。tile 总元素受 `reduction_block_limit` 约束。策略与参数进入
缓存键，JSON 的 `cross_domain` 和 `cross_tile` 显示实际放置；这个开关不运行时自动调优，
也不把某个 A800 尺寸的胜出配置自动推广至其他设备或程序。

### 13.7 生命周期与复用

生产值的生命周期起点扩展到生产 kernel 最早成员；读取值的终点扩展到消费者 kernel
最晚成员。输出终点扩展至程序结束，防止以源码节点间隙错误复用融合后的存储。

### 13.8 双后端

Triton 发出 pointwise、rowwise、matmul、transpose、rope 五类 kernel 与 host。
rowwise 每行一个 program，块宽为 next_power_of_2(D)，大轴资源开销仍需优化。
matmul 使用固定 tile，缺少 autotuning、tensor layout 优化与生产级性能策略。
2026-10-01 新增可选 hybrid GPU 模式：MatMul/RMSNorm 作为不跨越的物化边界。
MatMul 调用
PyTorch torch.mm（默认 CUDA BLAS 为 cuBLAS），RMSNorm 采用统一 FP32 库规约
序列后舍入；逐元素、RoPE 与独立 Softmax 继续生成 Triton。
这是同一算子语义的另一实现；内存计划依照新的分区重新计算。
显式 attention 算子在 hybrid 模式默认用 FP32 库 MatMul、缩放、Softmax、
FP32 库 MatMul 实现，最终 dtype 舍入；无 causal/dropout/GQA。默认纯 Triton 对
attention 给出后端未支持诊断，CPU 参考完整实现其数学定义。
可选 attention_impl="sdpa" 调用 SDPA math 并显式升 q/k/v 为 FP32。
SDK 与直接数学序列的舍入顺序可不同；SDK 端到端对照有未达标的 batch，
保留为实验选项。当前默认库数学序列，不宣称 FlashAttention。
固定形状可用 CUDA Graph 捕获同一 host 并回放，减少 Python/driver 启动开销；
输入地址、dtype、布局与精度策略保持不变，输出缓冲随每次回放覆盖。
性能记录区分普通 host 调用与 Graph 回放，后者不计输入拷贝及捕获。
GPU host 要求关闭 CUDA autocast 并检查库精度开关，不主动修改调用者进程的全局设置。
Python 参考机逐值舍入到 dtype，规约内部使用 double，因此是算法参考而非逐位 GPU 模拟器。
两后端共用计划不能替代独立数学 oracle 或 GPU 数值对比。

## 17. Agent 诊断与修复

错误类别为 syntax、type、shape、operator、dependency，携带源码行列与 pipeline。
规则抛 RuleError，AST pass 转为 CompileError。独立语义错误一次收集；PoisonType
抑制后续级联噪音。JSON 输出供 Agent 消费。
多错误收集仅证明单次编译提供了反馈；实际 LLM 自愈轮数需要真实模型调用与保存 trace。

## 19. 实现边界

类型规则完整（v0.1.2 起含 layout 算子，v0.6.1 起含 interleave）：

add
sub
mul
round
silu
relu
rms_norm
rmsnorm_hf
matmul
softmax
transpose
rope
attention
causal_attention
reshape
split
concat
interleave
exp
exp2
log
log2
neg_inf_like
ones_like
zeros_like
to_fp32
isfinite
where
broadcast_like
cast_like
cast_to
move_first_last
gather_row_offset
gather_rows
sum_axis
max_axis
softmax_axis
rms_norm_axis
div
sum_last
max_last
store_cache_4d
pad_cat
cumsum
cumsum_axis
segmented_cumsum
patchify
segmented_matmul
segmented_lora_matmul
scatter_by_index
linear_decode_forward
lt
le
gt
ge
eq
ne
abs
maximum
minimum
sqrt
rsqrt
tanh
sigmoid
softplus
erf
gelu
gelu_tanh
topk_last
gather_last
argmax_last
argmin_last
segment_rank
gated_delta_recurrence
add_broadcast
cumsum_totals
cummin_last
cummax_last
bitand
bitor
bitxor
shl
shr
float_bits
bits_float

以上清单记录前端注册的类型规则，不等同于全部目标后端已可执行。`linear_decode_forward` 当前为实验接入，2026-10-07 的冻结快照检验发现其 host ABI 和参考执行尚未闭环；不能以注册或 `complete` 标志宣称正确性。状态与证据见 `flowgpu/docs/COMPILER_STATUS_20261007.md`。

v0.1.2（2026-10-02）：三个 layout 算子进入语言层，签名如下——

| 算子 | 语义 | 类型规则 |
|---|---|---|
| reshape(x, [d1..dk]) | 行主序重解释，numel 不变 | shape 字面量每一维为正；两侧全字面时乘积必须相等，含符号维时守恒交由 host 运行时校验（与输入声明的符号正性同一策略）；拒绝 -1 占位符 |
| a, b, c = split(x) | 沿最后一轴等分，份数=左值个数 | 末轴必须为字面量且能整除份数；符号维无法表达 D/2 故拒绝；前导符号维原样保留 |
| interleave(a, b) | 沿最后一轴交替（split 二分的逆）：out[...,2i]=a[...,i] | 两操作数 shape/dtype 完全一致；末轴须为字面量（拼接宽度无法由符号维写出，与 concat 同一理由）；AWQ 的 word 内 nibble 重排需要它 |
| concat(a, b) | 沿最后一轴拼接（split 的逆） | dtype 相同、前导维完全一致；末轴字面量时结果末轴为和，否则运行时校验 |

多赋值语法 `a, b = split(x)` 进入语法；shape 字面量 `[N,32,64]` 作为 reshape 专用表达式。三个 layout 算子已由两个后端实现；Triton host 使用 reshape、切片和 cat 搬入计划的输出缓冲，尚未消除这些拷贝。

v0.3（2026-10-03）：竞赛反馈增加可组合原语。

| 原语 | 契约 |
|---|---|
| gather_row_offset(x:[T,D], i:[N]) | 取逻辑扁平索引 r*D+i[r]；负索引加 T*D 一次，仍越界则返回零；支持 int32/int64 索引与非连续输入 |
| gather_rows(x:[T,...], i:[N]) | 输出 [N,...]，按第一轴选取整行；每个负索引加 T 一次，仍越界则该整行返回零；支持 int32/int64 索引、全部存储 dtype 与非连续输入；维度仍须正数 |
| scatter_by_index(base:[N,...], index:int32[M], values:[M,...]) | 声明式索引写：`out = base` 然后把 `values[i]` 写到 `out[index[i]]`；**越界条目丢弃**（与 gather_rows 越界读返回零对称），重复索引为无序最后写赢 | rank≥1、dtype 相同、index 为 rank-1 整数、values.shape[1:]==base.shape[1:]、行数一致；base 与 index 的零轴须字面量（网格由它算） |
| segmented_matmul(x:[T,K], w:[S,K,N], indptr:[S+1]) | `out[t,:] = x[t,:] @ w[segment(t),:,:]`；段边界是显式 int32 输入，token 轴可保持符号 | x 为 rank-2、w 为 rank-3、dtype 相同、indptr 为 rank-1 整数；K/N 必须字面量（tile 需 constexpr）；S 字面量时 indptr 长度必须为 S+1 |
| patchify(x:[B,C,H,W], ph, pw) | `b c (h ph) (w pw) -> b (h w) (c ph pw)`；静态补丁几何，与 reshape 的 shape 字面量同性质 | rank 4、浮点、B/C/H/W 全字面量、ph/pw 为正整数字面量；H%ph 或 W%pw 非零即拒绝 |
| cumsum(x) | 沿最后一轴的前缀和；fp32 累加，operator 边界按声明 dtype 舍入（21.6） | 浮点；shape 不变 |
| pad_cat(x:[B,C,T,H,W], cache:[B,C,Tc,H,W], w_l, w_r, h_t, h_b, t_front, t_back) | `F.pad(cat([cache, x], dim=2), (w_l, w_r, h_t, h_b, t_front - Tc, t_back))`：缓存帧占据时间维前端填充，输出 `[B,C,T+t_front+t_back,H+h_t+h_b,W+w_l+w_r]` | rank 5、dtype 相同、B/C/H/W 一致且全为字面量、六个 padding 为静态整数字面量；`t_front < Tc` 或裁剪任一轴即拒绝（输出维 `T+t_front+t_back` 无法由符号维写出，与 concat 拒绝符号末轴同一理由） |
| move_first_last(x) | 将第 0 轴移到最后，输出连续张量 |
| broadcast_like(x, template) | 右对齐广播；仅允许相同维或单例维，template 提供形状 |
| cast_like(x, template), to_fp32(x) | 显式转换；前者保持 x 形状、采用 template dtype |
| cast_to(x, dtype) | 显式转换；target 由 dtype 字面量写出（`bf16`/`fp16`/`fp32`/`int32`/`int64`），不再需要一个模板张量。dtype 字面量只能出现在这个位置 |
| sum_last(x), max_last(x) | 浮点输入，FP32 累加/结果，保留末轴为 1；按平台 tile 上限循环归约 |
| isfinite(x), where(mask,a,b) | bool 有限值掩码；同形状、同 dtype 分支选择 |
| exp/exp2/log/log2/div | 显式浮点数学；div 同形状；不自动添加稳定性缩放 |
| ones_like/zeros_like/neg_inf_like | 同形状同 dtype 常量；负无穷仅限浮点 |

CUDA 支持上述索引表达式融合；独立物化模式仍可用于对照。其他平台保持原分区和白名单。
融合逐节点保留 dtype 舍入；归约使用有界二维 tile，按行宽选列 tile、并行最多 16 个输出行。
第八批语言样例保留 FP32 的 LSE 中间值；全无效权重保留参考语义的 NaN 输出和 -inf LSE。
当前样例不表达可选参数/返回 None、原地 scatter、副作用或动态压缩。

v0.5（2026-10-20）：值层补全（`flowgpu/compiler/backend/value_layer.py`）。

起因不是设想，而是普查：把 115 道题的**参考实现**逐题读下来，按它实际调用的原语分类
（`the language-gap survey`），最大缺口在字段级运算而不是寻址——
49/115 在比较取值，40/115 调用注册表里没有的初等函数。而当时 `where`（语言唯一的选择运算）
只能被 `isfinite` 驱动，因为**语言没有比较运算**：除了"这是不是有限值"之外造不出 bool。
`relu`/`silu`/`fused_sigmoid_mul` 都在，却没有 `sigmoid`；这是专用融合算子堆叠、
通用原语缺位的结果，不是有意的取舍。**v0.8 已删除 `fused_sigmoid_mul`**：它要的
就是 `FP32(x) * sigmoid(FP32(g))` 一次舍入，现在由
`cast_like(mul(to_fp32(x), sigmoid(to_fp32(g))), x)` 直接写出，通用链在 indexed 区域
融合下仍是 1 个 kernel、仍只舍入一次（守卫按 `WIDER_ON_DEVICE` 泛化，不再按算子名）。

| 组 | 算子 | 契约 |
|---|---|---|
| 比较 | lt, le, gt, ge, eq, ne | 同 dtype 同 shape，输出 bool；操作数须为数值（bool 拒绝：真值之间的比较是逻辑运算，本语言不声称具备） |
| 序 | abs, maximum, minimum | 同 dtype 同 shape；**对 int32/int64 合法**（不拓宽到浮点）；maximum/minimum 传播 NaN，与 `tl.maximum`/`torch.maximum` 一致 |
| 初等函数 | sqrt, rsqrt, tanh, sigmoid, softplus, erf, gelu, gelu_tanh | 浮点输入，输出同 dtype；fp32 计算、operator 边界一次舍入（21.6） |
| 选择 | topk_last(x, k) → (values, indices) | 沿最后一轴降序取前 k；**同值取较小列号**（torch argsort 的并列序未定义，本语言不继承未定义行为）；NaN 与 -inf **不入选**（与 `torch.topk` 把 NaN 放首位相反，路由 logit 走 NaN 时不选专家更安全）；候选不足 k 时以 value 0、index 0 补齐。k 为静态字面量（输出维 `[..., k]` 必须可写） |
| 选择 | gather_last(x, index) | `out[..., j] = x[..., index[..., j]]`；末轴的按索引取值，**不是** axis-0 行 gather 的组合（行由行内的值选择）。边界面与 `gather_rows` 一致：负索引加一次 extent，仍越界读作零。访问形状是 gather |
| 索引表 | segment_rank(ids, num_experts, m_max) → (counts, positions) | `positions[i] = ids[i]*m_max + 该专家内排位`（**稳定**：靠前的 slot 拿靠前的位），`ids[i] < 0` 或越出 `[0, num_experts)` 时为 -1；`counts[e]` 为路由到 e 的 slot 数。Task69 的 `atomic_add` 就是这个排位，写成计数扫描后无需原子、无需压缩原语 | ids 为 rank-1 整数；`num_experts` 与 `m_max` 为正整数字面量（输出维与目标空间由它们算出）；slot 数须字面量且 ≤ `num_experts*m_max`（否则同一专家的两个 slot 会撞到同一目标）；访问形状是 block，**任何目标都不会因寻址拒绝它** |
| 时间维递推 | gated_delta_recurrence(q,k,v,g,beta,state) → (output, final_state) | **语言唯一的时序算子**：`[V,K]` 状态驻留寄存器跨整个时间轴，逐 t 执行"衰减 → 经 k 读出 → delta 校正 → 秩一更新 → 经 q 读出"。**为什么是内核种类而不是一个算子行**：五步全是逐元素/归约，语言都有；缺的是循环本身——把状态物化成 `[B,T,H,V,K]` 在真实形状下约 10^10 字节，且第 t 步输出读的是第 t 步**之后**的状态。于是语言只长出一个**声明式递推**，而"DSL 里没有循环"（Design.md 2）的立场不变：循环活在发射出的内核里 | 六个操作数同 dtype 且全字面量（状态瓦片是 constexpr、时间循环由形状定界）；q/k 为 `[B,T,H,K]`、v 为 `[B,T,H,V]`、g/beta 为 `[B,T,H]`、state 为 `[B,H,V,K]`；**要求 HV == H**（GQA 比例需要 `repeat_interleave`，属单独的 `row_repeat` 项）；`acc="fp32"`（跨 t 的累加在 21.6 下必须声明）。任务里的 `scale`、q/k 的 L2 norm 留在 DSL 程序侧——它们是组合，不是递推 |

语义里属于**平台选择而非 IEEE**、因而在此明写的三条：`sqrt(负)` 为 NaN、`rsqrt(0)` 为 +inf
（`math.sqrt` 会抛异常，参考后端用 `_sqrt`/`_rsqrt` 显式给出设备值）；`softplus` 用
`max(x,0)+log(1+exp(-|x|))` 的无溢出形式，对 |x|<17 与 `log(1+exp(x))` 同精度，
对更大值不溢出（torch 在阈值 20 处切到恒等，本式与之差在舍入内，`test_value_layer.py` 钉住）；
`gelu` 是精确 erf 式、`gelu_tanh` 是 tanh 近似，两者不可互换（Task29 要求 `approximate="none"`，
Task71 要求 `"tanh"`）。

这四个算子注册为 `pointwise=False, acc=None`，与 `exp`/`log` 同组：一次表达式、不累加。
**是否融入逐元素区域是另一件需要真机测量的事**（`silu` 融、`exp` 不融），本轮无 GPU 配额，
因此取保守位置并明写，而不是猜。

**本轮的诚实边界**：两个后端的表达式都由 `value_layer.TRITON`/`REFERENCE` 一张表决定，
参考后端已逐一执行验证；Triton 侧 `tl.math.tanh`/`tl.math.erf`（libdevice 绑定，
`tl.tanh` 并不存在）**未在真机上执行过**——2026-10-20 GPU 主机不可达，
发射文本由测试钉住，设备语义待复测。

v0.6（2026-10-22）：分段前缀和 `segmented_cumsum`（Task153 `kda_prologue` 的 reset scan）。

Task152 的 `cumsum` 解决了"沿轴累积"，但每段在 `seg_start` 处重置的分段前缀和当时仍写不出来：
`kda_prologue` 在每个 chunk 内对 gate 做前缀和，chunk 边界即段边界。**它不是新内核，是一条恒等式**：

    y[..., t] = cumsum(x)[..., t] - (cumsum(x)[..., s[t] - 1] if s[t] > 0 else 0)

`s`（`seg_start`）逐位置给出所在段的首元素下标。语言把它注册为一等算子，但**后端不发射它**：
`flowgpu/compiler/ir/composite_ops.py` 在融合与 13.7 缓冲规划之前把节点展开成既有算子——

| 展开步骤 | 复用的 lowering |
|---|---|
| `cumsum(x)` | `scan` 种类（`scan_backend.py`，fp32 累加、边界一次舍入） |
| `max(sub(seg_start, 1), 0)` 作为 gather 索引 | 逐元素原语（`sub`/`maximum`/`zeros_like`） |
| `gather_last(prefix, index)` | `axis_gather` 种类（`topk_backend.py` 的末轴 gather，行 gather 族的边界规则） |
| `sub(prefix, where(eq(seg_start,0), 0, gathered))` | 逐元素原语 |

这样 scan 与 gather 各自只有一份实现、各自的 kernel 种类、平台闸门、精度声明与参考后端，
`segmented_cumsum` 不会长出第二份 cumsum 语义。**恒等式自身的越界索引是 fail-closed 的边界**：
`gather_rows` 族对负索引按"加一次 extent、仍越界读零"处理，若把 `s[t]-1` 直接交给 gather，
`s[t]=0` 会读到 `cumsum[..., -1]`（该行总和）并**静默算错**。展开因此先用 `maximum` 把索引夹到
`[0, N)`、再用 `eq/where` 把首段 lane 的减数置 0：不发出一条依赖 gather 越界约定的访问。

**契约**（类型规则在 `operator.py`，边界写死在此）：

- 轴为末轴（与 `cumsum` 一致；未提供 `_axis` 表面，因为逐位置的 `seg_start` 必须随 x 一起旋转，
  本版不实现——需要别的轴时在程序里 `move_first_last`）；
- 输入浮点、`seg_start` 为 int32/int64 且与 x **同形状**；输出 rank/dtype 与 x 相同；
- 全部 extent 字面量：组件 `cumsum`/`gather_last` 的 tile 与 grid 需要静态维；符号维是 shape 诊断；
- 合法数据：`seg_start[...,0]=0`、沿末轴非降、且 `seg_start[...,t] <= t`；段是相同 start 值的极大连续段，
  因此**长度 0 的段无法写出**（没有位置可以承载它），每个段至少含它的首位置。编译器不读设备数据，
  越界数据是契约违反、结果不声明；它保证的是"访问不越出本行"与"首段减数为 0"；
- 精度（21.6）：`acc="fp32"`，累加发生在 scan 内；scan 结果按声明 dtype 在算子边界舍入一次，
  分段减法再在两个已舍入的前缀值上按声明 dtype 舍入一次——这是组合的边界，不声称"只舍入一次"。

**发射形态（2026-10-22 本地）**：一次 `segmented_cumsum` 目前发 **4 个内核**：`scan`、索引预处理、
`axis_gather`、掩码减法。x 尺度的 pass 是三次（scan 读 x、gather 读 prefix、减法读写 prefix/subtractor），
索引预处理是 int 尺度的一次。`test/test_segmented_cumsum.py` 钉住这一形态。下一步合成方案
（见 `flowgpu/docs/OPERATOR_LOOP.md`）：把索引 clamp/mask 与减法折进一个 gather-correct 发射器
⇒ 2 次发射（scan + correction）；再进一步是 scan 内核内用寄存器级 gather 或 `tl.associative_scan`
的重置形式合成一趟。**状态：已合并、真机待验**（`device_validation='pending'`，2026-10-22：GPU 被
其他租户占满，零利用率准入未通过；复现命令与预期输出见 `flowgpu/docs/OPERATOR_LOOP.md` 第六节）。
比赛用的 Task153 内核**今天仍是手写 Triton**；本算子的价值是语言覆盖度与 4 → 2 → 1 的路线，
不是当前分数，未验证前也不宣称性能。

## 21. 可检查质量与实验

### 21.6 精度

fp32 MatMul 显式指定 IEEE 乘法精度，避免默认 TF32 截断输入。
浮点规约 acc 至少 fp32；store(v) 与 acc(producer) 分离。
`segmented_cumsum` 声明 `acc="fp32"`：它由 `cumsum` 组合而来，累加发生在 scan 内，
scan 的边界舍入与分段减法的 dtype 舍入各一次（见 v0.6 的精度条目）。
`argmax_last`/`argmin_last` 声明 `acc=None`：极值选择是精确比较，没有求和累加；
末轴 tile 升到 fp32 只为比较（fp16/bf16 到 fp32 无损），结果在算子边界按声明 dtype 舍入一次。
融合后的 RMSNorm/Softmax 也在算子边界舍入至声明 dtype；Softmax 的最大值规约
以负无穷排除填充位置，防止大负数、非二次幂末轴的指数全部下溢。
bf16 有 7 个显式尾数位（加隐含位共 8 位有效精度）。低精度误差取决于数据与求和顺序，
不能普遍宣称增长必然是 sqrt(N)。Python 参考规约与 GPU 树形规约允许舍入差异。

### 21.7 稳定性

softmax/RMSNorm 原语具备固定语义。模式检查器识别潜在展开，但现语言没有
exp/div/sum/rsqrt 基础算子，因此当前没有程序能触发候选；测试锁定该能力边界。

### 21.8 可移植性

可移植意味着两个后端都有 lowering。保留算子在前端拒绝。
可移植、编译成功、数值正确、GPU 安全和硬件性能是分别需要证据的结论。

### 21.9 流量仪器

A = T_actual_model / T_interface_floor。
model 按 kernel 的不同外部操作数读一次、物化结果写一次计价；floor 按接口读写计价。
CLI 浮点 A 对符号维度取 1，符号式本身保留。另有 placement_traffic 字段计入 GEMM tile
重读和索引区域中广播的重复读取；元数据输入和任意 gather 仍采用保守接口计费。
两种报告均不预测 cache 命中或实际 HBM 交易。
不能把 PyTorch allocator 峰值当作 HBM 流量，也不能把相比未融合版本的 0.60 比率称为 A。

**访问形状维度（2026-10-20 补）。** A 只数字节，因此两个"字节相同、访问形状不同"的
方案在它眼里完全等价——而实测不是：同一算子、同一字节量，flat 与 per-plane 两种分块在
H100 上差 4.8%–13.5%，一个逐 lane gather 读在国际A 是 1.81×、在昆仑芯是 0.02×。
这是**决策层的表示缺陷**：字节模型被当作已判定来比较方案。
修正是 `analysis/access_shape.py`：从轴身份契约（`ir/index_maps`）、受限视图契约
（`ir/view_ir.ViewSpec`，携带时）与算子身份导出**访问描述子**（kind/连续段长/步长/
是否数据相关及其出处），并规定
**字节代价与访问形状指向不同方案时报 `underdetermined`，不得报成等价**。
描述子是描述，不是设备模型：它不预测 cache、事务数或时间；`underdetermined` 是正确
答案，其解决方式只能是预注册测量，或一个本设计明确不宣称拥有的设备模型。
预备的 `ir/indexing_ir` 索引/散射契约暂不被消费（流水线尚未填充它们），因此描述子不从那里取。

### 21.10 实验协议与验证状态

flowbench_eval.py 对 50 个 canonical 程序实际编译、核对接口、解析生成代码，并把 CPU
输出与不读取 IR/融合计划/后端的 unfused AST 数学解释器对比。它不测 LLM Pass@1。
边缘 verifier 逐一检查 tl.load/tl.store 的 mask 参数并对照数学结果，未执行设备 mask。
GPU harness 先校验数值，再测 CUDA Event 的预热后中位数、allocator 峰值、版本与 seed。
源码生成时间和首设备调用（含 Triton JIT）分别记录。
Live API 使用环境变量密钥，Triton AST 检查与 FlowGPU CPU 语义检查分开标注。
本机 CPU 检验与远端 A800 复测分别留存记录，数值失败的 batch 不产生延迟结论。
历史带宽饱和率、崩溃率与 LLM 成功率缺少可核验原始记录，不作为当前结论。
具体本次设备通过范围、decoder 精度失败及性能结果见 flowgpu/docs/EXPERIMENT_AUDIT.md
与 private A800 evidence。

### 2026-10-04 独立语义扩展

| 表层算子 | 契约 |
|---|---|
| sum_axis(x,a), max_axis(x,a) | 浮点；有符号静态整数 a∈[-rank,rank)，FP32 结果，保留被规约轴为 1 |
| softmax_axis(x,a) | 与末轴 softmax 相同的稳定与舍入契约，沿静态 a，保持 shape/dtype |
| rms_norm_axis(x,w,a) | 沿静态 a；w 为匹配该轴的一维同 dtype 张量，epsilon=1e-6 |

整数字面量只在这些算子的最终 axis 参数位置接受。构图规范化为循环 move_first_last、已有末轴算子及逆旋转；旋转可能物化，未宣称最优布局。
参考后端可显式 emulate_fp32，以逐标量 FP32 舍入帮助分析数值传播；默认高精度不变。该模式不模拟设备树形规约、FMA、TF32 或近似数学指令。协议与实测边界见 flowgpu/docs/AXIS_FP32_SEMANTICS_20261004.md。

v0.7（2026-10-22）：带索引归约 `argmax_last` / `argmin_last`（`reduce_with_index`）。

承载"归约时把获胜位置也带出来"这一结构特征的，是**一个**算子族，不是两个算子：
DSL 没有字符串字面量模式参数，所以方向由名字拼写；`argmax_last` 与 `argmin_last`
共用同一条类型规则、同一份 lowering（`flowgpu/compiler/backend/reduce_with_index_backend.py`）
和同一个 kernel 种类 `reduce_with_index`。审计的分母因此只计 1。

| 算子 | 语义 | 类型规则与契约 |
|---|---|---|
| values, indices = argmax_last(x) | 沿最后一轴取最大值及其**最小索引** | 浮点输入；`values` 为输入 dtype、`indices` 为 int32，shape 均为 `x.shape[:-1]`。**同值取最小索引**：torch 的 `argmax`/`argmin` 不保证并列序，本语言自己声明（全相等行返回 0）。**NaN 永不入选**（与 `topk_last` 的候选规则一致）；整行 NaN 无候选，返回方向单位元（max 为 `-inf`、min 为 `+inf`）与索引 0。**±inf 按普通 max/min 规则参与**（+inf 可成为 argmax、-inf 可成为 argmin），这与 `topk_last` 把 `-inf` 当填充不同。末轴须为**非空字面量**，前导 extent 全字面量，否则静态拒绝；rank-1 输入因结果会退化为标量而被拒绝。`acc=None`：选择比较是精确的，不存在求和累加 |
| values, indices = argmin_last(x) | 沿最后一轴取最小值及其**最小索引** | 同上，方向相反；与 `argmax_last` 是同一结构特征的两个拼写 |

**发射形态**：一次调用发**一个** Triton kernel，每个前导行一个 program。末轴是一条带 mask 的
一维 lane tile，`(best, best_index)` 二元组跨**静态** chunk 循环（`tl.static_range`）保存在寄存器里；
chunk 宽度取平台已有的 `reduction_block_limit`（无记录时退回 `default_block_size`），不按平台名分支。
两个结果各写一次，地址是 `out + row`（标量基址、无 lane 项），写在循环之外。这正好绕开
昆仑芯实测拒绝的三种形态：`store_inside_loop_ok=False`（不在循环内写）、
`gather_load_ok=False`（地址是 affine、非 data-dependent）、
`affine_write_addresses=True`（`out + row` 是 affine）。`test/test_reduce_with_index.py`
用这三条契约字段检查发射文本，并钉住 tie/NaN/±inf/全 NaN 行与单次发射。

v0.8（2026-10-22）：删除专用融合算子 `fused_sigmoid_mul`，改为值层组合。

依据是审计（`flowgpu/docs/LANGUAGE_PRIMITIVE_AUDIT.md`）给出的"删除条件已满足"：
它是 kind-2 绕过中唯一一个替换机制早已存在的条目。删除前先证明了等价，再动手：
2-D 与 3-D 非连续 gate 两种形态都编译成 1 个 kernel，`reshape` 被融合，host 接受
strided 门控，舍入点与内建一致（`enable_fp_fusion=False` 由 primitive 路径与泛化后的
守卫共同保证）。专项测试把等价性、泛化后的守卫（改用 `silu` 钉住）与 3-D 迁移固化。
原语/语言特性比值由 14/2 降到 13/2。

v0.9（2026-10-22）：值层补齐 storage dtype、整数转换规则与 `round` 点算子。

| 表层算子 | 契约 |
|---|---|
| round(x) | 浮点；**ties-to-even**（与 Python `round` 一致），结果停留在输入 dtype，不跨越 dtype 边界 |

存储 dtype 从 int32/int64/bool 扩到**十种**：`int8`、`uint8`、`fp8e4m3`、`fp8e5m2` 加入值层与缓冲/ABI。
两种 FP8 格式都要：e4m3 是 FP8 算子的输出，e5m2 是量化算子的可选输出格式（`fp8_dtype` 参数）。
`fp8e4m3` 是**存储类**（`precision.DTYPE_CLASSES`），参与 `cast_to`、缓冲与 Triton `tl.float8e4nv`
映射，但算术算子拒绝它（须先加宽）——否则 fp8 会被当成可计算类型，舍入点无声消失。
浮点→整数一律**向零截断、按目标位宽回绕**（与 torch `.to(int)` 及硬件 `.to` 一致），不再按目标
位宽逐条拒绝；浮点→fp8e4m3 就近舍入、ties-to-even、在 448 饱和。参考后端对九种 dtype 各自给出
`ROUNDING`，因此"舍入发生在哪个边界"在参考语义里也是可执行的。e5m2 是 IEEE 型格式：5 位指数、2 位尾数、
最大有限值 57344，超出范围按格式规则溢出到 inf 而不是饱和（e4m3fn 无 inf，故在 448 饱和）——
两种格式的舍入规则因此分别声明，而不是共用一条"就近舍入"了事。

审计分母随行重新核定（`flowgpu/docs/LANGUAGE_PRIMITIVE_AUDIT.md`）：kind-2 内建 / 可移植结构特性
由 13/2 经 12/2、11/2 降到 **10/2 = 5.00**，每一次下降都先写完删除条件或伪证理由再改行。

v0.10（2026-10-22）：发射自检默认开启。

`_make_kernel` 在返回前用**Triton 的作用域规则**（`for` 体内的赋值不逃逸）检查内核引用的名字，
默认拒绝内核体里引用未绑定名字（`FLOWGPU_STRICT_EMISSION=0` 仅供 bisect）。这条闸门不是风格检查：
打开后在全量测试里找到 **20 个内核**，全部是**错误内核**而不是误报，且都出自三个结构性缺陷——
封顶二维 matmul 把 kernel 体留在 stride 循环之外、折叠旋转链从不绑定链内的中间值、归约发射器把
逐元素值的最终 sweep 当作逐行 tail 提升到列循环之外。三者都是一处结构性修复而非 20 个特例；
修复后同一形状（共享全 tile 生产者）无需人工改写即编译出结构有效的三个内核（严格闸门判定），
其设备侧数值等价性已生成独立 oracle 的验证负载、尚无卡可用（见
`flowgpu/docs/EMISSION_AND_FP8_DEVICE_VERIFICATION_20261022.md`，状态 PENDING_DEVICE）。
"0 个潜在缺陷"因此是一条可复算的度量，而不是记忆：`tools/audit_emission.py` 扫 `examples/`
的 39 个程序，`tools/verify_emit_fixes_device.py` 生成设备侧独立 oracle（不用编译器参考后端）
的验证负载。三条债的当前值见 `flowgpu/docs/LANGUAGE_PRIMITIVE_AUDIT.md` 的 Tracked debt summary。


v0.11（2026-10-22）：IR 的命名与目标形态 —— **分层纯数据流 SSA**。

这一版不改任何行为，只把一个被混用的名字拆清楚，因为它决定了后面能不能加迭代：
**SSA**、**纯（无副作用）**、**平面（单层图）** 是三件独立的事。

| 性质 | 今天的语言 | 目标形态 | 是否可放弃 |
|---|---|---|---|
| SSA（每值一个定义点、use-def 显式） | 有 | 有 | **不可**：融合判定、值编号、精度边界全靠它 |
| 纯数据流（节点无副作用、无可变状态） | 有 | 有 | **不可**：参考解释器、平台白名单、两条审计比值的前提 |
| 平面（迭代只能藏在 lowering 里，图是单层） | 是 | **否** | 可以，且这正是所有 kind-2 债务的来源 |

所以准确的命名是：**今天是「平面纯数据流 SSA」；加入结构化迭代区域后是「分层纯数据流 SSA」**。
加区域只丢掉"平面"这个词，不丢"纯"，也不丢"SSA"——MLIR 的 `scf.for` 用 block argument 携带
循环值，region 内部仍是 SSA；SISAL 是纯 dataflow + 循环，仍被称作 dataflow。

"分层"这个名字要成立，必须守住四条**可检查**的不变量（写在这里就是为了能被审计工具和测试引用）：

1. 每个节点是纯函数（无内存效果、无原子、无可变状态）；
2. 每个节点有类型契约（shape / dtype / precision），**区域节点也一样**（迭代域、并行方式、carry 契约、精度）；
3. 迭代只以**结构化有界区域**出现；区域在父图里是**一个节点**，融合/分区/平台白名单仍按节点粒度工作；
4. 不做数据相关控制流（无界 `while`、任意 CFG）——那会让图变成流图，前两条的收益同时消失。

边界同样明确：`for`/`while` 这类命令式构造**永不**进入这个语言；将来若加迭代，它是一个
`iterate(domain, init_carry) -> (results, final_carry)` 的 `fold`/`map` 组合子节点，而不是控制流。
升级触发条件（可伪证）：P1（carried-iteration 家族 lowering + `cummin`）与 P2（并行两段式 scan）
落地后，若仍有 ≥3 个目标 op（如 `SelectiveScan`、`LSTM`、`MhcSinkhorn`）在"分数占比可观 +
家族 lowering 无法表达 + 参考语义仍是纯的"三条同时成立，才引入区域；否则先不加。

命名不改度量：两层级形态下，区域节点仍是一个带 kernel kind 和删除条件的节点，
`carried_iteration` 作为 **portable feature** 在有设备证据后进入原语审计的分母。

v0.12（2026-10-24）：值层补整数位运算与位重解释，FP8 舍入可以在内核内完成。

| 算子 | 契约 |
|---|---|
| bitand(x,y) / bitor(x,y) / bitxor(x,y) | 整数类操作数；结果保留 tile 的 dtype；允许一个 rank-0 整数标量（掩码/移位量正是这个形状） |
| shl(x,k) / shr(x,k) | 同上；右移为算术右移，非负中间值上等价于逻辑右移（编码器只对已屏蔽符号位的值右移） |
| float_bits(x) | **fp32 → int32 位重解释**（不是 cast：cast 转换值，它重解释位） |
| bits_float(b) | int32 → fp32 的位重解释 |

它们都是**值层行**（`value_layer.py` 表驱动、四个发射点共用），不新增 kernel kind、不改变原语比值的分子：
它们把"位级舍入"这一类工作从"再加一个专用 lowering"变成"用现有值层表达"。

**动机来自比赛反馈而不是想象**：FlagOS 七张卡里有三张（天数、国际 A、昆仑）**完全编译不了 FP8 转换**，
另外华为的 `bishengir-compile` 也拒绝硬件 fp8 cast。于是 e4m3/e5m2 的舍入与打包改在内核里用整数算术完成
（`tools/fp8_encoder.py`，单-use 表达式树使每个变体只出 **1 个 kernel**）：
`float_bits` 取位 → 掩码/移位取指数与尾数 → 整数加偏置做 RTNE → 拼字节。
规则与参考后端的 `fp8e4m3`/`fp8e5` 一致（e4m3 就近舍入、448 饱和；e5m2 2 位尾数、次正规到 2^-16），
`test/test_fp8_encoder.py` 用按位布局独立写成的 oracle 逐字节校验。

同一天还修掉一个**通用调度缺陷**：融合分组只按"首成员下标"排序，而一个被融合的 pointwise 区域
可以消费在它首成员之后才定义的值——生产者的 kernel 被排到消费者之后，消费者读到未写缓冲
（参考后端直接 `float64('nan')` 崩）。修复是 `fusion.order_by_dependencies` 的**稳定拓扑排序**，
并且**每一个会重建/重排分组的 pass 都调用它**（indexed/reduction/cross-domain/row-gather）。
`test/test_kernel_order.py` 用最小复现钉住：关闭区域融合时旧代码顺序错误、参考实现返回 NaN。

v0.13（2026-10-24）：内核内公共子表达式消除（in-kernel CSE）——把"重算"交给编译器。

**触发它的不是优化想象，而是比赛里的一个真实取舍。** 软件 FP8 编码器需要在同一份
fp32 位模式上反复做同样的取指/移位/掩码；把中间值**共享**给多个消费者，会让融合分区
把它物化成一块缓冲（每份共享值一个 kernel，模块膨胀、流量翻倍），于是编码器被写成
**单-use 表达式树**：每个使用点重新算一遍，保住了"1 个 kernel"，代价是 ~2 倍的 ALU。

这两种代价本来不该由写程序的人承担：**同一个内核里**两个结构相同、操作数相同的表达式，
是**一条**设备指令，不是一次内存边界。`flowgpu/compiler/backend/cse.py` 的开关 +
两个发射点（`indexed_backend` 的 `canonical()` 与 `_pointwise_kernel` 的
`subexpressions`）在发射时做**局部哈希 consing**：

* key =（算子, 操作数寄存器名, 结果 dtype, part, attributes）；操作数名相同 ⇒ 表达式
  相同（递归地由同样的规则保证），于是复用先前发出的寄存器名；
* 只在**一个 kernel 内部**生效；跨 kernel 的共享仍然由融合分区决定是否物化——
  所以 kernel 数、缓冲区计划、分区结果**逐字节不变**（`test/test_cse.py` 钉住）；
* 保留 `cse.ENABLED` 开关，性能二分或 bug 归因不需要改发射器代码。

实测（163 的软件编码器，`tools/fp8_encoder.py`）：

| 指标 | 关闭 CSE | 开启 CSE |
|---|---|---|
| 发射行数（单 kernel） | 194 | **90** |
| `bitcast` 指令 | 24 | **1** |
| `>>` / `&` | 36 / 44 | **13 / 7** |
| generic 模块体积 | 73,739 B | **56,565 B** |

这条设计属于"分层纯数据流 SSA"的**第五个不变量**（前四个见 v0.11）：**每个值在一次
内核发射内只计算一次，跨内核的共享必须显式成为缓冲区**；写程序的人只表达数据流，
"哪里重算、哪里落盘"由编译器决定。

**覆盖面：每一个"一个融合成员一行"的发射器都用同一张表。** 除 pointwise/indexed 外，
还包含 rowwise（分块与两遍流式两种）、matmul、transpose-gather、matmul-transpose、
rope 的 prologue/epilogue（Triton 与 reference 两侧），以及 indexed reference 的元素
函数别名。共享表本身很薄（`cse.Sharing`：`previous` 查表、`record` 记账），真正的
工作量在于**统一语义**：这些发射器的成员是"从同一个索引空间读取的操作数"，只有操作数
寄存器名、dtype、part、attributes 全部一致才允许合并。

这次推广还顺带修掉两个**只在"matmul 融合了带标量操作数的 epilogue"时才触发**的潜伏缺陷：
`_matmul_kernel` 在 `names` 创建之前就 reserved 标量形参（`UnboundLocalError`），
以及标量操作数没有绑定进寄存器表、epilogue 因此发出未定义的 `s_ptr`。两者都有回归测试。

**两条发射路径共用同一个规范键**：`cse.key(op, operands, dtype, part, attributes)`
既被 Triton 的 `indexed_backend` / `_pointwise_kernel` 使用，也被 reference 后端的
`_pointwise_kernel` 与 indexed reference 内核使用（后者把重复值直接别名到同一个元素
函数）。这不是为了 CPU 端的速度——reference 每元素重算只影响它自己——而是为了让
**数值 oracle 与设备内核共享同一套"哪些表达式是同一个值"的判断**：如果 CSE 把两个
不同的表达式错误合并，reference 与设备的差异会出现在平台校验里；如果两条路径对共享
判断不一致，同样会出现。`test/test_cse.py` 用开关 on/off 的 reference 输出逐字节相等
把这件事钉住。

v0.14（2026-10-24）：carried-tuple 扫描——迭代区域 P1 的第一步。

`cumsum` 的扫描状态是一个 fp32 标量；`cummin_last` / `cummax_last` 的状态是
**(value, index) 元组**，因此它们不只是"又一个算子"：它们是语言里第一次出现
"跨迭代携带的结构化状态"。这一步按 `docs/ITERATION_REGION.md` 的 §6 落地：

* 语言/类型：多结果算子，`indices` 为 int64；并列取**最早**索引（合并函数左优先，
  因而结合律成立、可关联扫描）；NaN 出现后值保持 NaN、索引停在第一个 NaN；
* IR/分区：两个结果节点在 `MULTI_RESULT_OPS` 里合成一个 kernel 组；`scan` kind
  从单算子变成 carried 家族；
* reference 后端是数值 oracle（逐行顺序扫描），设备形式先**显式 refusal**
  （`TritonProgram.refusals`），等 torch 探针把 NaN/并列/维度契约钉住后再发射——
  这是"拒绝优于猜测"的同一纪律；
* 审计同步：`scan` 不再是单算子 kind（13→12），precision/primitive 清单与本文的
  算子表都随注册表更新。

下一步（P1 step 2）在本文件与 `docs/ITERATION_REGION.md` 里固定了顺序：torch 语义
探针 → tuple merge 的关联扫描实现 → 打开 refusal 并用 CANN `cummin` 的 20 个用例验收。
区域语法（v0.11 的分层纯数据流 SSA）仍等 P1+P2 完成与 ≥3 个"只能由区域表达"的算子。

v0.15（2026-10-25）：扫描的 prologue/epilogue 融合与行广播加法 `add_broadcast`。

GPU MODE 的 `prefixsum_v2` 把"再省一遍内存往返"变成了编译器需求：两级前缀和的
最后一遍是"扫描结果 + 每行偏移"，如果这一步只能作为独立 kernel，就必须把行内
扫描落成 1 GB 的中间缓冲再读回来。这一版让**扫描自己发射融合的逐点表达式**：

* 融合：`cumsum`/`cumsum_totals` 从 opaque anchor 列表中移除；点向消费者作为
  epilogue、点向生产者作为 prologue 进入 scan group（carried-tuple 扫描仍 opaque，
  等 tuple merge 发射后再放开）；
* 发射：scan 的 Triton/reference lowering 都逐 lane 计算 prologue 表达式、扫描
  carry、再逐 lane 计算 epilogue 表达式后存储；`SERVES`/`REFERENCE_SERVES` 与
  `_serve_scan` 的守卫只允许点向邻居，其它仍按名字拒绝；
* 多结果合并：`coalesce_store_cache_groups` 不再要求"每组一个成员"——运行结果带
  epilogue 时，携带的 totals 兄弟节点仍要并进同一个 kernel，合并后的
  `fused_values` 是组内寄存器而不是缓冲；
* 新算子 `add_broadcast(x, row)`：`row` 与 `x` 前导维相同、尾维为 1，把每行一个值
  加到该行所有元素上。语言的逐点算子**故意不做隐式广播**，所以这是显式的广播加法；
  它的行操作数只在 scan epilogue 里有地址映射，其它 lowering 由 `refusal_for`
  按名字拒绝而不是错算；
* 测试：`test_cumsum.py` 用 reference oracle 逐元素验证融合 epilogue（原"邻居各自
  成 kernel"的测试改成"邻居被吸收进 scan kernel"），审计四件套随新算子同步。

效果（GPU MODE `prefixsum_v2`，H100，2^28 元素）：用 `sum_last` 先算行和、再重算
扫描并把偏移折进 epilogue，内存往返从 4 次降到 3 次，实测 **1.443 ms → 1.074 ms**
（相对最初版 2.564 ms 降低 58.1%，约 H100 峰值带宽的 83%）。下一档 2 次往返需要
decoupled-lookback 式的跨 block 状态通信，属于 `iterate` 区域/原子操作的题目。
