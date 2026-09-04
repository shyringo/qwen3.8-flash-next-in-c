<h1 align="center">Qwen3.8-Flash-Next in C：笔记本 CPU 接近 10 token/s</h1>

<p align="center">
  <strong>只用一颗笔记本 CPU，精确 batch 验证吞吐接近 10 token/s。</strong><br>
  原生 C 语言运行 125B-A6B + 51B PLE：无需 GPU、CUDA、Python、PyTorch、权重转换或其他推理框架。<br>
  可以直接在终端聊天，也可以通过常驻的 OpenAI 兼容接口接入应用。
</p>

<table align="center">
  <tr>
    <td align="center"><strong>9.89 token/s</strong><br>batch-4 精确<br>验证吞吐</td>
    <td align="center"><strong>125B-A6B</strong><br>主模型参数<br>另含 51B PLE</td>
    <td align="center"><strong>5.03 token/s</strong><br>0.199 s/token<br>常驻聊天 TPOT</td>
    <td align="center"><strong>8.99 GiB</strong><br>2,048 上下文<br>实测峰值内存</td>
    <td align="center"><strong>不增加近似误差</strong><br>所有优化路径保持<br>所选 GGUF 的推理结果</td>
  </tr>
</table>

<p align="center">
  <a href="https://github.com/shyringo/qwen3.8-flash-next-in-c/actions/workflows/ci.yml"><img src="https://github.com/shyringo/qwen3.8-flash-next-in-c/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/shyringo/qwen3.8-flash-next-in-c/releases"><img src="https://img.shields.io/github/v/release/shyringo/qwen3.8-flash-next-in-c" alt="Release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/shyringo/qwen3.8-flash-next-in-c" alt="代码许可证"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a><br>
  <a href="#快速开始"><strong>快速开始</strong></a> ·
  <a href="#环境要求">环境要求</a> ·
  <a href="#实测性能">实测性能</a> ·
  <a href="#推理准确性">推理准确性</a> ·
  <a href="#本项目实现的推理优化">推理优化</a>
</p>

## 快速开始

Ubuntu、Debian 或 Windows WSL2：

```bash
sudo apt update
sudo apt install -y build-essential curl git
git clone https://github.com/shyringo/qwen3.8-flash-next-in-c.git
cd qwen3.8-flash-next-in-c
./qwen4.sh
```

macOS 需要先安装命令行开发工具和 OpenMP：

```bash
xcode-select --install
brew install libomp
git clone https://github.com/shyringo/qwen3.8-flash-next-in-c.git
cd qwen3.8-flash-next-in-c
./qwen4.sh
```

启动脚本会自动编译引擎，从 ModelScope 断点续传并校验固定版本的
Unsloth `UD-IQ1_S` 分片 GGUF（共 67.56 GiB），然后进入多轮对话。权重
无需转换。输入 `/reset` 开始新对话，输入 `/exit` 退出。

只推理一次：

```bash
./qwen4.sh --prompt "科技的边界在哪里？" --max-tokens 256
```

选择直接回答或显示思考过程：

```bash
./qwen4.sh --no-thinking
./qwen4.sh --thinking
./qwen4.sh --system "你是人类所需要的唯一入口"
```

让模型常驻，并启动仅限本机访问的 OpenAI 兼容接口：

```bash
./qwen4.sh --server 8080 --no-thinking
```

接口地址是 `http://127.0.0.1:8080/v1`，模型名是
`qwen3.8-flash-next-in-c`，无需 API Key。接口支持流式输出、函数工具、
并行工具调用和工具结果回传，完整说明见[本地 API](docs/QWEN4_API.md)。

无需重新编译即可调整上下文和线程数：

```bash
QWEN4_CONTEXT=16384 ./qwen4.sh
QWEN4_THREADS=8 ./qwen4.sh
```

默认上下文为 8,192 token。超过 2,048 token 后，QSA 会把注意力计算量
限制在固定预算内。内存足够时，最大可以配置到模型原生的 262,144 token。

## 环境要求

- 64 位 POSIX 系统、C 语言编译器、`make` 和 `curl`。Windows 请使用 WSL2。
- 至少预留 75 GB 磁盘空间，用于模型分片和下载过程中的余量。
- 建议至少 12 GB 内存。2,048-token 上下文的实测峰值为 8.99 GiB；配置容量
  每增加一个 token，运行时状态约增加 51 KiB。
- 推荐带 AVX2 的 x86-64 CPU，以使用最快的内核。其他 64 位 POSIX CPU
  可以使用已经测试过的通用标量版本。

WSL2 默认把模型放在 Linux 文件系统里的
`~/.cache/qwen3.8-flash-next-in-c/model`，避免把 67.56 GiB 权重放到较慢的
`/mnt/c` 或 `/mnt/d`。可以用 `QWEN4_MODEL_DIR` 改到其他位置。

当前版本支持文本聊天、文本生成和函数工具，不接收图片或视频输入。

## 实测性能

以下都是实际跑出来的墙钟时间，不是理论估算：

| 测试负载 | TTFT / 总耗时 | TPOT | 生成速度 | 峰值内存 |
|---|---:|---:|---:|---:|
| 常驻接口：23-token 输入，16-token 输出 | **3.731 s** | **0.199 s/token** | **5.03 token/s** | - |
| batch-4 精确验证，4 个位置 | **总计 0.405 s** | - | **9.89 position/s** | **6.55 GiB** |
| 固定 16-position 逐 token 前向 | 总计 3.216 s | - | **4.98 position/s** | - |
| batch-4 prefill，固定 16 positions | 总计 1.789 s | - | **8.94 position/s** | - |
| 16-token 单次推理，2,048 上下文 | - | - | - | **8.99 GiB** |

实测环境：Intel Core i5-1340P 笔记本、32 GB 主机内存、Windows 11、WSL2
Ubuntu 22.04、GCC 11.4、11 个计算线程、WSL2 ext4 上的固定模型分片、
已在内存中的模型页面、`OMP_WAIT_POLICY=ACTIVE` 和贪心推理。供电模式、温度、页面缓存和后台程序都会
明显影响 CPU 实测结果。

接近 10 的数据是一次精确计算 4 个位置时的总吞吐，不是单会话 TPOT。
基准会先逐 token 计算参考结果，再重置模型；只有 4 个 greedy token ID 和最终
248,320 维完整 logits 全部逐字节一致，才会报告速度。该测试使用 256-token
容量和更大的 Q8_0 重排内存；普通聊天仍使用更省内存的默认配置。

一组代表性组件分析中，逐 token 路径每个位置约包含：MoE 0.077 秒、
Attention/GDN 0.087 秒、Hyper-Connection 0.053 秒、输出头 0.010 秒。
温度会改变绝对值。常驻接口的 TTFT 不包含模型下载、
进程启动和模型映射；多轮请求还会精确复用相同的 token 前缀，省去重复的
历史上下文计算，同时保持 logits 不变。

运行固定的可复现实测：

```bash
./scripts/benchmark-qwen4.sh
```

运行带正确性检查的 batch 吞吐基准：

```bash
./scripts/benchmark-qwen4-batch.sh
```

基准测试会在请求运行期间使用 OpenMP active waiting。普通终端聊天和常驻服务
使用 passive waiting，模型空闲时不会让笔记本 CPU 持续满载。

## 本项目实现的推理优化

完整实现分成三类：复用的代码和数据、对公开格式或方案的适配，以及本项目
针对 Qwen3.8-Flash-Next 实现的工程优化。完整边界见
[优化与来源说明](docs/QWEN4_OPTIMIZATIONS.md)。主要项目实现包括：

- **元数据优先的分片 GGUF 加载。** 第一分片只有元数据和 tokenizer 也能
  正常工作；所有 tensor 直接从只读分片映射中定位，无需拼接或转换文件。
- **51B PLE 按需解码。** 每个 token 计算 16 个 n-gram 哈希，只解码
  26.82 GiB PLE 表里真正用到的 IQ4_NL 行。
- **增量 QSA 块索引。** 每 4 个 key 只做一次池化、归一化和 RoPE，用固定
  容量的 513 项堆选择长上下文中的高分块，并始终保留当前尾部。
- **原生四路残差计算。** 低秩门控 Hyper-Connection、分组 RMSNorm 和稳定
  scatter 全部直接用 C 实现。
- **专家去重 MoE 批处理。** 逐 token 解码并行执行 10 个路由专家和 1 个共享
  专家；batch-4 prefill 对多个 token 的专家去重，并行计算 union 后再恢复
  每个 token 原本的路由槽顺序。
- **按布局选择的低比特 SIMD。** 选择性 Q8_0 block-major 重排、跨行 F32
  router 和 `vpshufb` IQ4_NL 查表，在保持累加顺序的同时移除标量解码瓶颈。
- **转置 DeltaNet 状态。** 把相互独立的递归列放进连续 SIMD lanes，decay、
  prediction 和 update 每次处理 8 列。
- **保持递归顺序的 batch-4 prefill。** Hyper-Connection 和 GDN 投影复用权重
  读取，PLE、卷积、DeltaNet 和注意力状态仍严格按 token 顺序推进。
- **跨 token GDN 状态推进。** 每个独立 channel/head 在同一个 OpenMP 团队里
  依次推进 4 个 token，减少数百次短并行区同步，不改变递归更新顺序。
- **全注意力 batch 融合。** Q/K/V/O 权重一次服务 4 个因果位置；先写入 KV，
  再由每个 head 在一个并行区内依次推进 4 个位置。
- **Q8_0 行与 token 双维复用。** 每次 block-major 权重加载同时更新 8 个输出行
  和 4 个 token lane，每个结果使用独立累加器并保持原始 FMA 顺序。
- **量化输入复用与实测预读门控。** 同一份激活只量化一次，供兼容的投影共用；
  即时专家预读经成对热页实测会变慢，因此默认关闭，仅保留实验开关。
- **跨轮次精确前缀复用。** 常驻接口在 prefill 后保存递归状态、PLE、注意力和
  QSA 状态；下一次请求 token 前缀完全相同时，只计算新增消息。
- **原生聊天与函数工具接口。** 有容量边界的 C HTTP/JSON、UTF-8 安全 SSE、
  并行调用、结果回传、健康检查和模型发现，不依赖 Python 服务或外部推理引擎。

模型计算图和推理运行时不是其他推理引擎的套壳。代码与方案边界、上游版本和
许可证均记录在 [NOTICE](NOTICE)。

## 推理准确性

引擎**不会在所选量化 GGUF 之外再增加近似误差**。这里说的是推理运行时的
正确性；当前 1-bit 权重 GGUF 本身仍是原始 BF16/FP8 权重的有损量化。

固定模型和 16-token 输入下，所有经过验证的原生优化路径都保持同一个
248,320 维完整 logits SHA-256：

```text
34db8f9d482429a2a3a10bbe600a59b7ecf6813cf9c3fcd6f356430d19ac3e36
```

固定版本的 CPU-only llama.cpp 参考实现给出完全相同的 top-5 token ID 和
顺序，完整词表的余弦相似度为 0.994556324。逐 token/batch-4 计算、激活复用、
专家 union 并行、低比特 SIMD、转置递归状态、2,048/4,096 上下文状态维护和对话状态保存/恢复均分别
做过验证。详见[推理准确性证据](docs/QWEN4_CORRECTNESS.md)。

```bash
make strict
make portable
make sanitize
```

## 许可证与致谢

仓库代码使用 Apache License 2.0。模型权重使用单独的
[Qwen Community License 1.0](docs/MODEL_LICENSE.md)，下载或使用模型前请先
阅读官方条款。

启动脚本优先从 ModelScope 上的
[Unsloth Qwen3.8-Flash-Next GGUF](https://www.modelscope.cn/models/unsloth/Qwen3.8-Flash-Next-GGUF)
下载固定分片，并提供 Hugging Face 备用地址。官方模型和架构由
[Qwen](https://github.com/QwenLM/Qwen3.8-Flash-Next) 发布。

低比特 CPU 内核和 tokenizer 基础延续自
[qwen3.8-27b-in-c](https://github.com/shyringo/qwen3.8-27b-in-c)；面向磁盘的
MoE 方案借鉴了
[deepseek-v4-flash-0731-in-c](https://github.com/shyringo/deepseek-v4-flash-0731-in-c)，
其 tokenizer 源流致谢
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c)。GGUF/IQ 格式
和码表基于公开的 [ggml](https://github.com/ggml-org/ggml) 工作。完整归属见
[NOTICE](NOTICE)。
