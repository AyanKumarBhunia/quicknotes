# LLM Interview-Eve Rapid Revision

> **Purpose:** A consolidated, previous-day question bank for LLM / Generative-AI interviews at research labs and top product companies.
>
> **Format:** Every answer is intentionally short: definition, equation, tensor shape, trade-off, or memory hook. The goal is recall, not first-time learning.
>
> **Source basis:** The technical core consolidates the supplied CME 295 Lecture 1–9 notes. Items marked **↗ Industry 2026** extend beyond the course using primary papers and official engineering/research material available up to **6 October 2026**.
>
> **Not leaked interview questions:** This is a synthesized study guide covering the concepts, systems judgments, and research habits commonly required for serious LLM roles.

---

## How to use this file

1. On the first pass, answer each bold question aloud before reading the hook.
2. On the second pass, derive every boxed equation and tensor shape on paper.
3. On the final pass, practise the debugging and research-design sections as 60–90 second monologues.

### Two-hour emergency route

```text
25 min  Sections 1–3: LM + attention + architecture
20 min  Sections 4–6: training + distributed + serving
20 min  Sections 7–8: alignment + reasoning
15 min  Sections 9–11: RAG + agents + evaluation/safety
15 min  Sections 13–15: debugging + research + coding prompts
15 min  Sections 16–18: equations, shapes, and final 75 hooks
10 min  Speak the final 60-second answer aloud twice
```

### Priority

- **🔥** Must answer immediately.
- **◆** Common systems/design follow-up.
- **↗ Industry 2026** Particularly relevant to current frontier-model and agent work.

### Notation

```text
B       batch size
T       sequence length
V       vocabulary size
D       model width
Hq      query-head count
Hkv     key/value-head count
Dh      per-head width
L       number of Transformer layers
Dff     FFN hidden width
E       number of MoE experts
k       selected experts per token
P       parameter count
Ntok    training-token count
```

---

# 1. LLM foundations, tokenization, and objectives

1. **🔥 What is a language model?** — A model of sequence probability, usually factorized as next-token conditionals. *Hook: prefix → next-token distribution.*

1. **What is the causal factorization?**
$$
   p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
$$
   *Hook: sequence probability is a product of prefix-conditioned token probabilities.*

1. **What makes a modern LLM “large”?** — Parameters, training tokens, and compute; there is no universal parameter threshold.

1. **Why are most generative LLMs decoder-only?** — One unified causal objective supports open-ended text generation and in-context learning.

1. **Encoder-only vs encoder–decoder vs decoder-only?** — Encoder-only represents; encoder–decoder transforms; decoder-only autoregressively generates.

1. **Why tokenize text?** — Neural networks consume discrete IDs/vectors, not raw strings.

1. **Word vs subword vs byte/character tokenization?** — Larger units shorten sequences; smaller units improve coverage and robustness.

1. **Why is subword tokenization the usual compromise?** — It shares morphology while avoiding character-length sequences and most OOV failures.

1. **What is BPE?** — Repeatedly merge frequent adjacent symbol pairs. *Hook: frequency-driven bottom-up vocabulary construction.*

1. **What is a unigram tokenizer?** — Start with many candidate pieces and prune toward a probabilistic vocabulary. *Hook: choose the best segmentation under a token model.*

1. **Why use byte-level tokenization?** — Any string is representable without an unknown token; the cost is sometimes longer, less semantic sequences.

1. **Why does tokenizer choice affect model cost?** — Compute, KV cache, and billing scale with token count rather than human-visible word count.

1. **What is token fertility?** — The average number of tokens needed per word or unit of text; high fertility hurts latency and context efficiency.

1. **Why can multilingual tokenization be unfair?** — Some languages fragment into more tokens, raising cost and shortening effective context.

1. **What is an embedding lookup?**
$$
   E\in\mathbb{R}^{V\times D},\qquad X=E[\text{token\_ids}]\in\mathbb{R}^{B\times T\times D}.
$$
   *Hook: indexing is efficient one-hot multiplication.*

1. **Why is one-hot encoding not semantic?** — Distinct tokens are orthogonal, so every different pair looks equally unrelated.

1. **Static vs contextual embeddings?** — Static: one vector per token type; contextual: the vector depends on the current sequence.

1. **What is weight tying?** — Share the input embedding matrix with the output vocabulary projection, often `W_out = E^T`.

1. **What are logits?** — Unnormalized vocabulary scores; softmax converts them into probabilities.

1. **What is token-level cross-entropy?**
$$
   \mathcal L=-\sum_{v=1}^{V}y_v\log p_v=-\log p(y_{true}).
$$
   *Hook: punish low probability on the correct token.*

1. **What is the softmax-cross-entropy gradient?**
$$
   \frac{\partial \mathcal L}{\partial z}=p-y.
$$
   *Hook: predicted distribution minus target distribution.*

1. **What is perplexity?**
$$
   \mathrm{PPL}=\exp\!\left(-\frac1T\sum_t\log p(x_t\mid x_{<t})\right).
$$
   Lower means less surprise; it is not identical to usefulness or reasoning quality.

1. **Why is perplexity hard to compare across tokenizers?** — Different token units change sequence lengths and per-token likelihood scales.

1. **What is teacher forcing?** — During training, condition each prediction on ground-truth previous tokens.

1. **Why is training parallel but generation sequential?** — Training knows all shifted targets and uses a causal mask; inference does not know future tokens.

1. **What is exposure bias?** — Training sees correct prefixes, while inference increasingly conditions on its own mistakes.

1. **What is label smoothing?** — Replace a one-hot target with a slightly distributed target to discourage extreme confidence.

1. **What is sequence packing?** — Concatenate shorter samples into full training sequences to reduce padding waste while preserving boundaries/masks.

1. **What is loss masking in chat SFT?** — Compute loss only on assistant/output tokens, not system or user tokens.

1. **Base model vs instruct model?** — Base model predicts continuations; instruct model has post-training to follow user intent and product policies.

---

# 2. Attention and Transformer mechanics

1. **🔥 What do Q, K, and V mean?** — Q asks, K matches, V supplies.

1. **How are Q, K, and V produced?**
$$
   Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

1. **🔥 State scaled dot-product attention.**
$$
   \operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{D_h}}+M\right)V.
$$
   *Hook: compare, scale, mask, normalize, retrieve.*

1. **🔥 Derive the core attention shapes.**
   ```text
   Q, K, V:        [B, H, T, Dh]
   QK^T:           [B, H, T, T]
   attention @ V:  [B, H, T, Dh]
   concat heads:   [B, T, D]
   ```

1. **What does one row of an attention map mean?** — The distribution of one query position over all key positions.

1. **Why divide by `sqrt(Dh)`?** — Dot-product variance grows with head width; scaling prevents softmax saturation.

1. **What dimension does attention softmax use?** — The key-position dimension, so each query’s weights sum to one.

1. **Why separate K and V?** — Match on one representation, retrieve another; addresses and contents need not be identical.

1. **What is self-attention?** — Q, K, and V come from the same sequence.

1. **What is cross-attention?** — Q comes from the decoder/current stream; K/V come from another encoded modality or source.

1. **What is causal masking?** — Set future logits to `-∞`, giving zero probability after softmax.

1. **Padding mask vs causal mask?** — Padding mask hides fake tokens; causal mask hides future real tokens.

1. **What is multi-head attention?** — Run several learned Q/K/V matching spaces in parallel, concatenate, then project.

1. **Are heads forced to specialize?** — No; separate parameters permit specialization, but the vanilla loss does not guarantee it.

1. **Why does attention need position information?** — Content interactions alone do not encode order or distance.

1. **Absolute positional embedding?** — Add a learned position vector to each token embedding; simple but tied to a trained range.

1. **Sinusoidal positional encoding?** — Fixed multi-frequency sine/cosine features; relative offsets appear through phase differences.

1. **Relative-position bias?** — Add a learned or deterministic distance-dependent term directly to attention logits.

1. **🔥 What is RoPE?** — Rotate Q/K feature pairs by position-dependent angles so their dot product depends on relative offset.

1. **Why does RoPE become relative?**
$$
   (R_mq)^\top(R_nk)=q^\top R_{n-m}k.
$$
   *Hook: absolute rotations cancel into a relative rotation.*

1. **Which tensors does RoPE rotate?** — Q and K, not V; shapes stay unchanged.

1. **What is the long-context problem with RoPE?** — Extrapolating far beyond trained positions can distort phases and attention geometry.

1. **What is RoPE scaling/interpolation?** — Modify position frequencies or indices to extend context, trading extrapolation range against positional precision.

1. **What does the Transformer FFN do?** — Attention mixes tokens; the FFN transforms features independently at each position.

1. **What is a two-layer FFN?**
$$
   \mathrm{FFN}(x)=W_2\,\phi(W_1x+b_1)+b_2,
$$
   with `[B,T,D] → [B,T,Dff] → [B,T,D]`.

1. **What is SwiGLU?** — A gated FFN, roughly `SiLU(xW_g) ⊙ (xW_u)` followed by a down projection.

1. **Why are gated FFNs popular?** — They add multiplicative feature selection and often improve quality per parameter/compute.

1. **What is a residual connection?**
$$
   y=x+F(x).
$$
   *Hook: preserve an identity path for information and gradients.*

1. **LayerNorm vs RMSNorm?** — LayerNorm centers and scales; RMSNorm scales by root-mean-square without subtracting the mean.

1. **Pre-Norm vs Post-Norm?** — Pre-Norm: `x + F(norm(x))`; Post-Norm: `norm(x + F(x))`. Pre-Norm usually trains deep stacks more stably.

1. **Why can residual-stream scale still become a problem?** — Repeated additions can grow activations; initialization, normalization, and residual scaling control it.

1. **What is attention complexity?** — Dense attention is roughly `O(T²D)` compute and `O(T²)` score memory.

1. **Why is long context expensive even with FlashAttention?** — FlashAttention removes materialized score-memory waste, but exact dense attention still performs quadratic interactions.

1. **Sliding-window attention?** — Each token attends only within a local window; compute becomes roughly `O(Tw)` but global information paths lengthen.

1. **Sparse attention?** — Compute only selected token pairs; success depends on whether the sparsity pattern preserves needed dependencies.

1. **Linear attention?** — Rearrange or approximate the attention kernel to avoid a full `T×T` matrix; efficiency may cost expressivity/stability.

1. **Attention vs SSM recurrence?** — Attention retrieves explicitly from stored tokens; SSMs compress history into a recurrent state.

1. **What is gradient checkpointing?** — Store fewer activations and recompute them in backward; trade compute for memory.

1. **What is activation recomputation’s memory hook?** — *Forget forward intermediates now; pay to regenerate them later.*

---

# 3. Modern architecture choices: MHA, GQA, MLA, MoE, and hybrids

1. **🔥 MHA vs GQA vs MQA?**
   ```text
   MHA: Hq query heads, Hq KV heads
   GQA: Hq query heads, fewer Hkv KV heads
   MQA: Hq query heads, one KV head
   ```
   *Hook: keep many ways to ask; share memory representations.*

1. **Why does GQA help decoding?** — KV-cache size and bandwidth scale with `Hkv`, not `Hq`.

1. **What quality trade-off does MQA make?** — Maximum KV sharing saves memory but may reduce representational diversity relative to GQA/MHA.

1. **What is Multi-Latent Attention at a high level?** — Compress K/V information into a lower-dimensional latent before caching, then reconstruct/use it during attention.

1. **GQA vs MLA?** — GQA reduces the number of KV heads; MLA compresses the KV representation itself.

1. **What is sparse Mixture of Experts?** — A router sends each token through only top-`k` FFN experts.

1. **Where is MoE usually inserted?** — In the FFN sublayer, because FFNs contain a large fraction of parameters and per-token compute.

1. **What is the MoE equation?**
$$
   y(x)=\sum_{e\in\operatorname{TopK}(g(x))}\alpha_e(x)E_e(x).
$$

1. **Total vs active MoE parameters?** — Total counts every expert; active counts only experts used for a token’s forward pass.

1. **Does MoE reduce model storage?** — No; it raises total capacity while limiting active compute.

1. **What is routing collapse?** — A few experts receive most tokens while others are underused.

1. **How is routing collapse mitigated?** — Load-balancing losses, noisy routing, capacity controls, and better router design.

1. **What is expert capacity factor?** — Extra token slots allocated per expert relative to perfectly balanced routing.

1. **What happens when expert capacity is exceeded?** — Tokens may be dropped, rerouted, or processed by an overflow/shared path; all choices affect quality and throughput.

1. **What is expert parallelism?** — Place different experts on different devices and all-to-all dispatch routed tokens.

1. **Why can MoE be communication-bound?** — Token all-to-all traffic can dominate expert matmul savings.

1. **What is an always-on shared expert?** — A dense expert applied to every token alongside routed experts, capturing common knowledge.

1. **What is auxiliary-loss-free load balancing?** — Adjust routing biases or control traffic without adding a potentially quality-distorting balancing term to the main loss.

1. **What is multi-token prediction?** — Train auxiliary heads to predict several future tokens, adding denser supervision and enabling speculative-style acceleration.

1. **Dense vs MoE model selection?** — Dense is simpler and often better at small scale; MoE offers more capacity per active FLOP but adds routing/communication complexity.

1. **↗ Industry 2026: What is a hybrid attention–SSM model?** — Mix exact attention layers for retrieval with recurrent/state-space layers for cheaper long-sequence processing.

1. **What is Mamba’s mental model?** — A selective state-space recurrence whose dynamics depend on the input; linear-time sequence processing without a KV cache like attention.

1. **Why have Transformers remained dominant despite SSM efficiency?** — Mature scaling, hardware kernels, parallel training, retrieval quality, and ecosystem support.

1. **What is the fair way to compare a new architecture with a Transformer?** — Match data, compute, parameters/active FLOPs, tokenizer, optimizer, and evaluation protocol.

---

# 4. Pre-training data, scaling laws, and optimization

1. **🔥 What is pre-training?** — Large-scale next-token learning on broad text/code to build general capability.

1. **What is mid-training?** — Continue the LM objective on a smaller, higher-quality or capability-targeted data mixture before behavioral post-training.

1. **Pre-training vs SFT?** — Pre-training learns broad distributions; SFT teaches desired interaction patterns.

1. **Why is data mixture as important as token count?** — It controls which languages, domains, skills, and biases the model sees.

1. **What is deduplication for?** — Prevent memorization, benchmark leakage, and overweighting repeated content.

1. **Exact vs near deduplication?** — Exact removes identical records; near dedup uses similarity hashes/embeddings to remove variants.

1. **What is benchmark contamination?** — Evaluation examples or close variants appear in training, inflating measured generalization.

1. **How do you detect contamination?** — Hash/n-gram matching, semantic similarity, provenance audits, and performance on newly created private sets.

1. **What is data curriculum?** — Control sample ordering or mixture over training so the model sees easier/general data before harder/specialized data.

1. **What is domain upsampling?** — Increase the sampling probability of high-value but scarce domains such as code or math.

1. **What is the risk of too much upsampling?** — Overfitting, catastrophic capability trade-offs, and distorted language distribution.

1. **What is synthetic-data model collapse?** — Repeated training on narrow model-generated distributions erodes diversity and tails.

1. **When is synthetic data useful?** — When generated with strong teachers/verifiers, filtered for quality/diversity, and mixed with real data.

1. **🔥 Rough dense-training compute estimate?**
$$
   C_{train}\approx 6PN_{tok}.
$$
   *Hook: forward ≈ `2P`, backward ≈ `4P` operations per token.*

1. **What do scaling laws say?** — Loss often follows smooth power-law trends with model size, data, and compute over studied regimes.

1. **What is compute-optimal scaling?** — Allocate fixed compute between model size and training tokens rather than maximizing either alone.

1. **What is the Chinchilla memory hook?** — Roughly `Ntok ≈ 20P` in that study; treat it as an empirical regime, not a universal law.

1. **What does undertrained mean?** — Too many parameters for the number of tokens consumed under the chosen compute budget.

1. **Sample efficiency vs compute efficiency?** — Sample efficiency is quality per token; compute efficiency is quality per FLOP/time/cost.

1. **What is AdamW?** — Adam’s adaptive moments with decoupled weight decay.

1. **Why use learning-rate warmup?** — Early optimizer statistics and activations are unstable; ramping LR avoids destructive first updates.

1. **Why decay the learning rate?** — Large early steps explore; smaller late steps refine and stabilize.

1. **What is gradient clipping?** — Cap gradient norm/value to limit destructive updates, especially around spikes.

1. **Why exclude some parameters from weight decay?** — Biases and normalization scales often should not be shrunk like large weight matrices.

1. **What is gradient accumulation?** — Sum gradients across microbatches before an optimizer step to emulate a larger batch.

1. **Global batch size?** — `microbatch × accumulation steps × data-parallel workers`.

1. **What is the large-batch trade-off?** — Better hardware utilization/lower gradient noise, but fewer optimizer updates and possible generalization/optimization changes.

1. **What is token-based scheduling?** — Define training progress by processed tokens rather than steps, making runs comparable across batch changes.

1. **What should a scaling experiment control?** — Architecture, tokenizer, data quality/mixture, optimizer, training tokens, and evaluation contamination.

1. **Why can a lower training loss fail to improve product quality?** — Likelihood is a proxy; it may not track instruction following, factuality, or safety.

---

# 5. Numerical precision, memory, and distributed training

1. **FP32 vs FP16 vs BF16?** — FP16 has narrower exponent range; BF16 keeps FP32-like exponent range and is usually easier for LLM training.

1. **Why does FP16 need loss scaling?** — Small gradients can underflow; scaling moves them into representable range before unscaling.

1. **What is mixed-precision training?** — Use low precision for most compute/storage while retaining selected states/reductions in higher precision.

1. **What is FP8 training?** — Use 8-bit floating formats with scaling and careful accumulation to accelerate supported kernels.

1. **Quantization vs mixed precision?** — Quantization deliberately maps values to a lower-bit representation; mixed precision assigns different numerical formats to operations/states.

1. **What occupies training memory?** — Weights, gradients, optimizer states, activations, temporary buffers, and communication workspace.

1. **Why can Adam states dominate parameter memory?** — It stores first and second moments, often in higher precision, in addition to weights and gradients.

1. **What is data parallelism?** — Replicate the model, split data, then aggregate gradients.

1. **What communication dominates data parallelism?** — Gradient all-reduce or reduce-scatter/all-gather variants.

1. **What is tensor parallelism?** — Split large matrix operations across devices within each layer.

1. **What is pipeline parallelism?** — Place layer stages on different devices and stream microbatches through the pipeline.

1. **What is the pipeline bubble?** — Device idle time while filling/draining the pipeline or waiting for dependencies.

1. **What is sequence/context parallelism?** — Split token/sequence dimensions across devices to reduce activation or long-context memory.

1. **What is 3D parallelism?** — Combine data, tensor, and pipeline parallelism; MoE may add an expert-parallel dimension.

1. **What is ZeRO/FSDP?** — Shard optimizer states, gradients, and eventually parameters across data-parallel workers.

1. **ZeRO-1/2/3 memory hook?** — Shard optimizer states → plus gradients → plus parameters.

1. **Why is communication overlap important?** — Hide collective latency behind useful forward/backward compute.

1. **What is a straggler?** — A slower worker that forces synchronized peers to wait.

1. **What causes distributed training hangs?** — Collective mismatch, rank failure, timeout, network issue, or divergent control flow.

1. **What is checkpoint sharding?** — Store model/optimizer state in partitions so no single rank must materialize everything.

1. **Why use activation checkpointing selectively?** — Recompute cheap/high-memory layers; avoid recomputing already expensive or communication-heavy paths.

1. **What is MFU?** — Model FLOPs utilization: achieved model arithmetic divided by hardware peak, a proxy for system efficiency.

1. **Why is peak FLOP/s not enough?** — Memory traffic, communication, kernel launch overhead, and imbalance reduce sustained throughput.

1. **What is the roofline mental model?** — Performance is bounded by compute throughput or memory bandwidth depending on arithmetic intensity.

---

# 6. Inference and serving systems

1. **🔥 Prefill vs decode?** — Prefill processes the whole prompt in parallel; decode produces one new token per sequence step.

1. **Which phase is usually compute-bound?** — Prefill has large matrix multiplications and high arithmetic intensity.

1. **Which phase is usually memory-bandwidth-bound?** — Decode repeatedly reads weights and KV cache for relatively little work per token.

1. **What is TTFT?** — Time to first token: queueing + prompt/prefill + first decode work.

1. **What is TPOT/inter-token latency?** — Time between generated tokens during decode.

1. **Latency vs throughput?** — Latency measures one request’s delay; throughput measures aggregate tokens/requests per unit time.

1. **What is goodput?** — Throughput that also satisfies service-level objectives such as latency limits.

1. **🔥 What is a KV cache?** — Store past attention K/V tensors so each decode step computes only the new token’s projections.

1. **KV-cache shape per layer?**
   ```text
   K, V: [B, Hkv, T, Dh]
   ```

1. **Approximate KV-cache bytes?**
$$
   2\,L\,B\,T\,H_{kv}\,D_h\times \text{bytes per element}.
$$
   *Hook: K and V × layers × tokens × KV width.*

1. **Why can GQA improve throughput?** — Smaller KV cache reduces HBM capacity and bandwidth pressure.

1. **What is prefix/prompt caching?** — Reuse KV states for an identical prompt prefix across requests.

1. **Prompt caching vs ordinary KV caching?** — KV caching reuses past tokens within one sequence; prefix caching shares a repeated prefix across requests/sessions.

1. **What is continuous/in-flight batching?** — Add and remove requests at token-step boundaries rather than waiting for a static batch to finish.

1. **Why does static batching waste capacity?** — Sequences finish at different times, leaving padded/idle slots.

1. **What is PagedAttention?** — Manage KV cache in fixed-size non-contiguous blocks, like virtual memory pages, reducing fragmentation and enabling sharing.

1. **Internal vs external KV fragmentation?** — Internal: unused slots inside allocated blocks; external: free memory split into unusable gaps.

1. **What is chunked prefill?** — Split long prompt processing into chunks so prefill does not monopolize the GPU and starve decode traffic.

1. **What is prefill–decode disaggregation?** — Serve prompt ingestion and token generation on different worker pools optimized for their distinct bottlenecks.

1. **Why might disaggregation help?** — Isolate compute-heavy prefill from bandwidth/latency-sensitive decode and scale each independently.

1. **What is speculative decoding?** — A cheap draft proposes tokens; the target model verifies several in parallel while preserving the target distribution.

1. **Why is speculative decoding exact?** — An acceptance/correction rule ensures output samples match the target model distribution.

1. **When does speculative decoding help most?** — High draft acceptance, cheap draft model, sufficiently large verification blocks, and decode-bound workloads.

1. **What is multi-token prediction at serving time?** — Predict candidate future token blocks to reduce sequential target-model calls, often with verification.

1. **What is quantized inference?** — Store/compute weights, activations, or KV cache at lower precision to reduce memory and increase throughput.

1. **Weight-only vs W8A8 quantization?** — Weight-only compresses weights while activations stay higher precision; W8A8 quantizes both.

1. **What is KV-cache quantization?** — Lower K/V precision to extend context/concurrency and reduce bandwidth, with possible attention-quality loss.

1. **Post-training quantization vs quantization-aware training?** — PTQ calibrates an existing model; QAT trains while simulating quantization error.

1. **What are AWQ/GPTQ/SmoothQuant memory hooks?** — AWQ protects salient weights; GPTQ minimizes layerwise reconstruction error; SmoothQuant shifts activation outliers into weights.

1. **What is model parallel inference?** — Shard one model across devices because weights/KV/compute do not fit or meet latency on one accelerator.

1. **Why can tensor parallelism hurt decode latency?** — Every layer may require cross-device collectives for each generated token.

1. **What is expert-parallel serving?** — Distribute MoE experts across devices and route tokens to expert owners.

1. **↗ Industry 2026: What serving optimizations are now first-class?** — Paged KV, in-flight batching, FP8/FP4, disaggregated serving, expert parallelism, speculative decoding, and multi-token prediction.

1. **How do you choose a serving configuration?** — Optimize a Pareto frontier over quality, TTFT, TPOT, throughput, memory, reliability, and cost.

---
# 7. SFT, LoRA/QLoRA, preference tuning, and RLHF

1. **🔥 What is supervised fine-tuning?** — Train a pre-trained LM on curated instruction–response examples using next-token loss on desired outputs.

1. **What does an instruction-tuning sample look like?** — System/context + user instruction + assistant response, serialized with role/control tokens.

1. **Why mask prompt tokens in SFT loss?** — The prompt is conditioning context; the desired supervision is the assistant behavior.

1. **What makes SFT data high quality?** — Correctness, clear instructions, representative prompt coverage, consistent style, diversity, and low contamination.

1. **What is catastrophic forgetting in SFT?** — Narrow fine-tuning damages broad capabilities learned during pre-training.

1. **How do you reduce SFT forgetting?** — Lower LR, mix general data, regularize toward the base model, use adapters, and evaluate broad regressions.

1. **What is LoRA?** — Freeze `W₀` and learn a low-rank update:
$$
   W=W_0+\frac{\alpha}{r}BA,
   \quad A\in\mathbb{R}^{r\times d_{in}},\ B\in\mathbb{R}^{d_{out}\times r}.
$$

1. **Why does LoRA save memory?** — Optimizer states and gradients exist only for small adapter matrices.

1. **What does LoRA rank control?** — Update capacity; larger rank is more expressive but costs more parameters/compute.

1. **What is LoRA alpha?** — A scale for the low-rank update, commonly used as `alpha/r`.

1. **Where is LoRA commonly applied?** — Attention projections and sometimes FFN matrices; target choice is empirical.

1. **What is QLoRA?** — Keep the frozen base model quantized, dequantize for compute, and train higher-precision LoRA adapters.

1. **Why can QLoRA train a large model on less memory?** — Base weights use about 4 bits, while only adapters and selected states need training precision.

1. **What is NF4?** — A 4-bit quantization format designed around approximately normally distributed neural weights.

1. **What is double quantization?** — Quantize the quantization-scale metadata itself to save more memory.

1. **SFT vs preference tuning?** — SFT imitates a target response; preference tuning raises one response relative to another.

1. **What is a preference pair?** — `(prompt, chosen response, rejected response)`.

1. **Why are pairwise labels attractive?** — Humans often compare two answers more consistently than assigning absolute scores.

1. **Pointwise vs pairwise vs listwise feedback?** — Score one, compare two, or rank many.

1. **What is a reward model?** — A model mapping `(prompt, response)` to one scalar preference score.

1. **🔥 Bradley–Terry preference probability?**
$$
   P(y_w\succ y_l\mid x)=\sigma(r(x,y_w)-r(x,y_l)).
$$

1. **Reward-model loss?**
$$
   \mathcal L_{RM}=-\log\sigma(r_w-r_l).
$$
   *Hook: score the winner above the loser.*

1. **Why is reward-model training pairwise but inference pointwise?** — Pairwise differences train the scale ordering; deployment needs one scalar per candidate.

1. **What is RLHF?** — Train a preference reward model, then optimize the policy against it while constraining drift.

1. **LLM RL mapping?** — State = prompt + prefix; action = next token; policy = next-token distribution; rollout = completion; reward = sequence score.

1. **Reward vs value?** — Reward evaluates an observed outcome; value predicts expected future return from a partial state.

1. **What is advantage?**
$$
   A_t\approx R_t-V(s_t).
$$
   *Hook: better or worse than expected.*

1. **What is `pi_old`?** — The policy snapshot that generated the PPO batch; it stabilizes each update.

1. **What is `pi_ref`?** — A frozen SFT/reference model anchoring overall behavior.

1. **Why are `pi_old` and `pi_ref` different?** — `pi_old` limits one update step; `pi_ref` limits long-run drift from the aligned base.

1. **PPO probability ratio?**
$$
   \rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}.
$$

1. **PPO clipped objective?**
$$
   \min\big(\rho_tA_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\big).
$$
   *Hook: improve the action, but cap how much one batch can move the policy.*

1. **Why use a KL penalty to the reference model?** — Prevent reward exploitation, language degradation, and excessive distribution shift.

1. **What is reward hacking?** — The policy exploits flaws in the learned/proxy reward without satisfying the real objective.

1. **How do you detect reward hacking?** — Compare reward with human judgments, inspect high-reward samples, use held-out judges, and monitor KL/diversity.

1. **What is RLAIF?** — Preference or critique labels are generated by an AI evaluator rather than only human raters.

1. **What is Constitutional AI’s high-level idea?** — Generate critiques/revisions and preference signals using explicit principles, then train on them.

1. **🔥 What is DPO?** — Directly train the policy on chosen/rejected pairs without an explicit reward model or online PPO loop.

1. **DPO loss?**
$$
   \mathcal L_{DPO}=-\log\sigma\!\left(\beta\left[
   \log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}-
   \log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
   \right]\right).
$$

1. **Role of DPO beta?** — Controls preference sharpness/reference regularization; larger beta changes how aggressively policy ratios separate.

1. **PPO vs DPO?** — PPO is online RL with rollouts/rewards; DPO is offline supervised preference optimization.

1. **DPO’s main weakness?** — It is limited by a static preference dataset and can overfit label biases or lack on-policy coverage.

1. **What is IPO conceptually?** — A preference objective designed to avoid DPO’s strongly saturating logistic behavior by targeting a margin.

1. **What is KTO conceptually?** — Learn from desirable/undesirable examples without requiring paired responses, motivated by asymmetric human utility.

1. **What is ORPO conceptually?** — Combine SFT and an odds-ratio preference term without a separate frozen reference model.

1. **Why should you not memorize every preference acronym?** — Know the axes: paired vs unpaired, online vs offline, explicit reference, explicit reward model, and KL control.

---

# 8. Reasoning models, verifiers, GRPO, and test-time compute

1. **What is a reasoning model operationally?** — A language model trained or prompted to spend intermediate tokens/compute before a final answer.

1. **Why can chain-of-thought help?** — It decomposes one hard prediction into easier local steps and allocates more inference compute.

1. **Why are math and coding useful reasoning domains?** — Final answers or programs can often be mechanically verified.

1. **What is RLVR?** — Reinforcement learning from verifiable rewards: use deterministic checks rather than only learned preference rewards.

1. **Outcome reward vs process reward?** — ORM scores final success; PRM scores intermediate reasoning steps.

1. **PRM advantage?** — Denser credit assignment and earlier error localization.

1. **PRM risk?** — Expensive/noisy step labels and possible overoptimization of a particular reasoning style.

1. **Why can a correct final answer still hide bad reasoning?** — Outcome verification accepts lucky guesses or compensating errors.

1. **What is best-of-N?** — Sample `N` completions, score/verify them, return the best.

1. **What is self-consistency?** — Sample multiple reasoning paths and choose the most common final answer.

1. **What is pass@k?** — Probability that at least one of `k` attempts succeeds.

1. **Unbiased pass@k estimator from `n` samples with `c` successes?**
$$
   \widehat{pass@k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}.
$$

1. **Pass@k vs pass^k?** — At least one succeeds vs all `k` repeated attempts succeed; opportunity vs reliability.

1. **Why does sampling temperature matter for pass@k?** — Too low gives duplicate attempts; too high destroys per-sample quality.

1. **What is test-time compute?** — Extra sampling, revision, search, verification, or reasoning tokens used after training.

1. **↗ Industry 2026: What is compute-adaptive reasoning?** — Allocate deeper reasoning only to prompts where expected quality gain justifies latency/cost.

1. **Proposer–verifier mental model?** — A generator creates candidates; a verifier/ranker selects or guides better ones.

1. **Why can a weak verifier hurt search?** — Search amplifies verifier errors, selecting reward hacks rather than genuinely correct outputs.

1. **What is tree/search-based reasoning?** — Expand candidate intermediate states, score them, and allocate compute to promising branches.

1. **Search vs self-refinement?** — Search explores alternatives; refinement repeatedly edits one trajectory.

1. **What is a thinking budget?** — A limit on reasoning tokens, samples, search depth, time, or compute.

1. **Why is “more reasoning” not always better?** — Accuracy can plateau while latency, cost, and wandering continue to grow.

1. **What is budget forcing?** — Prompt/control the model to continue or terminate reasoning at a desired compute budget.

1. **What is GRPO?** — PPO-like policy optimization using same-prompt group reward statistics instead of a learned value model.

1. **Group-relative advantage?**
$$
   A_i=\frac{r_i-\operatorname{mean}(r_{1:G})}{\operatorname{std}(r_{1:G})+\epsilon}.
$$

1. **GRPO vs PPO?** — GRPO removes the critic/value model but needs multiple completions per prompt.

1. **When does GRPO struggle?** — All rewards identical, small/noisy groups, weak exploration, length bias, or unreliable verifiers.

1. **What is the zero-variance group problem?** — If every completion gets the same reward, normalized advantages provide no learning signal.

1. **What is GRPO length bias?** — Per-response length normalization can weight tokens differently across short/long outputs, unintentionally favoring verbosity.

1. **What is reward standardization bias?** — Group variance changes gradient scale; easy/hard prompts can be weighted unintentionally.

1. **What is asymmetric clipping intuition?** — Permit more recovery for very low-probability useful tokens without allowing high-probability tokens to collapse abruptly.

1. **Reasoning distillation?** — A strong teacher generates verified reasoning traces; a smaller student learns them with SFT or on-policy distillation.

1. **Why can distillation beat small-model RL from scratch?** — The teacher supplies a high-quality trajectory distribution that the small model may not discover efficiently.

1. **↗ Industry 2026: What is hybrid thinking/non-thinking behavior?** — One model switches between fast direct answers and deeper reasoning, often controlled by routing or a token budget.

1. **Why should internal reasoning traces not be treated as guaranteed explanations?** — Generated rationales may be incomplete, post-hoc, or optimized for reward rather than faithful mechanism reporting.

---

# 9. RAG, retrieval, and grounding

1. **🔥 What is RAG?** — Retrieve relevant external evidence, augment the prompt, then generate a grounded answer.

1. **Why use RAG instead of only fine-tuning?** — RAG updates knowledge without changing weights and can return provenance.

1. **Why not put the whole corpus in context?** — Finite attention, cost, latency, and distraction from irrelevant tokens.

1. **Offline RAG pipeline?** — Parse → chunk → contextualize/metadata → embed → index.

1. **Online RAG pipeline?** — Query rewrite/embed → candidate retrieval → rerank → construct context → generate → cite/verify.

1. **What is the chunk-size trade-off?** — Small chunks are precise but context-poor; large chunks are coherent but semantically diluted.

1. **Why overlap chunks?** — Preserve facts spanning boundaries; cost is storage and duplicate retrieval.

1. **What is a bi-encoder retriever?** — Encode query and document separately, enabling precomputed document vectors.

1. **Dense-retrieval shapes?**
   ```text
   query embeddings: [B, De]
   chunk embeddings: [M, De]
   scores:           [B, M]
   ```

1. **Cosine similarity?**
$$
   \cos(q,d)=\frac{q^\top d}{\|q\|\|d\|}.
$$

1. **When are cosine, dot product, and L2 rankings equivalent?** — When embeddings are unit normalized.

1. **What is a contrastive retriever loss?** — Raise similarity of matched query–document pairs relative to negatives.

1. **What is a hard negative?** — A plausible but irrelevant document the retriever currently confuses with the positive.

1. **Why are hard negatives useful?** — They teach fine relevance boundaries better than random negatives.

1. **What is BM25’s memory hook?** — Sparse lexical retrieval rewarding query-term overlap with TF saturation and IDF weighting.

1. **Dense vs BM25?** — Dense captures semantic paraphrases; BM25 excels at exact names, IDs, codes, and rare terms.

1. **Why use hybrid retrieval?** — Combine semantic recall with lexical precision.

1. **What is ANN search?** — Approximate nearest-neighbour indexing trades a little recall for huge latency savings over full scans.

1. **Candidate retrieval vs reranking?** — Retrieve broadly/cheaply, then score a small candidate set accurately.

1. **Why is a cross-encoder reranker more accurate?** — Query and document tokens interact jointly through attention.

1. **Why can a reranker not recover a missed document?** — It only sees first-stage candidates.

1. **What is HyDE?** — Generate a hypothetical answer/document, embed it, and retrieve documents near that document-like representation.

1. **HyDE failure mode?** — A hallucinated hypothetical document can steer retrieval toward the wrong topic.

1. **What is contextual retrieval?** — Prepend concise document-level context to each chunk before embedding/indexing.

1. **What is query rewriting?** — Transform the user request into search-friendly subqueries, expansions, or filters.

1. **Multi-query retrieval?** — Issue diverse rewrites and fuse results to improve recall.

1. **What is reciprocal-rank fusion?** — Merge ranked lists by summing reciprocal rank contributions rather than raw incomparable scores.

1. **Precision@k vs recall@k?** — Precision: fraction of retrieved top-k that are relevant; recall: fraction of all relevant items retrieved.

1. **MRR?** — Mean inverse rank of the first relevant result; rewards getting one answer near the top.

1. **NDCG?** — Discount graded relevance by rank and normalize by the ideal ordering.

1. **Retrieval recall vs answer faithfulness?** — Recall asks whether evidence arrived; faithfulness asks whether the answer is supported by it.

1. **How do you debug a wrong RAG answer?** — Separate retrieval miss, reranking/context-construction failure, and generator misuse.

1. **How do you reduce citation hallucination?** — Attach source IDs structurally, constrain citation format, and verify each claim–source match.

1. **What is “lost in the middle”?** — Models may underuse relevant evidence buried in long context despite nominal context capacity.

1. **RAG vs long-context prompting?** — Long context avoids retrieval misses but costs more and adds noise; RAG is selective but can miss evidence.

1. **RAG vs parametric memory?** — RAG is editable and attributable; weights are fast and compressed but stale and opaque.

---

# 10. Tool use, agents, context engineering, and memory

1. **What is tool/function calling?** — The model predicts a structured tool name and arguments; application code executes it.

1. **What does the model usually see about a tool?** — Name, description, argument schema, constraints, and sometimes examples—not backend implementation.

1. **Runtime boundary?** — The LLM proposes; trusted software validates, authorizes, executes, and returns an observation.

1. **Why use structured output?** — Make calls parseable and schema-valid rather than relying on free-form text extraction.

1. **Tool-selection error vs argument error?** — Wrong capability chosen vs correct capability called with wrong values.

1. **What is tool hallucination?** — Calling a nonexistent tool or inventing unsupported arguments.

1. **How do you reduce tool hallucination?** — Clear schemas, constrained decoding, tool routing, examples/SFT, and runtime validation.

1. **What is a tool router?** — A first stage selecting a small relevant tool subset before exposing full schemas to the main model.

1. **Why should the router emphasize recall?** — A needed tool excluded upstream cannot be selected downstream.

1. **What is ReAct?** — Interleave reasoning/planning, action, and observation until a stop condition.

1. **Tool call vs workflow vs agent?** — One invocation; predetermined sequence; adaptive loop with state and stopping rules.

1. **What makes an agent an agent?** — Goal, state, model/controller, tools, observation loop, and termination policy.

1. **Why do long-horizon agent errors compound?** — Every extra decision/tool call adds another chance of failure or state corruption.

1. **Reliability memory equation?** — If independent step success is `p`, `n` perfect steps are roughly `p^n`.

1. **Plan-then-execute vs interleaved planning?** — Global plan is efficient but brittle; interleaving adapts after each observation.

1. **What is reflection?** — Ask the model or a verifier to critique progress and revise the plan/output.

1. **Why can reflection fail?** — The critic shares the same blind spots and may confidently reinforce an error.

1. **What is agent state?** — Current goal, conversation, tool observations, intermediate artifacts, plan, and completion status.

1. **What is short-term vs long-term memory?** — In-context working state vs persistent external storage retrieved across sessions.

1. **↗ Industry 2026: What is context engineering?** — Curate the smallest high-signal set of instructions, evidence, state, and tools for the next model call.

1. **Why is context engineering more than prompt writing?** — It includes retrieval, compaction, memory, tool-result filtering, ordering, and subagent boundaries.

1. **What is context compaction?** — Summarize or externalize old state while preserving decisions, constraints, and unresolved tasks.

1. **What is the danger of lossy compaction?** — Dropped constraints or provenance silently change future agent behavior.

1. **What is just-in-time context?** — Fetch detailed information/tools only when the current step needs them.

1. **What is subagent decomposition?** — Give focused tasks clean context windows and have a coordinator integrate results.

1. **Subagent risk?** — Coordination overhead, duplicated work, inconsistent assumptions, and error aggregation.

1. **What is MCP?** — An open protocol for exposing tools/resources/prompts from servers to AI hosts/clients.

1. **MCP does not solve what?** — Tool quality, authorization, prompt injection, semantic correctness, or agent planning.

1. **What is A2A conceptually?** — A protocol for agents to advertise skills, exchange tasks/status, and coordinate across systems.

1. **What is idempotency in tool execution?** — Retrying the same request does not duplicate side effects.

1. **Why are idempotency keys important?** — Network/model retries should not send two payments, emails, or database updates.

1. **What is least privilege?** — Give an agent only the minimum tools/data/action scope needed for the task.

1. **When should a human approve an action?** — When consequences are high, irreversible, financial, external, or privacy-sensitive.

1. **What is sandboxing?** — Execute generated code/actions in an isolated environment with restricted resources and permissions.

1. **What is prompt injection?** — Untrusted content contains instructions attempting to override the agent’s intended policy.

1. **Why is RAG a prompt-injection surface?** — Retrieved documents are untrusted text entering the same context as trusted instructions.

1. **Prompt injection vs jailbreak?** — Injection comes from data/tool content attacking an application; jailbreak is a user trying to bypass model policy.

1. **How do you defend against prompt injection?** — Trust separation, least privilege, content labeling, allowlisted actions, validation, isolation, and confirmations.

1. **What should every tool return?** — A structured success/failure status plus minimal, meaningful result data.

1. **Why is `None` a bad action result?** — The model cannot distinguish success, failure, or no matches; explicit state prevents false confirmation.

1. **What is an agent harness?** — The orchestration around the model: prompts, state, tools, retries, memory, permissions, verification, and stopping logic.

1. **↗ Industry 2026: Why evaluate model + harness together?** — Agent success depends on orchestration and environment, not just the base checkpoint.

1. **↗ Industry 2026: What is persistent assistant memory’s central trade-off?** — Continuity/personalization versus privacy, stale beliefs, user control, and retrieval correctness.

---

# 11. Evaluation, benchmarks, calibration, and safety

1. **🔥 Why is there no single LLM metric?** — Outputs span factuality, reasoning, code, style, safety, latency, and cost.

1. **Three evaluator families?** — Humans, deterministic/reference metrics, and learned/LLM judges.

1. **Human-evaluation strength and weakness?** — Closest to intended value; expensive, slow, and subjective.

1. **Why track inter-rater agreement?** — Low agreement means the rubric or task may be ambiguous.

1. **Cohen’s kappa?**
$$
   \kappa=\frac{P_o-P_e}{1-P_e}.
$$
   *Hook: agreement beyond chance.*

1. **BLEU memory hook?** — Modified n-gram precision plus brevity penalty, traditionally for translation.

1. **ROUGE memory hook?** — Reference-overlap family, commonly recall-oriented for summarization.

1. **Why do overlap metrics fail for general assistants?** — Valid semantic paraphrases can share few words.

1. **What is exact match good for?** — Canonical short answers with reliable normalization.

1. **What is execution-based evaluation?** — Run generated code/queries/actions and score actual behavior or final state.

1. **What is LLM-as-a-judge?** — A model applies a rubric to one or more candidate responses and returns a verdict/rationale.

1. **Pointwise vs pairwise judging?** — Score one response vs choose between responses; pairwise is often easier to calibrate.

1. **Judge position bias?** — Preference changes when response order changes.

1. **Judge verbosity bias?** — Longer answers are preferred despite equal or worse correctness.

1. **Self-enhancement bias?** — A model may prefer outputs similar to its own style or family.

1. **How do you mitigate judge bias?** — Swap order, use crisp rubrics, blind identity, multiple judges, low temperature, and human calibration.

1. **Why use structured judge output?** — Guarantee parseable fields such as verdict, confidence, and error category.

1. **Why should a judge not be treated as an oracle?** — It is another fallible model and can be gamed or miscalibrated.

1. **How do you calibrate an LLM judge?** — Compare with expert human labels, analyze disagreement slices, and tune rubric/examples.

1. **What is factuality decomposition?** — Extract atomic claims, retrieve evidence, verify each claim, then aggregate.

1. **Faithfulness vs factuality?** — Faithfulness: supported by supplied context; factuality: true in the world.

1. **What is calibration?** — Predicted confidence should match empirical correctness frequency.

1. **ECE memory hook?** — Bin by confidence and average the gap between confidence and accuracy.

1. **Why is selective prediction useful?** — Let the model abstain/escalate when uncertain, trading coverage for accuracy.

1. **What is benchmark saturation?** — Top systems approach the ceiling, so score differences stop distinguishing real capability.

1. **What is Goodhart’s law?** — When a measure becomes a target, it ceases to be a good measure.

1. **Why use private/rolling evals?** — Reduce contamination and benchmark-specific overfitting.

1. **What is capability profiling?** — Report performance across task families rather than one leaderboard scalar.

1. **What is a Pareto frontier?** — Configurations where no objective improves without worsening another.

1. **Agent evaluation: final answer or trajectory?** — Prefer actual task/world-state success, then use trajectory analysis for diagnosis.

1. **What is a punt?** — The system declines/fails to act when a valid answer or tool path existed.

1. **How do you evaluate a tool router?** — Recall of required tools, precision/context cost, and downstream task success.

1. **How do you evaluate long-horizon agents?** — Task success, policy compliance, step count, cost, latency, recovery, and repeated-run reliability.

1. **↗ Industry 2026: Why are agent evals becoming CI infrastructure?** — Model, prompt, tool, and harness changes can regress behavior; task banks provide repeatable release gates.

1. **What is red teaming?** — Adversarially search for harmful, insecure, or policy-violating behavior before deployment.

1. **Model safety vs system safety?** — Model safety shapes outputs; system safety constrains context, permissions, tools, and consequences.

1. **Why is safety defense-in-depth?** — No single model filter reliably handles every attack or failure mode.

1. **What is data exfiltration risk?** — An agent sends private data through an external action or attacker-controlled channel.

1. **How do you reduce exfiltration risk?** — Data classification, least privilege, destination controls, secret isolation, confirmations, and audit logs.

1. **What is sycophancy?** — The model agrees with the user’s stated belief rather than optimizing truth/helpfulness.

1. **What is over-refusal?** — Safety tuning blocks benign requests, reducing usefulness.

1. **Why must safety evals match product policy?** — “Safe” is partly policy/context dependent; generic benchmarks may not represent deployment boundaries.

---
# 12. Multimodal LLMs and alternative generation architectures

1. **Why can a Transformer process images?** — It consumes vectors; image patches can be projected into a token sequence.

1. **ViT patch count?**
$$
   N_{patch}=\frac{H}{P}\frac{W}{P}.
$$

1. **ViT patch-vector size?** — A raw RGB patch has `P²C` values before projection to model width `D`.

1. **ViT shape flow?**
   ```text
   image:          [B, C, H, W]
   patches:        [B, Npatch, P*P*C]
   patch tokens:   [B, Npatch, D]
   + CLS token:    [B, Npatch+1, D]
   ```

1. **CNN vs ViT inductive bias?** — CNN hard-codes locality/translation structure; ViT learns interactions more freely but typically needs more data.

1. **What is CLIP?** — Contrastively align image and text embeddings so matched pairs are close and mismatched pairs are far.

1. **CLIP loss memory hook?** — Symmetric cross-entropy over image-to-text and text-to-image similarity matrices.

1. **What is a VLM projector?** — Map vision-encoder features from `Dvision` into the language model width `Dlm`.

1. **Early fusion vs cross-attention multimodal design?** — Early fusion concatenates modality tokens; cross-attention keeps visual features as external memory.

1. **Early-fusion benefit/cost?** — Simple unified decoder and rich interactions; visual tokens consume context and KV cache.

1. **Cross-attention benefit/cost?** — Separate visual memory can be efficient; architecture/training is more specialized.

1. **What is a resampler/query transformer?** — Compress a variable/high-count visual sequence into a smaller fixed set of learned visual tokens.

1. **Why compress visual tokens?** — High-resolution images/video otherwise dominate context length and attention cost.

1. **What is dynamic resolution?** — Adapt patching/cropping/token count to image size/content rather than forcing one fixed grid.

1. **What is visual grounding?** — Connect language spans to regions/coordinates rather than only producing a global description.

1. **Why is OCR hard for VLMs?** — Exact small text requires high spatial resolution, correct reading order, and low tolerance for token errors.

1. **How does video extend image tokenization?** — Add a temporal dimension; patch/tubelet tokens need spatial and temporal position information.

1. **Why is video expensive?** — Token count grows across frames, so dense spatiotemporal attention becomes enormous.

1. **What is modality imbalance?** — One modality dominates gradients/tokens, causing the model to ignore another modality.

1. **What is modality dropout?** — Randomly remove modalities during training to encourage robustness and prevent overreliance.

1. **What does “native multimodal” usually imply?** — Multiple modalities are trained jointly in one model/objective rather than connected only after separate pre-training.

1. **What is 2D/3D RoPE?** — Apply relative rotations along spatial axes, and optionally time, so position interactions reflect multi-axis offsets.

1. **Autoregressive image/text generation vs diffusion?** — AR extends a sequence; diffusion iteratively refines a whole noisy/masked canvas.

1. **What is masked-diffusion language modelling?** — Corrupt tokens into masks and learn iterative parallel denoising/unmasking.

1. **Why can diffusion LMs reduce sequential depth?** — Update many positions per refinement step instead of one token per forward pass.

1. **Why are diffusion LMs not automatically faster?** — Each refinement step processes a wide sequence, and quality may require many steps.

1. **Why are diffusion LMs attractive for fill-in-the-middle?** — They naturally condition on visible tokens on both sides of masked positions.

1. **AR vs diffusion memory hook?** — *AR grows a prefix; diffusion revises a draft.*

1. **What is an SSM alternative to attention?** — Maintain a recurrent state with sequence-linear updates rather than explicit all-pairs token retrieval.

1. **What is memory compression for long context?** — Summarize old activations/state into compact persistent representations instead of retaining every token.

1. **Compression risk?** — Once details are discarded, exact retrieval or constraint preservation may be impossible.

1. **↗ Industry 2026: Why are model routers increasingly important?** — Systems choose among fast, deep-reasoning, multimodal, or specialized models based on task and budget.

1. **↗ Industry 2026: What is “computer use” for an LLM?** — Perceive a GUI, choose actions, observe the changed screen, and iterate toward a goal.

1. **What makes computer-use evaluation harder than QA?** — Environment state, timing, irreversible actions, and harness behavior affect success.

1. **↗ Industry 2026: Why are multi-agent research systems emerging?** — Specialized generators, critics, searchers, and verifiers can explore and cross-check in parallel.

1. **Multi-agent caveat?** — More agents do not guarantee better answers; correlated errors and orchestration cost can dominate.

1. **↗ Industry 2026: Why are small language models still important?** — Cost, privacy, latency, edge deployment, and high-volume specialized tasks.

1. **How do small models gain frontier-like skills?** — Distillation, targeted data, retrieval/tools, quantization, and narrow task optimization.

1. **What is hardware–software co-design?** — Shape model architecture, precision, kernels, memory layout, and scheduling around accelerator capabilities.

---

# 13. Debugging LLM training, post-training, retrieval, and agents

## Training and optimization failures

1. **Loss is very small from the first step—what do you check?** — Label leakage, wrong shifting, padding domination, duplicated targets, tiny vocabulary/task, or logging scale.

1. **Loss does not decrease—first checks?** — Data/labels, attention mask, LR, optimizer step, gradient flow, train/eval mode, and parameter freezing.

1. **Loss is NaN—likely causes?** — Overflow, bad normalization, invalid log/softmax inputs, extreme LR, corrupted data, or distributed reduction issues.

1. **Loss spikes periodically—what might cause it?** — Bad data shards, LR schedule boundaries, long-sequence batches, expert imbalance, optimizer instability, or checkpoint/resume bugs.

1. **Gradient norm is zero—what do you inspect?** — Detached graph, frozen parameters, empty loss mask, saturated operations, or wrong optimizer parameter groups.

1. **Gradient norm explodes—what do you inspect?** — LR, initialization, sequence outliers, normalization, loss scale, and clipping.

1. **Training loss falls but validation rises?** — Overfitting, train/eval distribution mismatch, contamination, or too-aggressive fine-tuning.

1. **Validation improves but product quality does not?** — Offline metric misalignment, missing task slices, prompt/harness differences, or proxy overoptimization.

1. **Distributed workers see identical data—why bad?** — Effective batch diversity shrinks; verify sampler rank/seed/sharding.

1. **Distributed loss differs across ranks—possible cause?** — Unequal masks/tokens, data corruption, missed gradient synchronization, or model-state divergence.

1. **Resume from checkpoint changes trajectory—why?** — Missing optimizer/RNG/scheduler/data-loader state or nondeterministic kernels.

1. **GPU utilization is low during training—where to look?** — Data loader, small batches, communication, pipeline bubbles, kernel fragmentation, CPU preprocessing, or synchronization.

1. **Model trains but is extremely slow—what profile first?** — Time by data, forward, backward, optimizer, collectives, and checkpoint I/O.

1. **How do you debug precision instability?** — Compare FP32/BF16, inspect overflow/underflow, scaler behavior, reductions, and sensitive ops.

1. **How do you identify a bad data shard?** — Log sample IDs and per-batch loss; replay outlier batches deterministically.

## Transformer and MoE failures

1. **Attention is nearly uniform everywhere—possible reasons?** — Q/K scale too small, masking bug, poor initialization, or model not yet trained.

1. **Attention is one-hot too early—possible reasons?** — Q/K scale too large, missing `sqrt(Dh)`, precision overflow, or positional bug.

1. **Causal LM has suspiciously excellent training accuracy—check what?** — Future-token leakage or incorrectly constructed causal masks.

1. **Long-context quality collapses—what do you test?** — Position extrapolation, RoPE scaling, retrieval position, attention precision, and train-length mismatch.

1. **GQA conversion hurts quality—why?** — Too few KV groups, poor checkpoint conversion/averaging, or insufficient adaptation.

1. **KV cache produces different outputs from full recomputation—likely bug?** — Position IDs/RoPE offsets, cache ordering, mask length, or layer-state mismatch.

1. **MoE experts are imbalanced—what metrics?** — Token fraction, mean routing probability, overflow/drop rate, capacity use, and per-expert gradients.

1. **MoE quality drops when balancing improves—why?** — Balancing objective is overpowering task loss or forcing semantically bad routing.

1. **MoE throughput is poor despite sparse compute—why?** — All-to-all communication, tiny expert batches, imbalance, routing overhead, or memory movement.

## SFT, preference, and reasoning failures

1. **SFT model repeats the prompt—why?** — Loss was applied to prompt tokens, chat template is wrong, or response boundary tokens are missing.

1. **SFT model becomes verbose—what could drive it?** — Demonstration length bias, response-only data distribution, or evaluation rewarding length.

1. **Reward model always prefers longer answers—what is happening?** — Verbosity bias in labels/features; balance lengths and improve rubric/data.

1. **Reward-model scores drift arbitrarily—why?** — Bradley–Terry identifies score differences, not an absolute origin; calibrate differences/rankings.

1. **PPO reward rises while human quality falls—diagnosis?** — Reward hacking or out-of-distribution exploitation.

1. **PPO KL explodes—what do you change?** — Lower LR/clip range, strengthen adaptive KL, reduce epochs, or improve reward scale.

1. **DPO model becomes too deterministic—why?** — Preference data lacks diversity, beta/updates are aggressive, or chosen responses are narrow.

1. **DPO chosen and rejected log-probs both fall—can learning still occur?** — Yes; DPO optimizes their relative reference-adjusted margin, but absolute degradation should be monitored.

1. **GRPO gives no gradient on many prompts—why?** — Same reward for every sample in the group; improve exploration, group size, or reward density.

1. **Reasoning traces grow without accuracy gains—what do you inspect?** — Length incentives, stopping policy, verifier quality, and reward normalization.

1. **Verifier-selected solutions are wrong but high scoring—what happened?** — Verifier exploitation; strengthen evidence, process checks, adversarial training, or deterministic execution.

## RAG and agent failures

1. **Correct document never appears—where is the failure?** — Ingestion, chunking, embedding, index freshness, query rewrite, or first-stage recall.

1. **Dense retrieval misses an exact product/code/name—fix?** — Add BM25 or hybrid retrieval and preserve lexical metadata.

1. **Correct chunk is retrieved but answer is wrong—next checks?** — Context ordering/truncation, conflicting evidence, instruction hierarchy, and generation faithfulness.

1. **RAG returns many duplicates—fix?** — Source-aware deduplication, overlap control, maximal marginal relevance, or rank fusion.

1. **RAG latency is high—decompose it.** — Query generation, embedding, ANN, reranking, context assembly, prefill, and decode.

1. **Agent chooses no tool—possible causes?** — Router miss, unclear schema, prompt/SFT gap, or model capacity.

1. **Agent chooses the wrong tool—fix?** — Disambiguate names/scopes, route candidates, add contrastive examples, or enforce policies.

1. **Correct tool, wrong arguments—fix?** — Improve schema/type/unit descriptions, supply missing context, constrain decoding, and validate.

1. **Tool succeeds but model says it failed—cause?** — Noisy/oversized/unstructured result or synthesis prompt/model weakness.

1. **Tool fails but model claims success—cause?** — Missing explicit status and confirmation; require structured result and postcondition checks.

1. **Agent loops repeatedly—fix?** — Max-step budget, progress detector, repeated-action block, explicit finish criteria, and fallback/escalation.

1. **Agent cost unexpectedly explodes—inspect what?** — Repeated tool calls, context growth, subagent fan-out, retry policy, and reasoning budget.

1. **Agent benchmark score changes across machines—why?** — Environment timing, dependency versions, network state, tool nondeterminism, or infrastructure noise.

---

# 14. Research-scientist and system-design questions

1. **How do you turn an idea into a testable hypothesis?** — State the mechanism, predicted observable change, control, and falsifying result.

1. **What is the smallest useful first experiment?** — One dataset/task, strong baseline, one changed variable, and a diagnostic metric tied to the hypothesis.

1. **What makes an ablation informative?** — It isolates one mechanism while holding data, compute, optimization, and evaluation fixed.

1. **Why are “remove component” ablations sometimes insufficient?** — Removal changes parameter count/compute; use matched-capacity or replacement controls.

1. **How do you compare architectures fairly?** — Match training tokens, active FLOPs, wall-clock/hardware where relevant, data, tokenizer, and tuning budget.

1. **What is a compute-matched comparison?** — Give methods the same estimated training/inference computation rather than equal parameter count alone.

1. **What is a latency-matched comparison?** — Compare quality under the same serving delay or cost budget.

1. **Why run multiple seeds?** — Separate systematic improvement from optimization/evaluation variance.

1. **When do you need statistical significance?** — When effect size is near run-to-run or evaluator noise and decisions depend on the difference.

1. **What is an error taxonomy?** — Group failures by mechanism/stage so one fix can address a class rather than isolated examples.

1. **How do you build an evaluation set for a new product?** — Sample real traffic, define task/risk slices, label success, include adversarial cases, and freeze a regression set.

1. **Why keep a hidden test set?** — Prevent prompt/model iterations from overfitting the visible evaluation bank.

1. **What is a canary evaluation?** — A small fast suite run frequently to catch obvious regressions before expensive full evaluation.

1. **Offline vs online evaluation?** — Offline is repeatable/diagnostic; online measures real user outcomes but is slower, noisier, and riskier.

1. **What is an A/B test’s main benefit?** — Causal comparison on real traffic under randomized assignment.

1. **What can confound an A/B test?** — User heterogeneity, novelty, interference, logging changes, exposure duration, and metric gaming.

1. **How do you choose a primary metric?** — Tie it to the product objective and constrain it with safety/quality guardrails.

1. **What should you do when benchmark and human preference disagree?** — Audit the benchmark/judge and optimize the real target, not the convenient proxy.

1. **How do you interpret a negative result?** — Check power, implementation, optimization, and hypothesis assumptions; report what mechanism was ruled out.

1. **What would you do with 10× compute?** — Scale the bottleneck implied by evidence, not every dimension blindly; include an experiment ladder and stopping rules.

1. **What is research taste?** — Choosing high-leverage questions, decisive experiments, strong baselines, and tractable paths to learning.

1. **How do you critique a paper claiming a 2% gain?** — Ask about compute/data matching, tuning budget, seeds, contamination, effect size, and deployment cost.

1. **How do you critique a speedup claim?** — Compare matched quality, hardware, batch, sequence length, precision, warmup, and end-to-end latency.

1. **How do you critique a long-context claim?** — Separate nominal length from retrieval accuracy, reasoning across distant evidence, and cost.

1. **How do you critique an agent benchmark?** — Inspect environment realism, verifier correctness, repeated-run reliability, contamination, and harness dependence.

1. **How do you decide between a larger model and a stronger system harness?** — Find whether errors arise from model capability or missing context/tools/verification/control.

1. **When is a deterministic workflow better than an agent?** — When steps are known, consequences are high, and adaptivity adds little value.

1. **Design a production RAG assistant in one line.** — Ingest/version → hybrid retrieve → rerank → context/citations → generate → verify → monitor/evaluate.

1. **Design a coding agent in one line.** — Understand repo → plan → edit in sandbox → run tests/lint → inspect failures → iterate → summarize diff.

1. **Design a safe action agent in one line.** — Route tools → validate args → policy/permission gate → execute idempotently → verify postcondition → log/confirm.

1. **Design a high-throughput LLM service in one line.** — Quantized/sharded model + paged KV + continuous batching + chunked prefill + speculative decode + SLO-aware scheduler.

1. **How do you select a model for production?** — Choose the cheapest model/harness on the quality–latency–safety Pareto frontier for the task.

1. **Build vs fine-tune vs RAG vs prompt?** — Prompt for behavior, RAG for changing knowledge, fine-tune for persistent behavior/format, train from scratch only for major scale/domain needs.

1. **When should you distill?** — When a strong expensive teacher exists and deployment needs lower latency/cost or on-device execution.

1. **How do you keep up with the field without chasing noise?** — Follow primary papers, technical reports/code, reproduce key claims, and maintain a stable conceptual map.

---

# 15. Whiteboard and coding prompts you should be ready for

1. **Implement scaled dot-product attention.** — Expected: QK transpose, scale, add mask before softmax, multiply V, stable shapes.

1. **Implement a causal mask.** — Upper triangle above the diagonal is invalid and receives `-inf`.

1. **Implement multi-head reshape.** — `[B,T,D] → [B,H,T,Dh]`, attend, transpose/reshape back to `[B,T,D]`.

1. **Implement GQA.** — Produce `Hq` Q heads and `Hkv` K/V heads; repeat/map each KV group to multiple query heads.

1. **Implement RoPE.** — Pair features, apply position-dependent sine/cosine rotation to Q/K, preserve shape.

1. **Implement RMSNorm.** — `x * rsqrt(mean(x²)+eps) * weight`.

1. **Implement SwiGLU.** — `down(silu(gate(x)) * up(x))`.

1. **Implement a KV-cache decode step.** — Compute new Q/K/V, append new K/V, attend new Q over cached+new K/V.

1. **Implement top-k sampling.** — Mask logits outside top `k`, softmax, sample categorical.

1. **Implement top-p sampling.** — Sort probabilities, keep smallest prefix whose cumulative mass reaches `p`, renormalize, sample.

1. **Implement temperature.** — Divide logits by `tau` before softmax; handle near-zero temperature as greedy.

1. **Implement LoRA linear.** — `xW0^T + scale * (xA^T)B^T`, with `W0` frozen.

1. **Implement response-only SFT loss.** — Shift logits/labels, mark non-assistant labels `-100`, cross-entropy over remaining tokens.

1. **Implement Bradley–Terry reward loss.** — `-logsigmoid(r_chosen - r_rejected).mean()`.

1. **Implement DPO loss.** — Compute chosen/rejected completion log-probs under policy/reference, form reference-adjusted margin, apply `-logsigmoid(beta*margin)`.

1. **Implement GRPO advantages.** — Group rewards by prompt, subtract group mean, divide by group std with epsilon.

1. **Implement pass@k.** — `1 - C(n-c,k)/C(n,k)` with edge case `n-c < k → 1`.

1. **Implement contrastive retrieval loss.** — Similarity matrix `[B,B]`; cross-entropy with diagonal positives, often symmetric.

1. **Implement precision@k and recall@k.** — Relevant in top-k divided by `k`, and divided by total relevant count.

1. **Implement NDCG.** — Discount relevance by `log2(rank+1)` and divide by ideal DCG.

1. **Implement a tool-call validator.** — Parse structured output, validate schema/types/ranges/permissions, reject or request correction before execution.

1. **Implement an agent loop.** — Observe → choose action/finish → validate/execute → append structured observation → stop on success/budget.

1. **Implement continuous batching conceptually.** — Scheduler admits/removes sequences each decode step and allocates/frees KV blocks dynamically.

---
# 16. Equations to reproduce without notes

## Language modelling

$$
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t})
$$

$$
\mathcal L_{CE}=-\log p(y_{true}),
\qquad
\frac{\partial\mathcal L}{\partial z}=p-y
$$

$$
\mathrm{PPL}=\exp\left(-\frac1T\sum_t\log p(x_t\mid x_{<t})\right)
$$

## Attention

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
\operatorname{Attn}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{D_h}}+M\right)V
$$

$$
(R_mq)^\top(R_nk)=q^\top R_{n-m}k
$$

## Normalization and FFN

$$
\operatorname{RMSNorm}(x)=
\frac{x}{\sqrt{\frac1D\sum_jx_j^2+\epsilon}}\odot g
$$

$$
\operatorname{SwiGLU}(x)=W_d\big(\operatorname{SiLU}(xW_g)\odot xW_u\big)
$$

## Scaling and cache

$$
C_{train}\approx 6PN_{tok}
$$

$$
\text{KV bytes}\approx
2LBTH_{kv}D_h\times \text{bytes/element}
$$

## LoRA

$$
W=W_0+\frac{\alpha}{r}BA
$$

## Reward models and alignment

$$
P(y_w\succ y_l\mid x)=\sigma(r_w-r_l)
$$

$$
\mathcal L_{RM}=-\log\sigma(r_w-r_l)
$$

$$
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{old}(a_t\mid s_t)}
$$

$$
L^{clip}=\min\left(\rho_tA_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)
$$

$$
\mathcal L_{DPO}=-\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}
\right]\right)
$$

$$
A_i^{GRPO}=\frac{r_i-\bar r}{s_r+\epsilon}
$$

## Retrieval and evaluation

$$
S=E_qE_c^\top
$$

$$
\cos(q,d)=\frac{q^\top d}{\|q\|\|d\|}
$$

$$
\widehat{pass@k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

$$
\kappa=\frac{P_o-P_e}{1-P_e}
$$

---

# 17. Tensor shapes to reproduce without notes

```text
Token IDs:                    [B, T]
Embedding table:              [V, D]
Hidden states:                [B, T, D]
Vocabulary logits:            [B, T, V]
Labels:                       [B, T]

Q/K/V before head split:      [B, T, D]
Q after split:                [B, Hq, T, Dh]
K/V in MHA:                   [B, Hq, T, Dh]
K/V in GQA:                   [B, Hkv, T, Dh]
K/V in MQA:                   [B, 1, T, Dh]
Self-attention scores:        [B, Hq, T, T]
Cross-attention scores:       [B, Hq, T_target, T_source]
Attention output per head:    [B, Hq, T, Dh]
Concatenated output:          [B, T, D]

FFN:                          [B,T,D] -> [B,T,Dff] -> [B,T,D]
Router logits:                [B, T, E]
Top-k expert indices:         [B, T, k]

KV cache per layer:           K,V each [B, Hkv, T_cache, Dh]

Dense retrieval:
  queries                     [B, De]
  corpus                      [M, De]
  score matrix                [B, M]

Reward model:
  prompt-response hidden      [B, T, D]
  pooled state                [B, D]
  reward                      [B]

Preference batch:
  chosen/rejected token IDs   [B, T]
  completion log-probs        [B]

GRPO:
  completions                 [B, G, T]
  rewards                     [B, G]
  advantages                  [B, G]

ViT:
  image                       [B, C, H, W]
  flattened patches           [B, Npatch, P*P*C]
  patch tokens                [B, Npatch, D]

Early-fusion VLM:
  visual tokens               [B, Nvision, D]
  text tokens                 [B, Ttext, D]
  joint sequence              [B, Nvision+Ttext, D]
```

---

# 18. The final 75 memory hooks

1. Tokenization changes the model’s effective compute and context budget.
2. Subwords trade sequence length for vocabulary reuse.
3. Embedding lookup is efficient one-hot multiplication.
4. Causal LM = prefix to next-token distribution.
5. Training is parallel because targets are known; decoding is sequential because futures are not.
6. Q asks, K matches, V supplies.
7. Attention = compare, scale, mask, normalize, retrieve.
8. Attention scores are `[B,H,T,T]`.
9. `sqrt(Dh)` prevents softmax saturation.
10. Attention mixes tokens; the FFN transforms features.
11. Residuals preserve identity paths.
12. Pre-Norm usually stabilizes deep training.
13. RoPE rotates Q/K so dot products encode relative position.
14. GQA preserves query diversity while shrinking KV memory.
15. KV cache trades memory/bandwidth for avoided recomputation.
16. MoE adds total capacity while limiting active parameters.
17. Sparse MoE can become communication-bound.
18. Routing collapse means a few experts monopolize traffic.
19. Pre-training builds capability; post-training shapes behavior.
20. Data mixture controls what scale teaches.
21. Deduplication protects against memorization and leakage.
22. Synthetic data needs strong generation, verification, filtering, and mixing.
23. `6PN` is a planning approximation, not a physical law.
24. Sample efficiency is not compute efficiency.
25. Warmup protects fragile early optimization.
26. BF16 is usually safer than FP16 because of exponent range.
27. ZeRO shards optimizer, then gradients, then parameters.
28. FlashAttention is exact attention with less HBM traffic.
29. Prefill is parallel/compute-heavy; decode is sequential/bandwidth-heavy.
30. TTFT and TPOT diagnose different serving bottlenecks.
31. Continuous batching follows token steps, not fixed request batches.
32. PagedAttention applies virtual-memory ideas to KV cache.
33. Speculative decoding is useful only when the draft is cheap and accepted often.
34. Quantization buys memory/throughput at a possible accuracy cost.
35. LoRA changes parameterization, not the learning objective.
36. QLoRA quantizes the frozen base and trains adapters.
37. SFT imitates; preference tuning compares.
38. Reward models train pairwise but score pointwise.
39. `pi_old` stabilizes one PPO batch; `pi_ref` anchors overall behavior.
40. Reward hacking is proxy optimization without true objective satisfaction.
41. DPO is offline direct preference optimization; PPO is online policy optimization.
42. Reasoning tokens are inference-time compute.
43. Verifiable rewards make math/code attractive for RL.
44. ORM checks the destination; PRM checks the route.
45. Best-of-N needs both candidate diversity and a reliable verifier.
46. Pass@k measures opportunity; pass^k measures reliability.
47. GRPO replaces the critic with same-prompt group statistics.
48. Identical group rewards mean no GRPO learning signal.
49. More reasoning is not always better reasoning.
50. RAG changes context, not weights.
51. First-stage retrieval should prioritize recall.
52. A reranker cannot recover what retrieval discarded.
53. Dense retrieval finds meaning; BM25 protects exact terms.
54. Hybrid retrieval combines semantic and lexical strengths.
55. RAG errors occur in retrieval, context construction, or generation.
56. The model proposes a tool call; software executes it.
57. A tool is a capability; an agent is a control loop.
58. Every extra agent step is another failure opportunity.
59. Context is a finite attention budget—spend it on high-signal tokens.
60. Memory outside context needs retrieval, privacy, freshness, and user control.
61. MCP standardizes connection, not correctness or safety.
62. Least privilege matters more as model autonomy grows.
63. Prompt injection is untrusted data trying to become instructions.
64. Agent success should be measured by world state, not polished prose.
65. An LLM judge is a scalable proxy, not an oracle.
66. Position, verbosity, and self-preference can bias judges.
67. Benchmarks measure profiles, not universal intelligence.
68. Goodhart: once optimized directly, a metric loses meaning.
69. ViT turns image patches into tokens.
70. Early fusion spends context; cross-attention keeps modality memory separate.
71. AR grows a prefix; diffusion revises a canvas.
72. SSMs compress history; attention explicitly retrieves history.
73. Current frontier systems route between fast and deep compute.
74. Model quality and harness quality jointly determine agent performance.
75. Research taste = high-leverage question + decisive fair experiment.

---

# 19. What top-company interviews increasingly test

```text
FOUNDATIONS
  Can you derive attention, masking, RoPE, normalization, and tensor shapes?

TRAINING
  Can you reason about data quality, scaling, optimization stability,
  distributed memory, and fair compute-matched experiments?

INFERENCE
  Do you understand prefill/decode, KV-cache economics, batching,
  quantization, speculative decoding, and latency-throughput trade-offs?

POST-TRAINING
  Can you compare SFT, reward models, PPO, DPO, GRPO, verifiers,
  distillation, and adaptive test-time compute?

SYSTEMS
  Can you design and debug RAG, tool use, agents, memory, permissions,
  and end-to-end evaluation rather than treating the model in isolation?

RESEARCH JUDGMENT
  Can you identify the bottleneck, propose a falsifiable hypothesis,
  choose strong controls, and explain what a negative result teaches?
```

---

# 20. Source basis and current-industry additions

## Supplied course material

- `lecture_1.md` — tokenization, embeddings, RNNs, attention, original Transformer.
- `lecture_2.md` — positional methods, normalization, GQA/MQA, BERT/T5 families.
- `lecture_3.md` — decoder-only LLMs, MoE, decoding, prompting, KV cache, PagedAttention, speculative decoding.
- `lecture_4.md` — pre-training, scaling, distributed training, FlashAttention, SFT, LoRA/QLoRA.
- `lecture_5.md` — preference data, reward models, RLHF, PPO, DPO.
- `lecture_6.md` — reasoning, verifiable rewards, pass@k, GRPO, DeepSeek-R1 pipeline, distillation.
- `lecture_7.md` — RAG, reranking, tools, MCP, ReAct, agents, safety.
- `lecture_8.md` — human/judge evaluation, factuality, agent failure modes, benchmarks.
- `lecture_9.md` — whole-course synthesis, ViT/VLM, diffusion LMs, data/model/hardware trends.

## Primary/official material used for **↗ Industry 2026** additions

- OpenAI, **GPT-5 System Card** — routing between fast and deeper-reasoning modes and parallel test-time compute.
- OpenAI, **Introducing o3 and o4-mini** — reasoning-trained tool selection and multimodal reasoning.
- OpenAI, **Let’s Verify Step by Step** — process- versus outcome-supervised reward models.
- Anthropic, **Introducing the Model Context Protocol** — standardized model-to-data/tool connections.
- Anthropic Engineering, **Effective Context Engineering for AI Agents** — context as a finite resource, compaction, memory, and subagents.
- Anthropic Engineering, **Demystifying Evals for AI Agents** — model-plus-harness evaluation and task-based regression suites.
- Google DeepMind / UC Berkeley, **Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters**, arXiv:2408.03314.
- Google DeepMind, **Gemini Deep Think** and **Co-Scientist** reports — inference-time scaling, verification, and multi-agent scientific workflows.
- Qwen Team, **Qwen3 Technical Report**, arXiv:2505.09388 — unified thinking/non-thinking modes, thinking budgets, MoE, and distillation.
- DeepSeek-AI, **DeepSeek-R1**, arXiv:2501.12948 — GRPO, verifiable-reward reasoning RL, and reasoning distillation.
- Rafailov et al., **Direct Preference Optimization**, arXiv:2305.18290.
- Kwon et al., **PagedAttention / vLLM**, arXiv:2309.06180.
- Shah et al., **FlashAttention-3**, arXiv:2407.08608.
- Dao and Gu, **Transformers are SSMs / Mamba-2**, arXiv:2405.21060.
- NVIDIA, **TensorRT-LLM** technical documentation — paged KV, in-flight batching, FP8/FP4, disaggregated serving, expert parallelism, and speculative decoding.
- Google DeepMind, **MELODI** — hierarchical memory compression for long contexts.

---

# Final 60-second answer: “What is the modern LLM stack?”

> Text is tokenized, embedded, and processed by a causal Transformer using attention, positional encoding, residual pathways, normalization, and FFNs; modern models often add GQA, gated FFNs, sparse MoE, and hardware-aware kernels. Pre-training learns broad next-token capability from large curated mixtures, followed by SFT and preference/reasoning post-training such as reward modelling, DPO, PPO, or GRPO. At inference, the main systems concerns are prefill versus decode, KV-cache memory and bandwidth, batching, quantization, speculative decoding, and distributed serving. Real products add retrieval, tools, memory, agent loops, permissions, verification, and evaluation harnesses around the model. The strongest interview answer therefore connects every component to the failure it fixes, the objective it optimizes, its tensor/system cost, and how it can fail.
