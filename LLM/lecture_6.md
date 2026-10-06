# CME 295 Lecture 6 - LLM Reasoning, Verifiable Rewards, GRPO, Length Bias, DeepSeek-R1, and Distillation

> **Scope:** This note is grounded in the supplied **CME 295 Lecture 6 transcript**. It adds standard equations, tensor-shape derivations, implementation mental models, and intuitive explanations only when they directly clarify material taught in this lecture. Such additions are labelled **INTERVIEW CLARIFICATION** when the exact detail was not fully derived in class.
>
> **Goal:** Previous-day revision for ML/AI Research Scientist interviews: important questions, distinctions, equations, tensor shapes, mechanisms, training pipelines, failure modes, implementation ideas, and memory liners.
>
> **Transcript normalization:** The automatic transcript sometimes renders **PPO** as "PO," **KL** as "Kale," **AIME** as "AIM," **DeepSeek** as "DeepSync," **DeepSeek-R1-Zero** as "R10," and likely **DAPO** as "DPO" in the length-bias section. This note uses the standard technical names while preserving the lecture's intended content.

**Source transcript:** CME 295 Lecture 6 - `https://www.youtube.com/watch/k5Fh-UgTuCo`

---

## Lecture spine

```text
STANDARD LLM PIPELINE
    pre-training
        -> broad language/code capability
    supervised fine-tuning
        -> instruction-following behavior
    preference tuning
        -> preferred style, safety, and helpfulness

LIMITATION
    ordinary assistant model
        -> often answers directly
        -> limited multi-step problem solving

REASONING MODEL
    prompt
        -> explicit reasoning / thinking tokens
        -> final answer

WHY REASONING CAN HELP
    decompose a hard problem into easier subproblems
    + spend more inference-time compute by generating more tokens

EVALUATION
    code: execute tests
    math: parse and compare final answer
    pass@k: probability at least one of k samples succeeds
    consensus@k: answer occurring most often

TRAINING FROM VERIFIABLE REWARDS
    prompt
        -> sample multiple completions
        -> score format + correctness
        -> optimize policy with RL

GRPO
    group of completions for the same prompt
        -> group rewards
        -> relative standardized advantages
        -> PPO-like clipped policy update
        -> reference-model KL regularization
    key benefit: no learned value function

GRPO FAILURE MODES / EXTENSIONS
    response-length normalization -> length bias
    group standard deviation -> difficulty-related bias
    symmetric clipping -> weak recovery for low-probability tokens

DEEPSEEK-R1-ZERO
    base model -> reasoning RL only
    strong reasoning gains, but readability and language-mixing issues

DEEPSEEK-R1
    cold-start reasoning SFT
        -> reasoning RL
        -> rejection-sampled reasoning + general SFT
        -> final mixed reasoning/helpfulness/harmlessness RL

DISTILLATION
    large reasoning teacher generates complete reasoning traces
        -> smaller student learns them with SFT
```

---

## Notation

```text
B             number of prompts in a batch
G             number of sampled completions per prompt
T_p           prompt length
T_o           maximum completion length
T_i           actual length of completion i
V             vocabulary size
D             model hidden width

x             prompt / query
o_i           completion i for prompt x
o_{i,t}       token t of completion i

pi_theta      trainable policy model
pi_old        frozen snapshot from the previous policy iteration
pi_ref        frozen reference model, usually the starting SFT/base checkpoint

r_i           scalar reward for completion i
r_bar         mean reward within one prompt's completion group
s_r           reward standard deviation within that group
A_i           group-relative advantage for completion i
rho_{i,t}     current-to-old token probability ratio

G             group size; do not confuse with return G_t in generic RL notation
epsilon       clipping width
beta          reference-model KL strength
M_{i,t}       completion-token mask
```

### Priority legend

- **MUST REMEMBER:** answer immediately and derive the equation or shape.
- **SHOULD KNOW:** explain the mechanism, trade-off, or failure mode clearly.
- **LECTURE BOUNDARY:** mentioned but not fully developed in this lecture.
- **INTERVIEW CLARIFICATION:** standard detail added to make the lecture implementation-ready.

---

# 1. Where reasoning models fit

## Q1. MUST REMEMBER - What training stages does the lecture review before introducing reasoning?

```text
1. Pre-training
   Learn language, code, and broad patterns through next-token prediction.

2. Supervised fine-tuning / instruction tuning
   Learn to respond usefully to instructions.

3. Preference tuning
   Learn which plausible responses humans or the product prefer.

4. Reasoning-oriented training
   Learn to spend intermediate generation steps solving verifiable problems.
```

**Memory liner:**

> Pre-training builds capability, SFT teaches behavior, preference tuning chooses behavior, and reasoning training improves multi-step problem solving.

---

## Q2. MUST REMEMBER - What weaknesses of a vanilla LLM motivate later techniques?

The lecture lists four broad weaknesses:

1. **Limited multi-step reasoning:** it may lose track of a difficult math or coding problem.
2. **Static knowledge:** its parametric knowledge is bounded by the pre-training cutoff.
3. **All talk, no action:** text generation alone cannot directly perform external actions.
4. **Difficult evaluation:** free-form responses are not captured well by one classical NLP metric.

Only the first weakness is the main topic of this lecture.

**Memory liner:**

> A vanilla LLM can know and write a lot without reliably solving, updating, acting, or being easy to evaluate.

---

## Q3. LECTURE BOUNDARY - Which limitations are deferred to later lectures?

The transcript explicitly defers:

- fetching missing or current information;
- tool use / actions;
- broader LLM evaluation.

Do not turn this sheet into a retrieval, agent, or general evaluation note.

---

# 2. What is reasoning?

## Q4. MUST REMEMBER - How does the lecture define reasoning?

The lecture uses a practical, non-universal definition:

> Reasoning is the ability to solve a problem through a multi-step process.

Typical examples are math and coding tasks, though the intention is broader.

**Memory liner:**

> Reasoning means constructing useful intermediate steps rather than only retrieving or immediately emitting an answer.

---

## Q5. MUST REMEMBER - Knowledge question versus reasoning question?

```text
Knowledge question:
    "What is the course code?"
    -> retrieve a fact

Reasoning question:
    "A bear was born in 2020; how old is it in 2025?"
    -> combine facts through steps
```

The distinction is not always sharp, but the lecture uses whether a multi-step solution process is needed.

**Memory liner:**

> Knowledge retrieves; reasoning transforms information into a solution.

---

## Q6. MUST REMEMBER - What is a reasoning model in this lecture?

A reasoning model is still an autoregressive language model, but its generated output includes an intermediate reasoning segment before the final answer.

```text
vanilla model:
    prompt -> answer

reasoning model:
    prompt -> reasoning tokens -> final answer
```

**Memory liner:**

> The architecture may still be a decoder-only LM; the changed behavior is to generate a solution trajectory before answering.

---

## Q7. SHOULD KNOW - Is there a universally agreed definition of LLM reasoning?

No. The lecturer explicitly says the field lacks one agreed definition and adopts a task-oriented definition centered on solving multi-step problems.

**Interview implication:** state the operational definition and the benchmark being used rather than claiming that "reasoning" has one settled scientific meaning.

---

# 3. Chain of thought and inference-time compute

## Q8. MUST REMEMBER - What is chain of thought in the lecture's framing?

Chain of thought means producing intermediate reasoning steps before the final answer instead of returning only a direct answer.

Earlier prompting work encouraged this with worked in-context examples. Reasoning-model training aims to induce the behavior at scale.

**Memory liner:**

> Chain of thought externalizes a multi-step trajectory into generated tokens.

---

## Q9. MUST REMEMBER - Why can problem decomposition help a next-token model?

A difficult complete problem may be unlike anything exactly seen during training. Decomposition turns it into smaller subproblems whose patterns are more likely to resemble learned examples.

```text
hard unfamiliar problem
        ↓ decompose
smaller familiar patterns
        ↓ solve sequentially
final answer
```

**Memory liner:**

> Decomposition converts one hard extrapolation into a sequence of easier local predictions.

---

## Q10. MUST REMEMBER - Why are reasoning tokens also test-time compute?

Every newly generated token requires another autoregressive forward step. A longer reasoning trace therefore allocates more inference computation to the problem.

Very roughly:

```text
more reasoning tokens
    -> more decoder steps
    -> more model evaluations
    -> more opportunity to transform and revise the state
```

**Memory liner:**

> More thinking tokens mean more sequential compute, not free extra text.

---

## Q11. MUST REMEMBER - What is a reasoning compute budget?

It is the amount of generation effort allocated before the model must finalize an answer, commonly reflected in:

- allowed reasoning-token count;
- time or latency budget;
- number of sampled reasoning trajectories;
- explicit short/standard/extended thinking mode.

**Memory liner:**

> The compute budget controls how much inference-time search or deliberation the model can perform.

---

## Q12. SHOULD KNOW - Why is unlimited reasoning not desirable?

Longer traces increase:

- latency;
- output-token cost;
- context consumption;
- the opportunity to wander or overthink;
- serving cost for the provider.

The lecture later shows that performance may plateau while response length continues increasing.

**Memory liner:**

> Useful reasoning should maximize accuracy per unit of reasoning compute, not merely maximize token count.

---

## Q13. SHOULD KNOW - What does the user interface's "thinking" display mean according to the lecture?

The interface indicates that the model spent time generating a reasoning process. The visible text may be a summary rather than the raw internal reasoning chain.

The lecturer offers possible explanations for hiding the raw trace, but frames them as hypotheses rather than established facts.

---

## Q14. SHOULD KNOW - Why can hidden reasoning tokens still affect API cost?

The generated reasoning is part of model output computation even when the full raw trace is not displayed. The lecture notes that providers may charge for those output/reasoning tokens.

**Memory liner:**

> Hidden from the user does not mean absent from computation or billing.

---

## Q15. LECTURE BOUNDARY - Does the lecture establish that natural-language chain of thought is the only possible reasoning representation?

No. It explicitly mentions research on **continuous thoughts**, where intermediate computation occurs in hidden-representation space rather than as ordinary language tokens.

The lecture does not derive a continuous-thought architecture.

---

# 4. Reasoning benchmarks and verifiable tasks

## Q16. MUST REMEMBER - Why are coding and math especially convenient for reasoning training?

They often provide cheap, deterministic verification:

```text
code:
    compile / execute tests / check expected outputs

math:
    parse final answer / compare against ground truth
```

This creates a reward signal without asking a learned reward model to judge every completion.

**Memory liner:**

> Reasoning RL becomes attractive when correctness is mechanically verifiable.

---

## Q17. SHOULD KNOW - Which coding benchmarks are named?

The lecture mentions:

- HumanEval;
- Codeforces;
- SWE-bench.

The important interview point is their verification style, not memorizing a benchmark catalog:

- generated code is executed;
- tests or repository tasks determine success.

---

## Q18. SHOULD KNOW - Which math benchmarks are named?

The lecture mentions:

- AIME;
- GSM8K.

The generated response contains reasoning plus a final answer that can be parsed and matched to ground truth.

---

## Q19. MUST REMEMBER - Why should the final answer be emitted in a parseable format?

An automatic evaluator must reliably isolate the answer from the reasoning text.

```text
reasoning ...
<answer>42</answer>
```

or a boxed-answer convention allows the evaluator to compare the extracted value with the ground truth.

**Memory liner:**

> Verifiable reward requires a deterministic bridge from free-form text to a checkable answer.

---

## Q20. SHOULD KNOW - What can go wrong in answer parsing?

Even when the reasoning is correct, reward may fail because of:

- malformed answer delimiters;
- extra units or punctuation;
- equivalent expressions in different forms;
- multiple candidate answers;
- code that times out;
- nondeterministic tests.

**INTERVIEW CLARIFICATION:** in a production system, the verifier and parser are part of the learning system and must be tested like any other label-generating component.

---

# 5. pass@k

## Q21. MUST REMEMBER - What is pass@k?

`pass@k` estimates the probability that **at least one** of `k` sampled attempts solves the problem.

```text
sample k candidate solutions
        ↓
check each with a verifier
        ↓
pass@k asks whether at least one succeeds
```

**Memory liner:**

> pass@k measures opportunity under repeated sampling, not single-shot accuracy.

---

## Q22. MUST REMEMBER - Why is pass@k useful?

Some applications can afford several candidate solutions and can verify them automatically. Additional samples can improve the chance of finding one correct trajectory.

Coding is a natural example:

```text
sample multiple programs
    -> run tests
    -> keep any program that passes
```

---

## Q23. MUST REMEMBER - What do `n`, `c`, and `k` mean in the estimator?

```text
n = total generated samples for one problem
c = number of those n samples that are correct
k = hypothetical number of samples allowed at evaluation/use time
```

The estimator asks: if `k` samples were selected without replacement from the observed `n`, what is the probability that at least one is among the `c` correct samples?

---

## Q24. MUST REMEMBER - Derive the pass@k estimator.

Start with the complement:

$$
P(\text{at least one correct})
=
1-P(\text{all }k\text{ are incorrect}).
$$

Among `n` samples, `n-c` are incorrect. The probability that a size-`k` subset contains only incorrect samples is:

$$
\frac{\binom{n-c}{k}}{\binom{n}{k}}.
$$

Therefore:

$$
\boxed{
\widehat{\operatorname{pass@k}}
=
1-
\frac{\binom{n-c}{k}}{\binom{n}{k}}
}
$$

**Memory liner:**

> One minus the fraction of size-`k` subsets that contain only failures.

---

## Q25. MUST REMEMBER - What is the sequential product interpretation?

The probability of selecting `k` incorrect samples without replacement is:

$$
\frac{n-c}{n}
\cdot
\frac{n-c-1}{n-1}
\cdots
\frac{n-c-k+1}{n-k+1}.
$$

Hence:

$$
\widehat{\operatorname{pass@k}}
=
1-
\prod_{j=0}^{k-1}
\frac{n-c-j}{n-j}.
$$

The product and combinatorial forms are equivalent.

---

## Q26. MUST REMEMBER - Show that pass@1 reduces to empirical success rate.

Set `k=1`:

$$
\widehat{\operatorname{pass@1}}
=
1-
\frac{n-c}{n}
=
\frac{c}{n}.
$$

**Memory liner:**

> pass@1 is ordinary single-sample correctness.

---

## Q27. SHOULD KNOW - What if fewer than `k` failures exist?

If:

$$
n-c<k,
$$

then any subset of size `k` must include at least one success, so pass@k is 1. In implementation, the invalid combination term is treated as zero.

---

## Q28. MUST REMEMBER - Why not estimate pass@k by generating only k samples once?

One group of `k` samples has high variance. Generating a larger pool `n` and using the combinatorial estimator uses more evidence to estimate what would happen for any size-`k` subset.

**Memory liner:**

> Generate `n` for estimation; report the implied success chance for budget `k`.

---

## Q29. SHOULD KNOW - Is pass@k the same as best-of-N?

No.

| Concept | What it asks/does |
|---|---|
| pass@k | Evaluation metric: did at least one of k samples pass an objective verifier? |
| best-of-N | Inference method: score N candidates and return the highest-scoring one |

With a perfect verifier, a system can return one of the passing samples, but the concepts are still distinct.

---

## Q30. MUST REMEMBER - How does sampling temperature affect pass@k?

- **Very low temperature:** samples are individually plausible but highly similar; increasing `k` provides little new coverage.
- **Moderate temperature:** preserves quality while adding useful trajectory diversity.
- **Very high temperature:** increases diversity but degrades individual-sample quality.

Therefore the best temperature is usually intermediate and benchmark-dependent.

**Memory liner:**

> pass@k needs diversity, but diversity is useful only while samples remain competent.

---

## Q31. MUST REMEMBER - Why must papers report temperature with pass@k?

`pass@k` depends on the sampling distribution. Two models evaluated with different temperatures, top-p settings, or sample budgets are not directly comparable.

**Interview answer:**

> Report model, prompt, decoding distribution, `n`, `k`, verifier, and number of benchmark problems.

---

## Q32. SHOULD KNOW - What is consensus@k?

Generate `k` reasoning trajectories, extract their final answers, and choose the answer appearing most often.

```text
answers: [42, 42, 39, 42, 41]
consensus -> 42
```

It is closely related to self-consistency.

**Memory liner:**

> pass@k asks whether any sample is right; consensus@k asks which answer the samples agree on.

---

## Q33. SHOULD KNOW - When can consensus@k fail?

A majority can share the same systematic error. Consensus is strongest when:

- samples are meaningfully diverse;
- reasoning errors are not strongly correlated;
- the final-answer extractor is reliable.

---

## Q34. INTERVIEW CLARIFICATION - What is a simple implementation of pass@k?

```python
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    if not (0 <= c <= n):
        raise ValueError("Require 0 <= c <= n")
    if not (1 <= k <= n):
        raise ValueError("Require 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)
```

For large `n`, use a numerically stable product or log-combination implementation rather than materializing huge factorials.

---

# 6. Why reinforcement learning for reasoning?

## Q35. MUST REMEMBER - Why not begin with supervised reasoning-chain training?

The lecture gives three motivations:

1. High-quality long reasoning chains are expensive to write.
2. Human-written reasoning may not match the trajectories most natural for the model.
3. Math and code offer verifiable final-answer rewards, making RL feasible without labelled reasoning traces.

**Memory liner:**

> When trajectories are expensive but outcomes are cheap to verify, optimize outcomes and let trajectories emerge.

---

## Q36. SHOULD KNOW - Does this mean reasoning SFT is useless?

No. The DeepSeek-R1 pipeline later uses cold-start reasoning SFT to improve formatting, readability, and language consistency before more RL.

The lecture's narrower point is:

> Pure SFT is not the only way to bootstrap reasoning when high-quality chains are unavailable.

---

## Q37. MUST REMEMBER - What two simple reward components are introduced first?

```text
1. Format reward
   Did the output contain the required reasoning and answer delimiters?

2. Accuracy reward
   Did the final math answer match, or did the code pass tests?
```

A conceptual total reward is:

$$
r_i
=
\lambda_{fmt}r^{fmt}_i
+
\lambda_{acc}r^{acc}_i.
$$

The lecture mainly presents the idea, not one universal weighting scheme.

---

## Q38. MUST REMEMBER - Why is this called a verifiable reward?

The reward is computed by a deterministic rule, parser, compiler, test suite, or answer checker rather than inferred by a learned reward model.

**Memory liner:**

> Verifiable reward replaces subjective judgement with an executable correctness check.

---

## Q39. SHOULD KNOW - What is the main advantage over learned reward modelling?

You avoid training and trusting a separate model that may mis-rank solutions. The reward is often cheaper, less ambiguous, and easier to audit.

---

## Q40. SHOULD KNOW - What are the limitations of verifiable rewards?

- They apply only where an outcome can be checked.
- The verifier can be incomplete or exploitable.
- They usually supervise the final outcome more strongly than reasoning quality.
- Sparse binary rewards can make exploration difficult.
- Correct final answers can arise from flawed or accidental reasoning.

**INTERVIEW CLARIFICATION:** the reward function and verifier define what behavior training can select; an imperfect verifier creates a form of reward hacking.

---

## Q41. MUST REMEMBER - Why include a format reward if correctness is what matters?

The training pipeline needs a stable machine-readable separation between reasoning and answer. Without a format incentive, the model may omit delimiters or emit an answer that cannot be parsed and verified.

**Memory liner:**

> Formatting is not the task objective, but it makes the objective computable.

---

## Q42. SHOULD KNOW - Can the model game a format reward?

Yes. It can emit the required tags without meaningful reasoning. Therefore format reward must be combined with correctness reward; format alone does not train problem solving.

---

# 7. Controlling how much the model thinks

## Q43. MUST REMEMBER - Why is a dynamic reasoning budget useful?

Different prompts need different amounts of computation.

```text
simple factual query -> little or no reasoning
hard olympiad problem -> long reasoning trajectory
```

A fixed large budget wastes time on easy prompts; a fixed small budget truncates hard ones.

---

## Q44. SHOULD KNOW - What routing idea does the lecture suggest?

Use a lightweight classifier or other controller to estimate whether a prompt needs low or high reasoning effort, then allocate an appropriate budget.

The lecture presents this as an open research direction, not a settled recipe.

---

## Q45. MUST REMEMBER - What is budget forcing?

Budget forcing alters the generated stream to encourage the model to continue or stop reasoning.

Examples described in the lecture:

```text
continue:
    insert a token/phrase such as "Wait"

stop:
    insert a phrase such as "Time is up; now provide the answer"
```

**Memory liner:**

> Budget forcing steers the duration of autoregressive deliberation through the token context.

---

## Q46. SHOULD KNOW - Why might "Wait" induce additional reasoning?

The model has learned textual patterns where "wait" signals reconsideration, correction, or exploration of another path. Appending it changes the prefix and therefore the next-token distribution.

---

## Q47. SHOULD KNOW - What context-window issue arises during long reasoning?

Prompt tokens, reasoning tokens, and final-answer tokens share a finite context window. Excessive thinking can leave insufficient space for the final solution or cause earlier evidence to fall outside the usable context.

**Memory liner:**

> A reasoning budget is also a context-allocation budget.

---

## Q48. LECTURE BOUNDARY - What are continuous thoughts?

The lecture names work where intermediate reasoning is represented by hidden states rather than ordinary natural-language tokens, potentially making thought more compact or expressive.

No exact loss, architecture, or inference algorithm is derived here.

---

# 8. GRPO: the core mental model

## Q49. MUST REMEMBER - What does GRPO stand for?

**Group Relative Policy Optimization.**

It is an RL algorithm that updates a language-model policy using rewards made relative to a group of completions generated for the same prompt.

**Memory liner:**

> GRPO replaces a learned value baseline with a same-prompt group baseline.

---

## Q50. MUST REMEMBER - What is the complete high-level GRPO dataflow?

```text
prompt x
   ↓
current policy pi_theta
   ↓ sample G completions
{o_1, o_2, ..., o_G}
   ↓
verifier / reward function
{r_1, r_2, ..., r_G}
   ↓
normalize rewards within the group
{A_1, A_2, ..., A_G}
   ↓
PPO-style clipped token-probability objective
   +
reference-model KL penalty
   ↓
update pi_theta only
```

---

## Q51. MUST REMEMBER - What makes the group "relative"?

A completion is judged relative to other completions for the **same prompt**, not only by its absolute reward.

A correct answer to a difficult problem can be exceptional within its group; the same raw reward may be ordinary on an easy problem where nearly every sample succeeds.

**Memory liner:**

> Relative advantage puts a completion's reward in the context of that prompt's sampled difficulty.

---

## Q52. MUST REMEMBER - What group advantage formula is presented?

For one prompt with rewards `r_1, ..., r_G`:

$$
\bar r
=
\frac{1}{G}
\sum_{j=1}^{G}r_j,
$$

$$
s_r
=
\sqrt{
\frac{1}{G}
\sum_{j=1}^{G}(r_j-\bar r)^2
},
$$

and:

$$
\boxed{
A_i
=
\frac{r_i-\bar r}{s_r+\delta}
}
$$

where `delta` is a small numerical-stability term in implementation.

**Memory liner:**

> Better than the group mean gives positive advantage; worse gives negative advantage.

---

## Q53. MUST REMEMBER - What happens when a completion's reward is above the group mean?

$$
r_i>\bar r
\quad\Rightarrow\quad
A_i>0.
$$

The policy update tries to increase the probability of the sampled completion tokens, subject to clipping and reference regularization.

---

## Q54. MUST REMEMBER - What happens when a completion's reward is below the group mean?

$$
r_i<\bar r
\quad\Rightarrow\quad
A_i<0.
$$

The policy update tries to reduce the probability of the sampled completion tokens, again under constrained updates.

---

## Q55. MUST REMEMBER - Why does GRPO not need a value model?

The group mean and standard deviation provide a prompt-specific baseline directly from sampled completions. PPO ordinarily learns a value function to estimate expected future reward and then constructs advantages using reward and value estimates.

**Memory liner:**

> PPO learns the baseline; GRPO samples the baseline.

---

## Q56. SHOULD KNOW - What does GRPO pay for after removing the value model?

It must generate several completions per prompt.

```text
saved:
    value-model memory and training

added:
    G rollouts per prompt
    G reward evaluations
```

Thus GRPO is simpler in model state but potentially expensive in rollout generation.

---

## Q57. MUST REMEMBER - What is the token probability ratio?

For token `o_{i,t}`:

$$
\boxed{
\rho_{i,t}(\theta)
=
\frac{
\pi_\theta(o_{i,t}\mid x,o_{i,<t})
}{
\pi_{old}(o_{i,t}\mid x,o_{i,<t})
}
}
$$

Interpretation:

- `rho = 1`: current and old policies assign the same probability;
- `rho > 1`: current policy increased the token probability;
- `rho < 1`: current policy decreased it.

**Memory liner:**

> The ratio measures how much the policy changed on the sampled action.

---

## Q58. MUST REMEMBER - Why is `pi_old` needed?

`pi_old` is a frozen snapshot used to stabilize an update epoch. It provides the denominator for measuring how far the current policy has moved from the behavior policy that generated the rollouts.

Do not confuse it with `pi_ref`.

---

## Q59. MUST REMEMBER - `pi_old` versus `pi_ref`?

| Model | Meaning | Main role |
|---|---|---|
| `pi_old` | Policy snapshot from the previous/current rollout collection step | Local update stability and probability ratio |
| `pi_ref` | Frozen starting checkpoint, typically SFT/base model | Prevent global drift from the capable starting distribution |

**Memory liner:**

> Old controls step size; reference controls destination drift.

---

## Q60. MUST REMEMBER - What is the clipped surrogate term?

**INTERVIEW CLARIFICATION:** a standard token-level PPO/GRPO term is:

$$
L^{clip}_{i,t}(\theta)
=
\min\left(
\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i
\right).
$$

For positive advantage, clipping prevents excessive probability increases. For negative advantage, it prevents excessive probability decreases.

**Memory liner:**

> Reward or punish the sampled token, but cap how aggressively one update can change it.

---

## Q61. MUST REMEMBER - What role does KL regularization play?

It discourages the trainable policy from drifting too far from the frozen reference model.

Conceptually:

$$
J
=
\text{reward-oriented policy objective}
-
\beta D_{KL}(\pi_\theta\Vert\pi_{ref}).
$$

The reference preserves broad language ability, useful style, and behavior not captured by the sparse reasoning reward.

**Memory liner:**

> Optimize reasoning without forgetting how to be a good language model.

---

## Q62. SHOULD KNOW - Why is the KL coefficient `beta` important?

- Too small: policy may exploit the verifier, drift, or lose general behavior.
- Too large: policy remains too close to the reference and learns little new reasoning behavior.

**Memory liner:**

> `beta` trades reward improvement against distributional conservatism.

---

## Q63. MUST REMEMBER - Write a lecture-consistent GRPO objective.

**INTERVIEW CLARIFICATION:** one common form consistent with the lecture is:

$$
\boxed{
J_{GRPO}(\theta)
=
\mathbb{E}\left[
\frac{1}{G}
\sum_{i=1}^{G}
\frac{1}{T_i}
\sum_{t=1}^{T_i}
\left(
\min\left[
\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i
\right]
-
\beta D_{KL,i,t}
\right)
\right]
}
$$

Training code normally minimizes `-J_GRPO`.

The lecture later critiques the `1/T_i` normalization.

---

## Q64. MUST REMEMBER - What is shared across all tokens in one completion?

The completion-level group advantage `A_i` is commonly broadcast across its sampled tokens:

```text
completion i reward:        scalar r_i
completion i advantage:     scalar A_i
completion token logprobs:  [T_i]

A_i broadcasts over all T_i tokens
```

The token-specific part is the probability ratio and KL term.

---

## Q65. SHOULD KNOW - Why is completion-level reward applied through token log-probabilities?

The completion probability factorizes autoregressively:

$$
\log \pi_\theta(o_i\mid x)
=
\sum_{t=1}^{T_i}
\log \pi_\theta(o_{i,t}\mid x,o_{i,<t}).
$$

Increasing a successful completion's likelihood requires updating the token decisions that produced it.

---

# 9. GRPO tensor shapes

## Q66. MUST REMEMBER - What are the central GRPO tensor shapes?

Assume padded group completions:

```text
prompt token IDs:          [B, T_p]
completion token IDs:      [B, G, T_o]
completion mask:           [B, G, T_o]

current token logprobs:    [B, G, T_o]
old token logprobs:        [B, G, T_o]
reference token logprobs:  [B, G, T_o]

completion rewards:        [B, G]
group reward mean:         [B, 1]
group reward std:          [B, 1]
completion advantages:     [B, G]
broadcast advantages:      [B, G, 1] -> [B, G, T_o]

probability ratios:        [B, G, T_o]
per-token surrogate:       [B, G, T_o]
masked objective:          scalar
```

**Memory liner:**

> Rewards vary over completions; policy terms vary over completion tokens.

---

## Q67. MUST REMEMBER - How are token log-probabilities gathered?

Model logits:

```text
logits:              [B, G, T_o, V]
log_softmax(logits): [B, G, T_o, V]
completion IDs:      [B, G, T_o]
```

Gather the log-probability of the actually sampled token:

```text
sampled logprobs:    [B, G, T_o]
```

---

## Q68. SHOULD KNOW - Why is a completion mask necessary?

Completions have different lengths. Padding tokens must not contribute to:

- policy loss;
- KL penalty;
- length normalization;
- token counts.

```text
M_{i,t}=1  real completion token
M_{i,t}=0  padding token
```

---

## Q69. INTERVIEW CLARIFICATION - What is a vectorized group advantage computation?

```python
# rewards: [B, G]
mean = rewards.mean(dim=1, keepdim=True)                  # [B, 1]
std = rewards.std(dim=1, keepdim=True, unbiased=False)   # [B, 1]
advantages = (rewards - mean) / (std + 1e-6)             # [B, G]
```

Do not normalize across unrelated prompts if the method is meant to use same-prompt group relativity.

---

# 10. PPO versus GRPO

## Q70. MUST REMEMBER - What is the most important difference?

```text
PPO:
    reward + learned value function -> advantage

GRPO:
    group of same-prompt rewards -> relative advantage
```

**Memory liner:**

> GRPO removes the value function by comparing sibling completions.

---

## Q71. MUST REMEMBER - Which models are trained and frozen?

### Reasoning GRPO with verifiable rewards

```text
trainable:
    policy model

frozen:
    old policy snapshot during update
    reference model

not required:
    learned reward model
    value model
```

### PPO-based RLHF

```text
trainable:
    policy model
    value model

frozen:
    reward model
    reference model
    old policy snapshot during update
```

The exact engineering can share backbones, but the conceptual roles remain distinct.

---

## Q72. MUST REMEMBER - Why is there no reward model in the lecture's reasoning setup?

Math and code rewards are computed by verifiers. The reward component is a program/rule rather than a learned neural scorer.

**Memory liner:**

> Replace subjective preference prediction with objective outcome checking.

---

## Q73. SHOULD KNOW - How does PPO construct token-level training signals in the lecture's walkthrough?

The lecture describes:

- a terminal completion reward, attached at the end;
- per-token KL-related shaping;
- a per-token value estimate;
- generalized advantage estimation to produce token-level advantages.

GRPO instead broadcasts a group-derived completion advantage across tokens.

---

## Q74. LECTURE BOUNDARY - Is the GAE equation required here?

No. The lecture explicitly treats generalized advantage estimation as a complicated formula that is not derived.

Remember its purpose:

> Combine observed rewards and value estimates to estimate how much better each action was than expected.

---

## Q75. MUST REMEMBER - What similarities do PPO and GRPO retain?

Both use:

- policy-generated rollouts;
- current-to-old probability ratios;
- clipping to constrain updates;
- a reference-policy constraint in modern LLM use;
- advantages that determine whether sampled actions should be upweighted or downweighted.

---

## Q76. SHOULD KNOW - Where does the lecture place the KL term?

The lecture contrasts:

- GRPO: KL appears explicitly in the objective;
- the described PPO implementation: KL can be folded into per-token shaped rewards used in advantage estimation.

This is an implementation/formulation distinction, not a claim that only one arrangement is possible.

---

## Q77. MUST REMEMBER - PPO versus GRPO comparison table

| Question | PPO in RLHF | GRPO in reasoning |
|---|---|---|
| Multiple completions per prompt? | Not required by the basic walkthrough | Yes, needed for group baseline |
| Reward source | Usually learned reward model | Often executable verifier |
| Value model | Yes | No |
| Advantage | Reward/value via GAE | Standardized same-prompt group reward |
| Policy ratio and clipping | Yes | Yes |
| Reference KL | Yes in modern LLM formulations | Yes |
| Main burden | Multiple models and value learning | Rollout group generation and group design |

---

## Q78. SHOULD KNOW - Is GRPO automatically cheaper than PPO?

Not universally.

It saves value-model parameters, memory, optimization, and inference. But it generates `G` completions per prompt, which can dominate cost for long reasoning traces.

**Memory liner:**

> Fewer models does not necessarily mean fewer generated tokens.

---

# 11. A PyTorch-like GRPO loss mental model

## Q79. INTERVIEW CLARIFICATION - What is minimal pseudocode for the clipped part?

```python
# Shapes:
# new_logp, old_logp, ref_logp, mask: [B, G, T]
# rewards:                            [B, G]

mean = rewards.mean(dim=1, keepdim=True)
std = rewards.std(dim=1, keepdim=True, unbiased=False)
adv = (rewards - mean) / (std + 1e-6)                    # [B, G]
adv = adv.unsqueeze(-1)                                  # [B, G, 1]

ratio = torch.exp(new_logp - old_logp)                   # [B, G, T]
ratio_clip = ratio.clamp(1.0 - eps, 1.0 + eps)

surrogate_1 = ratio * adv
surrogate_2 = ratio_clip * adv
policy_term = torch.minimum(surrogate_1, surrogate_2)    # [B, G, T]

# Conceptual sampled-token reference penalty.
# Exact KL estimators vary across implementations.
kl_proxy = new_logp - ref_logp                           # [B, G, T]
objective_per_token = policy_term - beta * kl_proxy

length = mask.sum(dim=-1).clamp_min(1)                   # [B, G]
per_completion = (objective_per_token * mask).sum(-1) / length
objective = per_completion.mean()
loss = -objective
```

The lecture's length-bias section explains why the `divide by length` line deserves scrutiny.

---

## Q80. SHOULD KNOW - Why is `exp(new_logp-old_logp)` used?

Because:

$$
\exp(\log p_{new}-\log p_{old})
=
\frac{p_{new}}{p_{old}}.
$$

Computing in log space is numerically stable and matches language-model outputs.

---

## Q81. SHOULD KNOW - Why must rollout tokens be generated by the old/current behavior policy?

The probability ratio assumes the data came from the denominator policy. After rollouts are collected, the old log-probabilities are fixed while several optimization steps may update `pi_theta`.

This is the on-policy / near-on-policy logic inherited from PPO-style methods.

---

# 12. GRPO length bias

## Q82. MUST REMEMBER - What empirical pattern motivates the length-bias discussion?

During reasoning RL:

```text
RL training step increases
    -> average reasoning length increases
    -> benchmark performance initially improves
    -> later performance plateaus
    -> length may continue increasing
```

The final region indicates inefficient overthinking rather than useful additional computation.

---

## Q83. MUST REMEMBER - Where does the bias arise in vanilla GRPO?

The objective averages token contributions **separately inside each completion**:

$$
\frac{1}{T_i}
\sum_{t=1}^{T_i}\ell_{i,t}.
$$

Therefore each token in completion `i` receives an effective factor proportional to:

$$
\frac{1}{T_i}.
$$

Tokens in shorter responses have larger per-token weight.

**Memory liner:**

> Per-response averaging makes one token in a short answer count more than one token in a long answer.

---

## Q84. MUST REMEMBER - Why is this especially problematic for negative advantages?

For `A_i < 0`, the sampled completion is being discouraged. Because short responses have larger per-token magnitude, a short bad response is penalized more strongly per token than a long bad response.

```text
short incorrect response
    -> large 1/T_i
    -> strong negative update per token

long incorrect response
    -> small 1/T_i
    -> weaker negative update per token
```

This creates a relative incentive for bad responses to become longer.

**Memory liner:**

> Vanilla normalization can make "wrong and long" less punished than "wrong and short."

---

## Q85. MUST REMEMBER - What happens for positive advantages?

For `A_i > 0`, tokens in a short successful response are reinforced more strongly per token than tokens in a long successful response.

The problematic global length growth is most intuitively explained by the asymmetry on negatively rewarded responses: long failures receive diluted penalties.

---

## Q86. MUST REMEMBER - Why can the total per-completion average hide the issue?

Averaging makes every completion contribute a similar aggregate scale, which sounds fair. But policy learning occurs through token log-probabilities, and the gradient weight assigned to each token differs as `1/T_i`.

**Memory liner:**

> Equal response weight is not equal token weight.

---

## Q87. SHOULD KNOW - What correction does the lecture attribute to DAPO-like work?

The lecture describes a method that equalizes token-level contributions using a normalization shared across tokens rather than separately dividing each response by its own length.

A conceptual global-token normalization is:

$$
\frac{
\sum_i\sum_t M_{i,t}\ell_{i,t}
}{
\sum_i\sum_t M_{i,t}
}.
$$

**INTERVIEW CLARIFICATION:** exact DAPO implementations include additional changes; this equation captures only the lecture's token-level normalization intuition.

---

## Q88. SHOULD KNOW - What correction does Dr.GRPO propose at the lecture's level?

The lecture says Dr.GRPO removes the problematic per-response length factor rather than allowing each output length to rescale all of its token contributions.

The exact full objective is not derived in the transcript.

---

## Q89. MUST REMEMBER - What empirical effect do these corrections target?

They reduce unnecessary growth in incorrect-response length while preserving useful reasoning length for correct solutions.

```text
correct solutions:
    reasoning length can remain comparable

incorrect solutions:
    avoid endlessly longer failed traces
```

**Memory liner:**

> Preserve useful deliberation; remove the incentive for verbose failure.

---

## Q90. SHOULD KNOW - How would you diagnose length bias in an experiment?

Track at least:

- mean and quantiles of completion length;
- accuracy/reward versus training step;
- length separated by correct versus incorrect completions;
- reward versus length scatter/curves;
- token-normalized and response-normalized objectives;
- KL drift and entropy.

A red flag is length increasing after correctness saturates, especially mainly among incorrect samples.

---

# 13. Group standard-deviation bias

## Q91. MUST REMEMBER - What role does the group standard deviation play?

$$
A_i
=
\frac{r_i-\bar r}{s_r+\delta}.
$$

It normalizes advantage scale across groups. But groups with different reward dispersion receive different effective scaling.

---

## Q92. SHOULD KNOW - Why can difficulty interact badly with this normalization?

For a very easy or very hard prompt, group rewards may be nearly identical:

```text
all correct  -> very low standard deviation
all wrong    -> very low standard deviation
```

This can produce:

- no useful relative signal when all rewards are exactly equal;
- unstable or disproportionate scaling when variance is tiny but nonzero;
- a dependence of gradient scale on group reward dispersion rather than only learning value.

The lecture identifies this as another target for later GRPO modifications.

---

## Q93. MUST REMEMBER - What if all group rewards are identical?

Then:

$$
r_i-\bar r=0
$$

for every completion. With stable implementation, all advantages should effectively be zero, so that prompt provides no policy-gradient learning signal.

**Memory liner:**

> A relative method cannot rank a group with no reward differences.

---

## Q94. SHOULD KNOW - How can group size affect the advantage estimate?

Small `G` gives a noisy estimate of the prompt-specific reward distribution. Larger `G` improves relative comparison but increases rollout cost.

This is a core GRPO trade-off:

```text
larger group
    -> better within-prompt baseline and exploration
    -> more generation cost
```

---

# 14. Asymmetric clipping

## Q95. MUST REMEMBER - What limitation of symmetric clipping does the lecture identify?

The standard upper ratio bound is:

$$
\rho_{i,t}\leq 1+\epsilon.
$$

Therefore the largest allowed new probability is roughly:

$$
p_{new}
\leq
(1+\epsilon)p_{old}.
$$

If `p_old` is extremely small, the absolute allowed increase is also extremely small. A useful but initially unlikely token may recover too slowly.

**Memory liner:**

> Multiplicative clipping barely moves a probability that starts near zero.

---

## Q96. SHOULD KNOW - Why not simply use a very large symmetric epsilon?

A large lower-side allowance could let common/high-probability tokens collapse too aggressively. The lecture motivates different lower and upper clipping widths.

Conceptually:

$$
\rho
\in
[1-\epsilon_{low},
 1+\epsilon_{high}],
$$

with potentially:

$$
\epsilon_{high}>\epsilon_{low}.
$$

This gives low-probability actions more room to grow without equally increasing the room for destructive decreases.

---

## Q97. SHOULD KNOW - What general lesson should an interviewer hear?

> Relative objectives, normalizers, and clipping rules are not neutral bookkeeping. They create concrete incentives over length, difficulty, exploration, and probability mass.

---

# 15. DeepSeek-R1-Zero: what pure reasoning RL demonstrates

## Q98. MUST REMEMBER - What is the purpose of DeepSeek-R1-Zero in the lecture?

It is a proof of concept showing that substantial reasoning behavior can emerge by applying RL with verifiable rewards directly to a pre-trained base model, without first supplying supervised reasoning traces.

**Memory liner:**

> R1-Zero tests whether outcome-based RL can bootstrap reasoning from a base LM.

---

## Q99. MUST REMEMBER - What model does R1-Zero start from?

The lecture says it begins from the DeepSeek-V3 base model:

```text
pre-trained decoder-only base LM
    + modern architecture components such as MoE and MLA
    - no reasoning SFT before the reasoning RL stage
```

The architectural details are background; the lecture's main point is the absence of prior reasoning supervision.

---

## Q100. MUST REMEMBER - What rewards are used in the R1-Zero description?

1. **Accuracy / correctness reward** for the final answer.
2. **Formatting reward** for placing reasoning and answer in required delimiters.

```text
<think> ... reasoning ... </think>
<answer> ... final answer ... </answer>
```

The prompt template explains the required format in plain text.

---

## Q101. MUST REMEMBER - What does the R1-Zero result demonstrate?

As reasoning RL proceeds, benchmark accuracy rises even without supervised reasoning chains. This supports the claim that verifiable outcome rewards can induce useful multi-step behavior.

**Memory liner:**

> Correctness reward can select reasoning trajectories even when the trajectory text was never labelled by humans.

---

## Q102. SHOULD KNOW - What kinds of behavior can emerge during RL?

The lecture points to reasoning traces containing reconsideration patterns such as "wait" and alternative attempts. These indicate that autoregressive self-correction patterns can be reinforced when they lead to higher reward.

Do not over-interpret the token "wait" itself as a formal cognitive module; it is a learned sequence pattern.

---

## Q103. MUST REMEMBER - What are the main R1-Zero problems described?

- language mixing inside reasoning traces;
- syntax/formatting problems;
- poor readability;
- weak human-facing polish.

The policy is free to discover any reward-achieving token pattern because it lacks a strong supervised anchor for readable reasoning.

**Memory liner:**

> Pure outcome RL can discover capability without producing clean human-facing traces.

---

## Q104. SHOULD KNOW - Why can final-answer reward fail to enforce readable reasoning?

The verifier observes whether the answer is correct, not whether the reasoning is elegant, monolingual, or understandable. Many internal trajectories may obtain the same terminal reward.

**Interview lesson:** optimization only constrains properties represented in the reward.

---

# 16. DeepSeek-R1: the full training pipeline

## Q105. MUST REMEMBER - What problem does the full R1 recipe solve?

It preserves the reasoning gains demonstrated by R1-Zero while improving:

- readability;
- language consistency;
- formatting;
- general assistant behavior;
- helpfulness and harmlessness.

---

## Q106. MUST REMEMBER - What are the main R1 stages in the lecture?

```text
DeepSeek-V3 base model
        ↓
1. Cold-start reasoning SFT
        ↓
2. Reasoning-oriented RL
        ↓
3. Rejection sampling + large mixed SFT
        ↓
4. Final mixed RL for reasoning + helpfulness + harmlessness
        ↓
DeepSeek-R1
```

**Memory liner:**

> Small clean supervision anchors the format, RL grows reasoning, filtered data broadens behavior, and final RL aligns the complete assistant.

---

## Q107. MUST REMEMBER - What is cold-start data?

A relatively small set of high-quality reasoning examples used before large-scale reasoning RL. The lecture says humans rewrite or clean reasoning traces so they have:

- consistent formatting;
- readable syntax;
- consistent target language;
- coherent chain-of-thought presentation.

The resulting prompt-response pairs are used for SFT.

---

## Q108. SHOULD KNOW - Why is the cold-start set much smaller than later stages?

Its purpose is not to teach every reasoning problem. It anchors the policy in a clean output manifold so subsequent RL explores from a readable, well-formatted starting point.

**Memory liner:**

> Cold-start SFT teaches how reasoning should look; RL teaches which reasoning succeeds.

---

## Q109. MUST REMEMBER - What rewards are used in the next reasoning RL stage?

The lecture describes:

- verifiable answer reward;
- formatting reward;
- language-consistency reward.

A conceptual combination is:

$$
r
=
\lambda_{acc}r_{acc}
+
\lambda_{fmt}r_{fmt}
+
\lambda_{lang}r_{lang}.
$$

The exact weights are not derived in the lecture.

---

## Q110. MUST REMEMBER - What is the language-consistency reward?

The lecture describes a simple heuristic based on the fraction of output tokens belonging to the target language. It discourages mixed-language reasoning traces.

**Memory liner:**

> Add a reward for staying in the intended language because correctness alone does not enforce it.

---

## Q111. SHOULD KNOW - What are the risks of a language-consistency heuristic?

**INTERVIEW CLARIFICATION:**

- code, formulas, names, and borrowed terms may be misclassified;
- maximizing token purity may produce unnatural wording;
- the heuristic does not measure reasoning quality;
- tokenizer-level language detection can be noisy.

It should be viewed as a targeted regularizer, not the main reasoning objective.

---

## Q112. MUST REMEMBER - What happens after the first reasoning RL stage?

The model generates a larger reasoning dataset. Candidate solutions are filtered so only high-quality responses are retained. This is described as **rejection sampling**.

```text
reasoning prompts
    ↓ current reasoning model
many candidate responses
    ↓ correctness / formatting / judge filters
accepted high-quality traces
    ↓
SFT dataset
```

---

## Q113. MUST REMEMBER - What does rejection sampling mean here?

Generate candidate completions, reject those that fail correctness or quality criteria, and retain the accepted completions as supervised training targets.

It is not the same object as the exact-distribution rejection sampling used in speculative decoding, though both use accept/reject language.

**Memory liner:**

> Turn a strong but imperfect generator into a data producer by filtering its outputs.

---

## Q114. SHOULD KNOW - Why use SFT again after RL?

The accepted reasoning traces become dense token-level supervision. SFT can:

- consolidate successful trajectories;
- improve stability;
- expose the model repeatedly to clean outputs;
- combine reasoning behavior with general assistant data.

**Memory liner:**

> RL discovers successful behavior; filtered SFT distills it into a stable next-token objective.

---

## Q115. MUST REMEMBER - What data is mixed in the larger SFT stage?

The lecture describes:

- a large set of reasoning prompt-response pairs generated and filtered from the current model;
- non-reasoning instruction data recycled from the general V3 assistant pipeline.

It reports roughly 200k non-reasoning pairs and a reasoning-to-non-reasoning mixture around 3:1.

The exact counts should be treated as lecture-reported orders/ratios, not universal design rules.

---

## Q116. MUST REMEMBER - Why include non-reasoning data?

A useful assistant must not turn every request into a long math-style derivation. General instruction data preserves:

- ordinary question answering;
- writing and dialogue;
- concise responses;
- broad domains;
- user-facing assistant behavior.

**Memory liner:**

> Reasoning specialization should not erase ordinary assistant competence.

---

## Q117. MUST REMEMBER - What is the final RL stage intended to optimize?

It combines:

- reasoning rewards for reasoning tasks;
- helpfulness rewards for user-visible answers;
- harmlessness/safety rewards for the entire generated output.

The lecture notes that harmlessness should apply to the whole output, including the reasoning portion, while helpfulness is focused more on what the user receives.

---

## Q118. SHOULD KNOW - Why apply harmlessness to hidden reasoning tokens too?

Even if intermediate text is not displayed, unsafe internal trajectories may leak, influence the answer, or be exposed in another interface. The lecture frames safety as covering the full output trajectory.

---

## Q119. MUST REMEMBER - Give a one-line R1 versus R1-Zero distinction.

> R1-Zero is RL-from-base proof of concept; R1 is a multi-stage system that adds clean supervision, filtered data, general capabilities, and final alignment.

---

## Q120. SHOULD KNOW - Why is the R1 recipe iterative rather than one-shot?

Each stage creates a better data generator or policy for the next stage:

```text
clean traces
    -> better RL exploration
    -> stronger reasoning generator
    -> better filtered SFT data
    -> better final aligned policy
```

**Memory liner:**

> Training alternates between policy improvement and converting improved behavior into cleaner supervision.

---

# 17. Reasoning distillation

## Q121. MUST REMEMBER - What problem does reasoning distillation solve?

A very large reasoning model may be too expensive to deploy. Distillation transfers its behavior to a smaller student.

```text
large reasoning teacher
    -> generate reasoning traces
    -> train smaller student
```

---

## Q122. MUST REMEMBER - How does the lecture's reasoning distillation differ from soft-logit distillation?

### Earlier soft-target distillation

```text
teacher next-token probability distribution
    -> student matches full distribution
```

### Reasoning-trace distillation in this lecture

```text
teacher generates complete reasoning + answer sequences
    -> store sequences offline
    -> student performs SFT on those token sequences
```

**Memory liner:**

> Here the teacher transfers trajectories as data, not necessarily logits at every step.

---

## Q123. MUST REMEMBER - What is the reasoning-distillation data format?

```text
prompt x
teacher output:
    <think>
    reasoning tokens ...
    </think>
    <answer>
    final answer
    </answer>
```

The student is trained with next-token loss on the teacher output, normally masking prompt tokens from the response loss as in SFT.

---

## Q124. SHOULD KNOW - Why can trace distillation be effective?

The teacher supplies intermediate targets that reveal a successful solution path. The student no longer has to discover the behavior only through sparse terminal reward.

**Memory liner:**

> RL searches for trajectories; distillation turns found trajectories into dense supervision.

---

## Q125. MUST REMEMBER - What finding about small models is emphasized?

The lecture reports that, at smaller model scales, distilling reasoning traces from a strong teacher can perform better than trying to reproduce the same reasoning capability through RL from scratch.

---

## Q126. SHOULD KNOW - What are the risks of reasoning distillation?

**INTERVIEW CLARIFICATION:**

- teacher errors and stylistic quirks are copied;
- trace diversity may be too low;
- student may imitate verbose reasoning without equivalent internal capability;
- filtered data can narrow behavior;
- student capacity may limit faithful transfer.

---

## Q127. SHOULD KNOW - How would you improve the distilled dataset?

Possible lecture-consistent strategies:

- generate multiple teacher traces per prompt;
- verify final answers;
- reject malformed or low-quality traces;
- retain diverse successful trajectories;
- mix reasoning and ordinary instruction data;
- balance problem difficulty and domains.

---

# 18. Complete reasoning-training dataflows

## Q128. MUST REMEMBER - What is the minimal RL-from-verifier loop?

```text
for each prompt x:
    sample G completions from pi_old
    verify format and final answer
    produce rewards r_1 ... r_G
    normalize rewards within the prompt group
    compute current / old token probability ratios
    compute clipped objective and reference KL
    update pi_theta
```

---

## Q129. INTERVIEW CLARIFICATION - What is high-level GRPO pseudocode?

```python
for prompts, ground_truth in dataloader:
    with torch.no_grad():
        completions = old_policy.generate(
            prompts,
            num_return_sequences=group_size,
            do_sample=True,
            temperature=temperature,
        )

        rewards = verifier(
            prompts=prompts,
            completions=completions,
            ground_truth=ground_truth,
        )  # [B, G]

        old_logp = token_logprobs(old_policy, prompts, completions)
        ref_logp = token_logprobs(reference_model, prompts, completions)

    new_logp = token_logprobs(policy, prompts, completions)

    advantages = group_normalize(rewards)  # [B, G]
    loss = grpo_loss(
        new_logp=new_logp,
        old_logp=old_logp,
        ref_logp=ref_logp,
        advantages=advantages,
        completion_mask=completions.mask,
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

`old_policy` is periodically refreshed from `policy` according to the update scheme.

---

## Q130. SHOULD KNOW - What should the verifier return?

At minimum, a scalar reward per completion:

```text
[B, G]
```

For debugging, retain component metrics separately:

```text
accuracy_reward:       [B, G]
format_reward:         [B, G]
language_reward:       [B, G]
execution_status:      [B, G]
parsed_answer_valid:   [B, G]
```

Do not keep only the weighted sum; component logs expose reward exploitation.

---

## Q131. SHOULD KNOW - Why store component rewards?

A rising total reward can hide regressions. For example:

- format reward rises while correctness stalls;
- language consistency rises while answer quality falls;
- output length exploits an objective normalizer;
- verifier errors appear as false successes.

**Memory liner:**

> Never debug a composite reward using only the composite scalar.

---

## Q132. MUST REMEMBER - What is the SFT consolidation loop after rejection sampling?

```text
strong reasoning policy
    -> sample many traces
    -> verify and filter
    -> create fixed prompt-response dataset
    -> ordinary response-only SFT
```

Tensor flow:

```text
input IDs:       [B, T_p + T_o]
labels:          [B, T_p + T_o]
prompt labels:   ignore index
response labels: teacher-generated token IDs
logits:          [B, T_p + T_o, V]
loss:            cross-entropy on response tokens
```

---

# 19. Failure modes and debugging questions

## Q133. MUST REMEMBER - Reward rises but accuracy does not. What do you inspect?

1. Reward-component weights.
2. Parser and verifier correctness.
3. Format-reward exploitation.
4. KL scale and policy drift.
5. Completion length and truncation.
6. Duplicate completions and low exploration.
7. Train-evaluation benchmark mismatch.

**Memory liner:**

> A model improves the reward you wrote, not the outcome you intended.

---

## Q134. MUST REMEMBER - Every completion in a group is identical. What happens?

- Rewards are identical or nearly identical.
- Relative advantages are zero or uninformative.
- `pass@k` gains disappear.
- GRPO receives little learning signal.

Potential causes:

- temperature too low;
- deterministic decoding;
- mode collapse;
- prompt overly constraining;
- repeated random seeds.

---

## Q135. MUST REMEMBER - All rewards in a group are zero. What does GRPO learn?

With pure group-relative normalization, nothing useful from that group because all centered rewards are zero.

Possible responses:

- improve exploration;
- increase group size;
- use curriculum / easier prompts;
- use denser partial rewards if valid;
- improve the base model or cold start.

---

## Q136. SHOULD KNOW - All rewards in a group are one. What happens?

Again, relative advantages are zero. The prompt confirms success but does not distinguish which sampled trajectory is better.

This suggests the problem may be too easy for the current policy or the reward too coarse.

---

## Q137. MUST REMEMBER - Training reward improves while KL explodes. Interpretation?

The policy is moving far from the reference. It may be:

- over-optimizing a narrow verifier;
- forgetting general behavior;
- exploiting reward loopholes;
- becoming syntactically or stylistically degraded.

Check `beta`, learning rate, clipping, and reward design.

---

## Q138. MUST REMEMBER - Reasoning length grows but pass@1 plateaus. What do you suspect?

- vanilla per-response normalization length bias;
- overthinking;
- insufficient stop incentive;
- reward not charging for compute;
- malformed termination behavior;
- training distribution rewarding verbose templates.

Plot correct and incorrect lengths separately.

---

## Q139. SHOULD KNOW - pass@k improves but pass@1 does not. What does it mean?

The policy has useful solutions in its sampling distribution but does not place enough mass on them for single-shot reliability.

Potential next steps:

- improve policy training;
- use self-consistency or verification at inference;
- distill selected successful traces;
- adjust sampling and curriculum.

**Memory liner:**

> Search can find the answer, but the base distribution does not choose it reliably.

---

## Q140. SHOULD KNOW - pass@1 improves but pass@k saturates early. What might it mean?

The model is strong but samples are highly correlated. Additional sampling produces near-duplicates rather than new approaches.

Consider moderate temperature or diversity-oriented sampling.

---

## Q141. SHOULD KNOW - The model emits correct answers but malformed tags. What should change?

- validate the prompt template;
- strengthen or rebalance format reward;
- add clean formatting SFT;
- ensure stop tokens and parser rules agree;
- log formatting and correctness independently.

Do not discard correct reasoning behavior by treating all formatting failures as the same conceptual failure during debugging.

---

## Q142. SHOULD KNOW - The model mixes languages. What interventions are lecture-grounded?

- cold-start SFT with consistent-language traces;
- a language-consistency reward;
- rejection sampling to remove mixed traces;
- later SFT on filtered outputs.

---

## Q143. SHOULD KNOW - RL from scratch is unstable on a small model. What does the lecture suggest?

Use reasoning distillation from a larger teacher. The lecture reports that smaller models benefit more from learning verified teacher traces than rediscovering reasoning solely through RL.

---

## Q144. SHOULD KNOW - Why can a correct final answer still be a poor training example?

The trace may be:

- accidental;
- unreadable;
- unsafe;
- language-mixed;
- excessively long;
- based on a brittle shortcut.

The R1 pipeline therefore combines correctness verification with formatting, language, filtering, and later alignment.

---

# 20. Key distinctions interviewers may probe

## Q145. MUST REMEMBER - Chain-of-thought prompting versus reasoning-model training?

| Chain-of-thought prompting | Reasoning-model training |
|---|---|
| Elicits steps through instructions/examples | Changes model parameters to favor useful reasoning trajectories |
| No weight update | Uses SFT, RL, or both |
| Behavior can be brittle to prompt | Behavior is internalized in the policy distribution |

---

## Q146. MUST REMEMBER - Outcome supervision versus process supervision?

### Outcome supervision in this lecture

Reward depends mainly on final correctness.

### Process supervision

Would assess intermediate steps directly.

**LECTURE BOUNDARY:** process-reward models and step-level supervision are not developed in this transcript.

**Memory liner:**

> Outcome reward asks whether the solution ended correctly; process reward asks whether the path was correct.

---

## Q147. MUST REMEMBER - Verifier versus reward model?

| Verifier | Learned reward model |
|---|---|
| Program/rule/test/ground-truth checker | Neural model trained from preference labels |
| Usually objective and domain-specific | Can evaluate subjective dimensions |
| Cheap and auditable when available | Broader but imperfect and hackable |
| No training required | Requires reward-model training data |

---

## Q148. MUST REMEMBER - Reasoning RL versus preference RLHF?

```text
Preference RLHF:
    reward approximates human preference
    commonly PPO + learned reward model

Reasoning RL:
    reward often checks objective correctness
    commonly GRPO + verifiable reward
```

Both constrain policy drift and optimize sampled completions, but their reward source and training machinery differ.

---

## Q149. MUST REMEMBER - GRPO versus DPO?

This lecture focuses on GRPO. From the previous lecture:

```text
DPO:
    offline chosen/rejected pairs
    direct supervised preference loss
    no online rollout group or verifier needed per step

GRPO:
    online sampled completion groups
    scalar/verifiable rewards
    group-relative advantages
    PPO-style clipped policy update
```

**Memory liner:**

> DPO learns from stored preference pairs; GRPO learns from current-policy outcome-scored rollouts.

---

## Q150. MUST REMEMBER - GRPO versus best-of-N?

```text
Best-of-N:
    do not change model weights
    generate N, score N, return best
    pushes cost to every inference request

GRPO:
    use groups during training
    update policy so better behavior becomes more likely
    aims to improve future single/multi-sample generation
```

---

## Q151. MUST REMEMBER - pass@k versus consensus@k versus best-of-N?

| Method | Requires ground-truth verifier? | Returns what? | Primary role |
|---|---:|---|---|
| pass@k | Yes for evaluation | Metric | Evaluate chance any of k succeeds |
| consensus@k | No external score if answers can be canonicalized | Majority answer | Self-consistency inference |
| best-of-N | Requires scorer/verifier | Highest-scoring candidate | Reranked inference |

---

## Q152. SHOULD KNOW - Why does more reasoning sometimes help but sometimes hurt?

Helps by:

- decomposing the problem;
- exploring alternatives;
- correcting mistakes;
- allocating more compute.

Hurts by:

- wandering from a correct path;
- consuming context;
- increasing latency/cost;
- amplifying learned verbosity;
- triggering objective length bias.

**Memory liner:**

> Reasoning length is a resource, not a monotonic quality metric.

---

# 21. Equations to memorize cold

## 1. Autoregressive completion probability

$$
\pi_\theta(o_i\mid x)
=
\prod_{t=1}^{T_i}
\pi_\theta(o_{i,t}\mid x,o_{i,<t}).
$$

$$
\log \pi_\theta(o_i\mid x)
=
\sum_{t=1}^{T_i}
\log \pi_\theta(o_{i,t}\mid x,o_{i,<t}).
$$

---

## 2. pass@k

$$
\boxed{
\widehat{\operatorname{pass@k}}
=
1-
\frac{\binom{n-c}{k}}{\binom{n}{k}}
}
$$

---

## 3. Group reward mean

$$
\bar r
=
\frac{1}{G}
\sum_{i=1}^{G}r_i.
$$

---

## 4. Group reward standard deviation

$$
s_r
=
\sqrt{
\frac{1}{G}
\sum_{i=1}^{G}(r_i-\bar r)^2
}.
$$

---

## 5. GRPO group-relative advantage

$$
\boxed{
A_i
=
\frac{r_i-\bar r}{s_r+\delta}
}
$$

---

## 6. Policy probability ratio

$$
\boxed{
\rho_{i,t}(\theta)
=
\frac{
\pi_\theta(o_{i,t}\mid x,o_{i,<t})
}{
\pi_{old}(o_{i,t}\mid x,o_{i,<t})
}
}
$$

---

## 7. Clipped surrogate

$$
\boxed{
L^{clip}_{i,t}
=
\min\left(
\rho_{i,t}A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)A_i
\right)
}
$$

---

## 8. KL-regularized conceptual objective

$$
\boxed{
J
=
\mathbb{E}[L^{clip}]
-
\beta D_{KL}(\pi_\theta\Vert\pi_{ref})
}
$$

---

## 9. Per-response token averaging that creates length dependence

$$
\boxed{
L_i
=
\frac{1}{T_i}
\sum_{t=1}^{T_i}\ell_{i,t}
}
$$

Each token's scale is proportional to `1/T_i`.

---

## 10. Conceptual global-token normalization

$$
\boxed{
L_{token}
=
\frac{
\sum_i\sum_t M_{i,t}\ell_{i,t}
}{
\sum_i\sum_t M_{i,t}
}
}
$$

This is the intuition behind equalizing token-level contributions; it is not a complete reproduction of every DAPO/Dr.GRPO detail.

---

# 22. Tensor shapes to memorize cold

```text
PROMPT AND GROUPED COMPLETIONS

prompt IDs:                [B, T_p]
completion IDs:            [B, G, T_o]
completion mask:           [B, G, T_o]

MODEL OUTPUTS

logits:                    [B, G, T_o, V]
sampled-token logprobs:    [B, G, T_o]
old sampled logprobs:      [B, G, T_o]
reference sampled logprobs:[B, G, T_o]

REWARD AND ADVANTAGE

reward components:         [B, G, R_components]
total rewards:             [B, G]
group mean:                [B, 1]
group std:                 [B, 1]
advantages:                [B, G]
broadcast advantages:      [B, G, 1] -> [B, G, T_o]

POLICY OBJECTIVE

probability ratio:         [B, G, T_o]
clipped ratio:             [B, G, T_o]
per-token surrogate:       [B, G, T_o]
per-token KL:              [B, G, T_o]
masked per-completion sum: [B, G]
final loss/objective:      scalar
```

### Key gather operation

```text
log_softmax logits:  [B, G, T_o, V]
completion token IDs:[B, G, T_o]
                   gather over V
sampled logprobs:    [B, G, T_o]
```

### Reward broadcast

```text
advantages:          [B, G]
unsqueeze:            [B, G, 1]
broadcast with token terms:
                     [B, G, T_o]
```

---

# 23. Previous-day oral questions

Answer each in 30-90 seconds.

1. What operational definition of reasoning does this lecture use?
2. How is a reasoning model different from a vanilla assistant model?
3. Why can generating more reasoning tokens be viewed as additional compute?
4. Why is chain-of-thought decomposition useful for next-token models?
5. Why is reasoning length not always beneficial?
6. What is a reasoning compute budget?
7. Why may the displayed thought summary differ from raw reasoning tokens?
8. Why are code and math good domains for reasoning RL?
9. What makes a reward verifiable?
10. Why is parseable answer formatting important?
11. Define pass@k.
12. Derive pass@k from the complement event.
13. Why is pass@1 equal to `c/n`?
14. How does temperature affect pass@k?
15. Why must decoding parameters be reported with pass@k?
16. What is consensus@k?
17. Compare pass@k, consensus@k, and best-of-N.
18. Why can pure reasoning SFT be difficult to bootstrap?
19. Why might human-written reasoning differ from model-useful reasoning?
20. What are the format and accuracy rewards?
21. Why can format reward alone be gamed?
22. What is budget forcing?
23. What context-window issue arises during long reasoning?
24. What does GRPO stand for?
25. Walk through one GRPO training iteration.
26. Why sample several completions for the same prompt?
27. Derive the group-relative advantage.
28. What does a positive group advantage mean?
29. What does a negative group advantage mean?
30. Why does GRPO not need a value function?
31. What computational cost replaces value-model training?
32. Define the current-to-old policy ratio.
33. Distinguish `pi_old` and `pi_ref`.
34. Why does the objective use clipping?
35. Why does it also use reference-model KL?
36. What happens if `beta` is too large or too small?
37. What are the main GRPO tensor shapes?
38. How do you gather sampled-token log-probabilities?
39. Why is a completion mask required?
40. Compare PPO and GRPO advantage estimation.
41. Why is no learned reward model needed for verifiable reasoning tasks?
42. Is GRPO always cheaper than PPO?
43. Where can KL appear in PPO versus GRPO formulations?
44. What empirical pattern signals reasoning overlength?
45. Why does `1/T_i` create unequal token weights?
46. Why are long bad answers under-penalized relative to short bad answers?
47. How do token-level normalization corrections address this?
48. What is the standard-deviation bias in group normalization?
49. What happens when all group rewards are equal?
50. Why can symmetric clipping prevent low-probability useful tokens from recovering?
51. What motivates asymmetric clipping?
52. What does R1-Zero demonstrate?
53. What rewards are used in R1-Zero?
54. What readability problems appear under pure RL?
55. Give the four-stage R1 pipeline.
56. What is cold-start reasoning data?
57. Why use a language-consistency reward?
58. What is rejection sampling in the R1 data pipeline?
59. Why run SFT after reasoning RL?
60. Why mix non-reasoning data with reasoning data?
61. What does the final R1 RL stage optimize?
62. R1 versus R1-Zero in one sentence?
63. How is reasoning-trace distillation performed?
64. How does it differ from soft-logit distillation?
65. Why can distillation beat RL from scratch for a small student?
66. Reward improves but accuracy is flat: what do you inspect?
67. All completions are identical: what do you change?
68. All rewards are zero: what signal remains?
69. pass@k improves but pass@1 does not: what does it mean?
70. Length grows after accuracy plateaus: what do you plot?
71. Why keep reward components separate in logs?
72. Verifier versus reward model?
73. Outcome supervision versus process supervision?
74. GRPO versus DPO?
75. GRPO versus best-of-N?

---

# 24. Thirty memory liners

1. **Reasoning:** Solve through intermediate steps, not only direct recall.
2. **Reasoning model:** Prompt -> reasoning trajectory -> answer.
3. **Test-time compute:** Every reasoning token costs another autoregressive step.
4. **Decomposition:** Replace one hard extrapolation with familiar subproblems.
5. **Compute budget:** Allocate only as much thinking as the prompt needs.
6. **Verifiable task:** Correctness can be checked by code, tests, or ground truth.
7. **Parser:** Free-form reasoning needs a deterministic final-answer interface.
8. **pass@k:** At least one of k samples succeeds.
9. **pass@k formula:** One minus choosing all k from the failures.
10. **Temperature:** Too low duplicates; too high degrades; moderate creates useful diversity.
11. **Consensus:** Let independent trajectories vote on the final answer.
12. **Why RL:** Outcomes are cheap to verify while trajectories are expensive to label.
13. **Format reward:** Makes the output machine-checkable, not necessarily intelligent.
14. **Budget forcing:** Modify the prefix to continue or stop deliberation.
15. **GRPO:** Compare completions from the same prompt.
16. **Group advantage:** Reward minus group mean, scaled by group dispersion.
17. **Value model:** PPO learns a baseline; GRPO samples one.
18. **Policy ratio:** Current token probability divided by old token probability.
19. **Clipping:** Improve or suppress a token, but not too much in one update.
20. **Reference KL:** Learn reasoning without abandoning the base model.
21. **Old versus reference:** Old stabilizes the step; reference stabilizes the model identity.
22. **Length bias:** Per-response averaging gives short-response tokens larger weight.
23. **Bad incentive:** Long wrong answers can be penalized less per token.
24. **Std bias:** Group reward dispersion changes gradient scale.
25. **Asymmetric clipping:** Give tiny probabilities more room to grow than large ones to collapse.
26. **R1-Zero:** Pure verifiable RL can induce reasoning but not polished traces.
27. **Cold start:** Small clean SFT anchors readable reasoning before RL.
28. **Rejection sampling:** Generate, verify, filter, then turn successes into SFT data.
29. **R1:** Cold-start SFT -> reasoning RL -> mixed SFT -> final alignment RL.
30. **Distillation:** Let a strong teacher search; teach the student the successful trajectory.

---

# 25. Final interview-day checklist

You should be able to do all of the following without notes.

## Reasoning mental model

- [ ] Define reasoning operationally.
- [ ] Explain a reasoning model without claiming a new architecture is required.
- [ ] Explain why reasoning tokens are inference-time compute.
- [ ] Explain why more tokens can help and why unlimited tokens can hurt.
- [ ] Explain dynamic compute budgets and budget forcing.

## Evaluation

- [ ] Explain code and math verification.
- [ ] Derive `pass@k = 1 - C(n-c,k)/C(n,k)`.
- [ ] Show `pass@1 = c/n`.
- [ ] Explain temperature-quality-diversity trade-offs.
- [ ] Distinguish pass@k, consensus@k, and best-of-N.

## Reward design

- [ ] Explain accuracy, format, and language-consistency rewards.
- [ ] Explain why verifiable rewards remove the learned reward model.
- [ ] Explain reward hacking through an imperfect verifier.
- [ ] Explain why final-answer correctness does not guarantee good reasoning text.

## GRPO

- [ ] Walk through grouped rollout generation.
- [ ] Write the group-relative advantage equation.
- [ ] Derive the policy probability ratio.
- [ ] Explain clipping for positive and negative advantages.
- [ ] Distinguish `pi_old` from `pi_ref`.
- [ ] Explain the reference KL term and `beta`.
- [ ] Derive the tensor shapes `[B,G,T,V] -> [B,G,T]`.
- [ ] Explain why GRPO removes the value function.
- [ ] Compare PPO and GRPO in models, signals, and cost.

## GRPO failure modes

- [ ] Derive the `1/T_i` per-token weight.
- [ ] Explain why it favors longer incorrect responses.
- [ ] Explain token-level/global normalization intuition.
- [ ] Explain group-standard-deviation bias.
- [ ] Explain why all-equal group rewards produce no relative signal.
- [ ] Explain symmetric versus asymmetric clipping.

## R1 pipeline

- [ ] Explain R1-Zero's base -> RL-only setup.
- [ ] Explain its readability and language-mixing failures.
- [ ] Recite the R1 four-stage pipeline.
- [ ] Explain cold-start data.
- [ ] Explain rejection sampling and later SFT.
- [ ] Explain why non-reasoning assistant data is retained.
- [ ] Explain final helpfulness/harmlessness alignment.

## Distillation

- [ ] Explain reasoning-trace distillation.
- [ ] Contrast it with soft-logit distillation.
- [ ] Explain why it can be preferable for smaller models.

---

# 26. One-minute recap

```text
A reasoning model is still an autoregressive LM, but it generates
intermediate solution tokens before the final answer. Those tokens provide
extra test-time compute and can decompose a hard problem into easier steps.

Math and code are convenient because the final answer can be verified. This
makes pass@k meaningful and makes RL possible with programmatic rewards
instead of a learned reward model.

GRPO samples several completions for the same prompt. It turns each reward
into a group-relative advantage, then applies a PPO-like clipped token-level
policy update plus reference-model KL regularization. Its defining systems
advantage is that it does not train a value model.

The details of the objective matter. Per-response division by output length
can penalize short failures more than long failures, encouraging longer bad
answers. Group standard-deviation normalization and symmetric clipping also
create difficulty and exploration biases.

R1-Zero shows that verifiable RL alone can induce reasoning, but traces can
be unreadable or language-mixed. The full R1 recipe adds cold-start reasoning
SFT, reasoning RL, rejection-sampled mixed SFT, and final reasoning plus
helpfulness/harmlessness RL. A large R1 teacher can then generate verified
reasoning traces that are used as SFT data for smaller distilled models.
```

---

# 27. Final compact comparison table

| Method/stage | Input supervision | Online generation? | Reward/value model? | Main purpose |
|---|---|---:|---|---|
| Reasoning SFT | Prompt + labelled reasoning/answer | No | No | Imitate clean reasoning traces |
| PPO RLHF | Prompts + learned preference reward | Yes | Reward + value | Align with human preference |
| GRPO reasoning RL | Prompts + programmatic/verifiable reward | Yes, groups | No value; usually no learned reward | Discover and reinforce successful reasoning |
| Best-of-N | Prompt + scorer at inference | Yes | Scorer/verifier | Return best sampled candidate without training |
| Rejection-sampled SFT | Filtered successful model generations | Offline data generation | Verifier/judge during filtering | Consolidate strong behavior as dense supervision |
| Reasoning distillation | Teacher-generated reasoning sequences | Teacher generation offline | No RL required for student | Transfer reasoning behavior to a smaller model |

---

# 28. Five highest-value whiteboard derivations

## Derivation 1: pass@k

$$
P(\geq 1\text{ success})
=
1-P(0\text{ successes})
=
1-
\frac{\binom{n-c}{k}}{\binom{n}{k}}.
$$

## Derivation 2: policy ratio from log-probabilities

$$
\rho
=
\frac{p_{new}}{p_{old}}
=
\exp(\log p_{new}-\log p_{old}).
$$

## Derivation 3: group advantage

$$
A_i
=
\frac{r_i-\frac{1}{G}\sum_jr_j}
{\sqrt{\frac{1}{G}\sum_j(r_j-\bar r)^2}+\delta}.
$$

## Derivation 4: completion log-probability

$$
\log \pi(o\mid x)
=
\sum_t\log\pi(o_t\mid x,o_{<t}).
$$

## Derivation 5: length-dependent token weight

$$
L_i
=
\frac{1}{T_i}\sum_t\ell_{i,t}
\quad\Rightarrow\quad
\frac{\partial L_i}{\partial \ell_{i,t}}
=
\frac{1}{T_i}.
$$

Therefore a token in a length-20 output has five times the direct averaging weight of a token in a length-100 output.

---

# 29. Minimal mental map to retain after the interview

```text
REASONING = more structured autoregressive compute

VERIFY OUTCOME
    ↓
SAMPLE A GROUP
    ↓
COMPARE EACH REWARD TO ITS SIBLINGS
    ↓
UPWEIGHT BETTER TRAJECTORIES
DOWNWEIGHT WORSE TRAJECTORIES
    ↓
CLIP LOCAL CHANGE
REGULARIZE GLOBAL DRIFT
    ↓
WATCH THE INCENTIVES CREATED BY NORMALIZATION

R1-ZERO:
    capability can emerge from RL

R1:
    supervision + RL + filtering + alignment make it deployable

DISTILLATION:
    transfer successful large-model trajectories to a smaller student
```
