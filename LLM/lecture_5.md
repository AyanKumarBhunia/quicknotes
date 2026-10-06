# CME 295 Lecture 5 - Preference Tuning, Reward Models, RLHF, PPO, Best-of-N, and DPO

> **Scope:** This note is grounded in the supplied **CME 295 Lecture 5 transcript**. It adds standard equations, tensor-shape derivations, implementation mental models, and intuitive explanations only when they directly clarify material taught in this lecture. These additions are labelled **INTERVIEW CLARIFICATION** when the exact detail was not derived in class.
>
> **Goal:** Previous-day revision for ML/AI Research Scientist interviews: important questions, distinctions, equations, tensor shapes, mechanisms, implementation ideas, failure modes, and memory liners.
>
> **Transcript note:** The automatic transcript sometimes renders **PPO** as “PO,” **RLHF** as “RHF,” and **KL** as “Kale.” This note uses the standard technical names while preserving the lecture's intended content.

**Source transcript:** CME 295 Lecture 5 - `https://www.youtube.com/watch/PmW_TMQ3l0I`

---

## Lecture spine

```text
PRE-TRAINING
    learn broad language/code capability
            ↓
SUPERVISED FINE-TUNING (SFT)
    learn to follow instructions and behave like an assistant
            ↓
PREFERENCE TUNING
    learn which valid-looking responses humans/products prefer

PREFERENCE DATA
    prompt x
      ├── chosen response y_w
      └── rejected response y_l
            ↓
    pointwise / pairwise / listwise feedback
            ↓

RLHF
    Stage 1: train a reward model
        Bradley-Terry preference model
        pairwise training → pointwise scalar reward

    Stage 2: optimize the language-model policy
        sample rollouts from current policy
        frozen reward model scores completion
        value function estimates expected reward
        advantage = better/worse than expected
        PPO constrains policy updates
        reference-model KL prevents excessive drift

ALTERNATIVES
    Best-of-N
        generate N completions → reward-score → return best

    DPO
        preference pairs + trainable policy + frozen reference
        direct supervised preference objective
        no separately trained reward model or PPO loop
```

---

## Notation

```text
B           batch size
V           vocabulary size
D           hidden/model width
T_p         prompt length
T_c         completion length
T           total prompt + completion length
x           prompt

y_w         winning / chosen completion
y_l         losing / rejected completion

pi_theta    trainable policy / language model
pi_old      policy snapshot from the previous PPO update
pi_ref      frozen reference model, normally the SFT checkpoint

r_phi(x,y)  scalar reward-model score
V_psi(s_t)  value estimate for a partial generation/state
A_t         advantage estimate
rho_t       PPO probability ratio

epsilon     PPO clipping width
beta        KL/reference strength or DPO temperature parameter
sigma       logistic sigmoid
```

### Priority legend

- **MUST REMEMBER:** answer immediately and write the equation or shape.
- **SHOULD KNOW:** explain the mechanism and trade-off clearly.
- **LECTURE BOUNDARY:** explicitly deferred or not fully derived in this lecture.
- **INTERVIEW CLARIFICATION:** standard detail added to make the lecture material implementation-ready.

---

# 1. Where preference tuning fits in the LLM pipeline

## Q1. MUST REMEMBER - What are the three main model-development stages reviewed in this lecture?

```text
1. Pre-training
   Learn broad language, code, and knowledge through next-token prediction.

2. Supervised fine-tuning / instruction tuning
   Learn to respond to instructions like a useful assistant.

3. Preference tuning
   Adjust which possible responses the model prefers according to
   human, product, safety, style, or other preference criteria.
```

**Memory liner:**

> Pre-training builds capability, SFT teaches the task format, and preference tuning chooses the desired behavior among plausible outputs.

---

## Q2. MUST REMEMBER - Why is an SFT model not necessarily fully aligned?

SFT teaches the model examples of what it **should** produce. A model can therefore answer the requested task while still having undesirable properties such as:

- unfriendly tone;
- unsafe wording;
- poor helpfulness;
- unnecessary refusal;
- bad style;
- responses that are valid but not preferred.

**Memory liner:**

> Instruction following is not identical to human preference alignment.

---

## Q3. MUST REMEMBER - What is preference tuning?

Preference tuning changes the model so that, for the same prompt, it assigns relatively more probability to preferred responses and less probability to dispreferred responses.

```text
prompt x
   ├── preferred response y_w  ↑ probability
   └── rejected response  y_l  ↓ probability
```

It is primarily a **relative behavioral objective**, not simply another collection of perfect demonstrations.

**Memory liner:**

> SFT says “imitate this answer”; preference tuning says “prefer this answer over that one.”

---

## Q4. MUST REMEMBER - What is a preference pair?

A preference pair contains:

```text
(x, y_w, y_l)
```

where:

- `x` is the prompt;
- `y_w` is the winning/chosen response;
- `y_l` is the losing/rejected response;
- both responses are judged relative to the **same prompt**.

**Memory liner:**

> One prompt, two candidate completions, one ordering.

---

## Q5. MUST REMEMBER - Why can preference labels be easier to collect than SFT demonstrations?

Writing an ideal response from scratch is cognitively difficult. Comparing two existing responses and selecting the better one is often easier.

```text
SFT annotation:
“Write the perfect poem.”

Preference annotation:
“Which of these two poems is better?”
```

**Memory liner:**

> Ranking is often easier than generation.

---

## Q6. MUST REMEMBER - What signal can preference tuning add that ordinary SFT lacks?

Preference tuning adds an explicit **negative signal**:

```text
chosen answer   → increase relative preference
rejected answer → decrease relative preference
```

Ordinary response-only SFT trains on desired output tokens but does not directly say that a particular alternative completion should become less likely.

**Memory liner:**

> SFT supplies positive demonstrations; preference pairs contain positive and negative evidence.

---

## Q7. SHOULD KNOW - Why not fix every bad answer by adding another SFT example?

The lecture gives several reasons:

1. High-quality SFT demonstrations are expensive to construct.
2. Adding narrow corrections can distort the prompt/data mixture.
3. The model may need a relative preference rather than one exact answer.
4. Rewriting every failure into a perfect demonstration is difficult to scale.

However, the lecture also warns:

> If the model fails broadly, preference tuning may be treating the symptom; inspect the SFT data first.

---

## Q8. MUST REMEMBER - Is LoRA an alternative to preference tuning?

No. They act on different axes:

```text
LoRA:
How are parameters updated efficiently?

Preference tuning / PPO / DPO:
What objective is optimized?
```

LoRA can be used while performing SFT, DPO, or another preference objective.

**Memory liner:**

> LoRA changes the parameterization of training; preference tuning changes the learning objective.

---

## Q9. SHOULD KNOW - Does preference tuning mainly teach new factual knowledge?

The lecture's framing is **no**. Its central role is to reshape the existing completion distribution toward preferred behavior, such as friendlier wording or safer style, while preserving the underlying content.

```text
same factual content
        +
preferred tone / safety / helpfulness
```

**Memory liner:**

> Preference tuning usually changes which known behavior is selected, not the model's fundamental factual curriculum.

---

# 2. Preference-data formats and collection

## Q10. MUST REMEMBER - What are pointwise, pairwise, and listwise feedback?

### Pointwise

Assign one absolute score to each response:

```text
response y_i → score 0.82
```

### Pairwise

Choose the better of two responses:

```text
y_1 > y_2
```

### Listwise

Rank several responses:

```text
y_3 > y_1 > y_4 > y_2
```

**Memory liner:**

> Pointwise scores one, pairwise compares two, listwise ranks many.

---

## Q11. MUST REMEMBER - Why does the lecture focus on pairwise preference data?

Pairwise comparison avoids asking annotators to calibrate an absolute score scale and is simpler than consistently ranking a large list.

**Memory liner:**

> Pairwise labels retain useful ordering information without requiring a universal scoring scale.

---

## Q12. SHOULD KNOW - Why can pointwise ratings be difficult?

Human annotators may agree that response A is better than response B while disagreeing on whether A should receive `0.7`, `0.8`, or `0.9`.

The issue is **scale calibration**, not necessarily preference identification.

---

## Q13. MUST REMEMBER - How can two candidate responses be generated for a pair?

The lecture's recipe is:

```text
prompt x
   ↓ sample with positive temperature
response y_1

same prompt x
   ↓ sample again
response y_2
```

A positive sampling temperature permits different completions from the same policy.

**Memory liner:**

> Sample multiple stochastic completions, then compare them.

---

## Q14. MUST REMEMBER - Where should preference prompts come from?

They should resemble the target inference distribution, for example:

- real user logs;
- a desired prompt set;
- deliberately constructed safety or capability prompts.

**Memory liner:**

> Preference data is only useful if its prompts represent the behavior you care about at deployment.

---

## Q15. MUST REMEMBER - Who or what can judge a preference pair?

The lecture mentions:

- human raters;
- an LLM-as-a-judge;
- task metrics or rules, such as reference-based metrics where applicable;
- manually rewritten responses.

If human labels train the reward model, it is **RLHF**. If AI-generated feedback is used, one may call it **RLAIF**.

---

## Q16. SHOULD KNOW - Binary versus graded preference labels?

### Binary

```text
response 1 is better than response 2
```

### Graded

```text
much better / better / slightly better / slightly worse / worse / much worse
```

The lecture emphasizes that binary pairwise feedback is common because subjective graded scales are harder to calibrate consistently.

---

## Q17. MUST REMEMBER - What is the “rewrite” route to preference data?

Start from an undesirable production/logged response and rewrite it into a preferred response:

```text
prompt x
bad logged response y_l
human rewrite y_w
        ↓
preference pair (x, y_w, y_l)
```

This produces a strong contrast but requires more annotation effort than choosing between two existing outputs.

---

## Q18. SHOULD KNOW - What properties can preferences represent?

A reward or preference dimension can represent:

- helpfulness;
- harmlessness/safety;
- factuality;
- friendliness;
- relevance;
- style;
- concision;
- a holistic product-quality score.

The lecture notes that one may train separate reward models per dimension or use a combined score.

---

## Q19. MUST REMEMBER - Why are annotation guidelines important?

Human preference labels are sensitive to instructions given to raters. Ambiguous guidelines produce noisy and inconsistent labels.

```text
unclear criterion
    ↓
annotator disagreement
    ↓
noisy reward model
    ↓
unreliable policy optimization
```

**Memory liner:**

> The reward model can only be as coherent as the preference rubric used to train it.

---

## Q20. SHOULD KNOW - What are common preference-data failure modes?

- Prompts do not match deployment traffic.
- Chosen and rejected answers differ for irrelevant reasons.
- The “winner” is only marginally better or labels are inconsistent.
- Position, verbosity, style, or identity bias leaks into labels.
- All sampled answers are poor.
- Raters optimize the rubric rather than the actual product objective.

The lecture directly emphasizes subjectivity and guideline sensitivity; the remaining examples are **INTERVIEW CLARIFICATIONS** of the same data-quality problem.

---

# 3. RL terminology mapped to language models

## Q21. MUST REMEMBER - What are the basic reinforcement-learning objects?

```text
state s_t
   ↓ policy pi_theta(a_t | s_t)
action a_t
   ↓
reward
   ↓
update policy parameters theta
```

- **Agent:** chooses actions.
- **State:** information available before the action.
- **Action:** choice made by the agent.
- **Policy:** action distribution given the state.
- **Reward:** signal evaluating resulting behavior.

---

## Q22. MUST REMEMBER - How are RL objects mapped to an autoregressive LLM?

| RL term | LLM interpretation |
|---|---|
| Agent | The language model |
| State `s_t` | Prompt plus tokens generated so far |
| Action `a_t` | Next generated token |
| Action space | Vocabulary of size `V` |
| Policy `pi_theta(a_t|s_t)` | Next-token probability distribution |
| Trajectory / rollout | Entire generated completion |
| Reward | Score assigned to the prompt-completion pair |

**Memory liner:**

> The LLM state is the prefix, the action is the next token, and the policy is softmax over the vocabulary.

---

## Q23. MUST REMEMBER - Write the LLM policy equation.

For state/prefix `s_t=(x,y_{<t})`:

$$
\pi_\theta(a_t\mid s_t)
=
P_\theta(y_t=a_t\mid x,y_{<t}).
$$

Tensor shapes:

```text
hidden state at t: [B, D]
vocabulary logits: [B, V]
policy probs:      [B, V]
```

---

## Q24. MUST REMEMBER - What is a completion or rollout?

A completion/rollout is the full sequence sampled after the prompt:

$$
y=(y_1,\dots,y_{T_c}).
$$

Its probability factorizes autoregressively:

$$
\boxed{
\pi_\theta(y\mid x)
=
\prod_{t=1}^{T_c}
\pi_\theta(y_t\mid x,y_{<t})
}
$$

and its log-probability is:

$$
\boxed{
\log \pi_\theta(y\mid x)
=
\sum_{t=1}^{T_c}
\log \pi_\theta(y_t\mid x,y_{<t})
}
$$

**INTERVIEW CLARIFICATION:** Only completion-token positions should contribute to this sequence log-probability; prompt positions are conditioning context.

---

## Q25. MUST REMEMBER - Is the preference reward token-level or completion-level in this lecture?

The reward model supplies roughly **one scalar for the full prompt-completion pair**:

```text
(x, complete y) → scalar reward r
```

This is much sparser than SFT, where every completion token supplies a next-token learning signal.

**Memory liner:**

> SFT gives dense token supervision; RLHF begins from a sparse sequence-level reward.

---

## Q26. SHOULD KNOW - Why is sequence-level feedback still able to train token probabilities?

The policy generated the sequence through a chain of token actions. Policy-gradient machinery attributes the eventual outcome back to the probabilities of sampled actions.

```text
token decisions
    ↓ compose rollout
completion reward
    ↓ credit assignment
update token-action probabilities
```

The exact policy-gradient derivation is not developed in the lecture.

---

## Q27. MUST REMEMBER - On-policy versus off-policy?

### On-policy

Train using outputs sampled from the **current** policy being optimized.

```text
current pi_theta → new rollouts → score → update pi_theta
```

### Off-policy

Train using data generated by another model, an older policy, humans, or a fixed dataset.

**Memory liner:**

> On-policy learns from what the current model does now; off-policy learns from externally supplied behavior.

---

## Q28. MUST REMEMBER - Is PPO on-policy?

Yes. During PPO-based RLHF, the current policy generates rollouts that are scored and then used for updates.

SFT and DPO use fixed datasets and are supervised/off-policy-style procedures in the lecture's comparison.

---

# 4. RLHF end-to-end mental map

## Q29. MUST REMEMBER - What does RLHF stand for?

**Reinforcement Learning from Human Feedback.**

“Human feedback” refers to the human preference labels used to train the reward model, not to a human manually issuing every policy update.

---

## Q30. MUST REMEMBER - What are the two RLHF stages taught here?

```text
STAGE 1 - Reward modelling
preference pairs
    ↓
train r_phi(x,y)
    ↓
scalar quality model

STAGE 2 - Policy optimization
current policy samples y ~ pi_theta(.|x)
    ↓
frozen reward model scores r_phi(x,y)
    ↓
PPO updates pi_theta
while constraining drift from pi_ref / pi_old
```

**Memory liner:**

> First learn what humans prefer; then optimize the policy against that learned preference signal.

---

## Q31. MUST REMEMBER - What is frozen and what is trained during the RL stage?

- **Reward model:** frozen.
- **Reference/SFT model:** frozen.
- **Current policy:** trained.
- **Value function/value head:** trained, usually jointly with the policy.
- **Old policy:** a fixed snapshot for a PPO update epoch, then refreshed.

**Memory liner:**

> The judge is frozen; the actor and critic learn.

---

## Q32. MUST REMEMBER - Why does RLHF need a reference model?

The policy should improve preference reward without destroying the useful distribution learned during pre-training and SFT.

The reference model acts as an anchor:

```text
higher preference reward
        +
stay reasonably close to SFT behavior
```

---

## Q33. SHOULD KNOW - RLHF versus RLAIF?

```text
RLHF:
preference labels originate from humans

RLAIF:
preference labels originate from an AI-based evaluator
```

The downstream reward-model/PPO mechanism can otherwise look similar.

---

## Q34. MUST REMEMBER - Why is RLHF feedback called sparse?

For a completion containing many tokens:

```text
SFT:   one target-token signal at many positions
RLHF:  approximately one final reward for the completion
```

The policy therefore faces a harder credit-assignment problem.

---

## Q35. SHOULD KNOW - What is the full RLHF actor loop?

```text
1. Sample prompt x.
2. Current policy generates completion y.
3. Frozen reward model computes r_phi(x,y).
4. Reference/current/old-policy log-probabilities are computed.
5. Value function estimates expected return for partial prefixes.
6. Construct advantages.
7. Apply PPO policy and value updates.
8. Refresh old-policy snapshot and repeat.
```

The lecture explains this conceptually rather than as production code.

---

# 5. Reward modelling and Bradley-Terry preferences

## Q36. MUST REMEMBER - What does a reward model compute?

$$
\boxed{r_\phi(x,y)\in\mathbb{R}}
$$

It receives one prompt and one response and returns one scalar score.

```text
prompt + response token IDs: [B, T]
transformer hidden states:    [B, T, D]
selected pooled state:        [B, D]
reward head D → 1
scalar scores:                [B]
```

**Memory liner:**

> The reward model maps a contextualized prompt-response pair to one real-valued preference score.

---

## Q37. MUST REMEMBER - What is the Bradley-Terry model?

Given two completions `y_i` and `y_j` for prompt `x`:

$$
\boxed{
P(y_i \succ y_j\mid x)
=
\frac{e^{r_\phi(x,y_i)}}
{e^{r_\phi(x,y_i)}+e^{r_\phi(x,y_j)}}
}
$$

Equivalent sigmoid form:

$$
\boxed{
P(y_i \succ y_j\mid x)
=
\sigma\big(r_\phi(x,y_i)-r_\phi(x,y_j)\big)
}
$$

where:

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

**Memory liner:**

> Preference probability is the sigmoid of the reward-score difference.

---

## Q38. MUST REMEMBER - Why does Bradley-Terry use a score difference?

Only the relative ordering matters:

```text
r_w >> r_l → preference probability near 1
r_w =  r_l → preference probability 0.5
r_w << r_l → preference probability near 0
```

**Memory liner:**

> The winner need not have an absolute “correct score”; it only needs a higher score than the loser.

---

## Q39. MUST REMEMBER - Derive the reward-model pairwise loss.

For data where `y_w` is preferred to `y_l`, maximize:

$$
P(y_w\succ y_l\mid x)
=
\sigma\big(r_w-r_l\big).
$$

Maximum likelihood over independent pairs gives:

$$
\prod_i \sigma(r_{w,i}-r_{l,i}).
$$

Take log, convert maximization to minimization:

$$
\boxed{
\mathcal L_{RM}
=
-\mathbb E_{(x,y_w,y_l)}
\left[
\log\sigma
\left(
r_\phi(x,y_w)-r_\phi(x,y_l)
\right)
\right]
}
$$

**Memory liner:**

> Train the reward gap to be positive for every chosen-rejected pair.

---

## Q40. MUST REMEMBER - What are the reward-model loss shapes?

```text
chosen score r_w:         [B]
rejected score r_l:       [B]
score difference:         [B]
preference probability:   [B]
pairwise losses:          [B]
mean loss:                scalar
```

PyTorch mental map:

```python
loss = -torch.nn.functional.logsigmoid(chosen_reward - rejected_reward).mean()
```

---

## Q41. MUST REMEMBER - Is the reward model pairwise at inference time?

No.

```text
training:
(x, y_w) and (x, y_l) → compare two scores

inference:
(x, y) → one scalar score
```

It is **pairwise-trained but pointwise-used**.

**Memory liner:**

> Pairs teach the scale ordering; a single candidate can later be scored alone.

---

## Q42. INTERVIEW CLARIFICATION - What reward ambiguity follows from the pairwise loss?

If the same constant `c` is added to both scores:

$$
(r_w+c)-(r_l+c)=r_w-r_l.
$$

Therefore pairwise comparisons do not identify the absolute reward offset.

**Memory liner:**

> Bradley-Terry identifies relative score differences, not a unique absolute origin.

This explains why reward normalization/rescaling can matter when rewards enter an RL objective.

---

## Q43. SHOULD KNOW - Is reward modelling classification or regression?

The lecture treats it as a probabilistic pairwise-ranking objective:

- labels are binary preferences;
- the model emits continuous scalar scores;
- score differences enter a logistic classification likelihood.

A precise description is:

> Continuous scoring model trained through pairwise logistic classification.

---

## Q44. MUST REMEMBER - What model architectures can implement a reward model?

The lecture gives two broad options:

1. A decoder-only language-model backbone plus a scalar head.
2. An encoder-only model such as BERT using a sequence representation such as `[CLS]`.

Modern LLM pipelines commonly reuse a decoder-only SFT backbone with a scalar reward head.

---

## Q45. INTERVIEW CLARIFICATION - Which hidden state can feed a causal-LM reward head?

A common implementation uses the hidden state at the final non-padding token:

```text
H:                     [B, T, D]
last valid index:      [B]
pooled hidden state:   [B, D]
linear reward head:    [D, 1]
reward:                [B]
```

The lecture specifies an end-of-sequence classification head conceptually but does not mandate one pooling implementation.

---

## Q46. MUST REMEMBER - Why must the reward depend on both prompt and response?

A response is not universally good or bad. Its quality depends on whether it appropriately answers the prompt.

```text
same response
   + prompt A → helpful
   + prompt B → irrelevant or unsafe
```

**Memory liner:**

> Reward is conditional response quality, not an unconditional sentence score.

---

## Q47. SHOULD KNOW - Can one reward model represent every preference dimension?

Possibly, through a holistic score, but the lecture notes that usefulness, friendliness, and safety are different dimensions. Separate reward models or multi-objective combinations may be appropriate.

**Memory liner:**

> “Good” must be defined along an explicit preference dimension.

---

## Q48. SHOULD KNOW - Why normalize reward-model outputs during RL?

Raw Bradley-Terry score scale is not inherently calibrated. PPO update magnitude can become sensitive to reward scale, so implementations often normalize or rescale rewards/advantages.

The lecture mentions reward normalization but does not prescribe one exact method.

---

## Q49. MUST REMEMBER - What does a reward-model score mean numerically?

A score such as `0.8` or `-2.0` is not a probability by itself. It is an unbounded scalar whose **difference** with another score determines a Bradley-Terry preference probability.

```text
reward score: real number
sigmoid(score difference): probability
```

---

## Q50. SHOULD KNOW - What is RewardBench in the lecture?

It is mentioned as a benchmark for evaluating whether reward models rank preferred and rejected responses appropriately.

The lecture does not derive its benchmark composition.

---

# 6. Policy optimization: reward versus staying close to the SFT model

## Q51. MUST REMEMBER - What happens after the reward model is trained?

The policy stage repeats:

```text
prompt x
   ↓
current policy pi_theta samples completion y
   ↓
frozen reward model computes r_phi(x,y)
   ↓
update pi_theta to favor higher-reward behavior
```

The reward model is not trained during this stage.

---

## Q52. MUST REMEMBER - What is the high-level RLHF optimization objective?

The lecture's core objective is:

```text
maximize preference reward
        while
preventing excessive drift from the SFT/reference model
```

A standard mathematical statement is:

$$
\boxed{
\max_\theta
\mathbb E_{x,\,y\sim\pi_\theta(\cdot|x)}
\left[
r_\phi(x,y)
-
\beta D_{KL}
\big(
\pi_\theta(\cdot|x)
\|\pi_{ref}(\cdot|x)
\big)
\right]
}
$$

**INTERVIEW CLARIFICATION:** In practical language-model RL, KL can be evaluated token by token along sampled prefixes rather than as one exact distribution over all possible full sequences.

---

## Q53. MUST REMEMBER - Why not maximize reward without a reference constraint?

The lecture gives three central reasons:

1. **Preserve prior capability:** the SFT model already contains useful language and task behavior.
2. **Avoid reward hacking:** the reward model is an imperfect proxy.
3. **Improve stability:** unconstrained policy updates can become unstable.

**Memory liner:**

> Optimize the learned preference proxy, but do not let the policy abandon the model that made it capable.

---

## Q54. MUST REMEMBER - What is reward hacking?

Reward hacking occurs when the policy exploits imperfections in the reward function and obtains a high measured score without satisfying the intended goal.

Lecture analogy:

```text
true goal: informative lecture
proxy reward: loud applause
exploitation: tell jokes instead of teaching
result: high reward, failed true objective
```

**Memory liner:**

> A policy can optimize the metric more successfully than it optimizes what the metric was meant to represent.

---

## Q55. MUST REMEMBER - Why is reward hacking especially plausible with a learned reward model?

The reward model is trained on finite, noisy preference data and can be wrong outside that distribution. Policy optimization actively searches for high-score regions, including regions where the reward model's errors can be exploited.

```text
ordinary evaluation:
model sees typical candidates

policy optimization:
model searches for candidates that maximize the evaluator
```

**Memory liner:**

> Optimization pressure finds evaluator blind spots.

---

## Q56. SHOULD KNOW - What is catastrophic forgetting in this context?

Large policy changes may damage useful capabilities learned during pre-training and SFT while over-specializing to the preference objective.

A reference-model constraint helps retain the original behavior distribution.

---

# 7. KL divergence and the reference-model anchor

## Q57. MUST REMEMBER - Define KL divergence.

For discrete distributions `P` and `Q`:

$$
\boxed{
D_{KL}(P\|Q)
=
\sum_i P(i)
\log\frac{P(i)}{Q(i)}
}
$$

**Memory liner:**

> KL measures the expected log probability-ratio when outcomes are sampled from `P`.

---

## Q58. MUST REMEMBER - Why is KL divergence not a distance metric?

In general:

$$
D_{KL}(P\|Q)\neq D_{KL}(Q\|P).
$$

It is asymmetric and does not satisfy the usual metric axioms.

**Memory liner:**

> KL is a directional discrepancy, not a geometric distance.

---

## Q59. MUST REMEMBER - What are the two basic KL properties highlighted in the lecture?

$$
D_{KL}(P\|Q)\ge 0
$$

and:

$$
D_{KL}(P\|Q)=0
\iff P=Q
$$

The lecture notes that non-negativity can be proved using Jensen's inequality.

---

## Q60. MUST REMEMBER - What does policy-to-reference KL mean intuitively?

At each generation state, compare the current policy's next-token distribution with the SFT reference distribution:

```text
current policy probs:   [B, T_c, V]
reference policy probs: [B, T_c, V]
KL per token:           [B, T_c]
```

A large KL means preference tuning has substantially changed which tokens the model would produce.

---

## Q61. INTERVIEW CLARIFICATION - What sampled KL approximation is common in LLM RL?

For an action/token sampled from the current policy:

$$
\widehat{KL}_t
\approx
\log\pi_\theta(a_t|s_t)
-
\log\pi_{ref}(a_t|s_t).
$$

Summing this along the generated completion gives a sampled sequence-level penalty.

This is not the exact categorical KL over every vocabulary entry, but it is directly computable on sampled actions.

---

## Q62. MUST REMEMBER - What is the role of `beta` in the KL-regularized objective?

In:

$$
r(x,y)-\beta D_{KL}(\pi_\theta\|\pi_{ref}),
$$

- larger `beta` places more weight on staying close to the reference;
- smaller `beta` permits more policy movement toward reward maximization.

**Memory liner:**

> `beta` sets the reward-versus-conservatism trade-off.

---

## Q63. SHOULD KNOW - Why can too much KL regularization be harmful?

If the penalty dominates, the trainable policy remains almost identical to the SFT model and preference tuning produces little behavioral improvement.

```text
beta too high → safe but little adaptation
beta too low  → larger adaptation but more reward hacking/instability risk
```

---

# 8. Reward, value, and advantage

## Q64. MUST REMEMBER - What is the reward in this lecture?

The reward is a scalar evaluation of a **completed** prompt-response pair:

$$
r_\phi(x,y).
$$

```text
prompt + full completion → reward model → scalar
```

---

## Q65. MUST REMEMBER - What is the value function?

The value function estimates expected future reward from a **partial generation** when the policy continues generating:

$$
\boxed{
V_\psi(s_t)
\approx
\mathbb E_{y_{t:}\sim\pi_\theta}
\left[
\text{future return}\mid s_t
\right]
}
$$

where:

$$
s_t=(x,y_{<t}).
$$

**Memory liner:**

> Reward scores the finished answer; value predicts how promising the current prefix is before the answer is finished.

---

## Q66. MUST REMEMBER - Reward versus value?

| Quantity | Input | Output meaning | Granularity |
|---|---|---|---|
| Reward model `r_phi(x,y)` | Prompt + full completion | How good the completed response is | Completion-level |
| Value `V_psi(s_t)` | Prompt + partial completion | Expected eventual reward if policy continues | Token/state-level |

---

## Q67. MUST REMEMBER - What is advantage?

At the lecture's level, advantage means:

> How much better or worse was the sampled behavior than what was expected from that state?

A simplified mental equation is:

$$
\boxed{A_t\approx R_t-V_\psi(s_t)}
$$

where `R_t` is a suitable return/target.

**Memory liner:**

> Advantage is reward relative to a baseline expectation.

---

## Q68. MUST REMEMBER - Why use advantage instead of raw reward?

Subtracting a value baseline reduces variance:

```text
raw reward:
large variation caused by easy/hard prompts and noisy outcomes

reward - expected reward:
focus on whether this sampled action was unusually good or bad
```

Lower-variance gradient estimates usually improve training stability and sample efficiency.

---

## Q69. MUST REMEMBER - What does the sign of advantage mean?

```text
A_t > 0:
the sampled token/action was better than expected
→ increase its probability

A_t < 0:
the sampled token/action was worse than expected
→ decrease its probability

A_t ≈ 0:
behavior was near expectation
→ weak policy pressure
```

---

## Q70. MUST REMEMBER - How is the value function implemented?

The lecture describes a scalar value head attached to the policy backbone:

```text
policy hidden states: [B, T_c, D]
value head:           D → 1
value predictions:   [B, T_c]
```

The value head is trained as a regression problem, usually jointly with the policy.

---

## Q71. LECTURE BOUNDARY - What is Generalized Advantage Estimation (GAE)?

The lecture names GAE as the common method used to estimate advantages but explicitly does not derive its formula.

For this lecture, remember only:

```text
reward + sequence of value estimates
        ↓ GAE
lower-variance token-level advantage estimates
```

Do not treat the exact `gamma`/`lambda` recursion as required content from this lecture.

---

## Q72. SHOULD KNOW - Why is the value model another source of complexity?

It adds:

- another output head or model component;
- a separate regression loss;
- extra memory and compute;
- another target-estimation problem;
- additional hyperparameters through advantage estimation.

This contributes to PPO-RLHF's engineering burden.

---

# 9. PPO: Proximal Policy Optimization

## Q73. MUST REMEMBER - What does PPO stand for?

**Proximal Policy Optimization.**

“Proximal” means policy updates should remain close to a trusted policy rather than changing arbitrarily in one step.

---

## Q74. MUST REMEMBER - Which two notions of “stay close” appear in the lecture?

1. **Current policy versus old policy:** constrain each PPO update step for stability.
2. **Current policy versus reference SFT model:** constrain total drift from the aligned starting checkpoint.

**Memory liner:**

> `pi_old` controls step size; `pi_ref` anchors the whole RL run.

---

## Q75. MUST REMEMBER - Define the PPO probability ratio.

For token action `a_t` in state `s_t`:

$$
\boxed{
\rho_t(\theta)
=
\frac{
\pi_\theta(a_t|s_t)
}{
\pi_{old}(a_t|s_t)
}
}
$$

Equivalent log-space computation:

$$
\rho_t
=
\exp
\left(
\log\pi_\theta(a_t|s_t)
-
\log\pi_{old}(a_t|s_t)
\right).
$$

**Memory liner:**

> The ratio asks how much more or less likely the new policy makes the sampled token than the old policy did.

---

## Q76. MUST REMEMBER - What do ratio values mean?

```text
rho_t = 1:
new and old policies assign the same probability

rho_t > 1:
new policy increases the sampled action's probability

rho_t < 1:
new policy decreases the sampled action's probability
```

---

## Q77. MUST REMEMBER - Write the PPO clipped objective.

$$
\boxed{
L^{clip}(\theta)
=
\mathbb E_t
\left[
\min
\left(
\rho_t(\theta)A_t,
\operatorname{clip}
\left(
\rho_t(\theta),
1-\epsilon,
1+\epsilon
\right)A_t
\right)
\right]
}
$$

This expression is conventionally **maximized**. A codebase that minimizes a loss uses `-L^{clip}`.

---

## Q78. MUST REMEMBER - Why is there a `min` in the PPO objective?

The `min` chooses the more conservative of:

- the unconstrained probability-ratio objective;
- the clipped objective.

This removes the incentive for a beneficial-looking update to grow beyond the allowed trust region.

**Memory liner:**

> PPO accepts useful probability movement but stops rewarding movement once it becomes too large.

---

## Q79. MUST REMEMBER - Explain PPO clipping when `A_t > 0`.

Positive advantage means the sampled action should become more likely.

```text
increase rho_t
      ↓
increase objective
      ↓
once rho_t > 1 + epsilon,
additional increase no longer improves clipped objective
```

**Memory liner:**

> Reinforce a good action, but cap how much its probability can rise in one update.

---

## Q80. MUST REMEMBER - Explain PPO clipping when `A_t < 0`.

Negative advantage means the sampled action should become less likely.

```text
decrease rho_t
      ↓
increase objective for negative A_t
      ↓
once rho_t < 1 - epsilon,
additional decrease is clipped
```

**Memory liner:**

> Suppress a bad action, but cap how much its probability can fall in one update.

---

## Q81. MUST REMEMBER - What are the PPO tensor shapes for language modelling?

```text
completion token IDs: [B, T_c]
new token log-probs:  [B, T_c]
old token log-probs:  [B, T_c]
probability ratios:   [B, T_c]
advantages:           [B, T_c]
completion mask:      [B, T_c]
clipped objectives:   [B, T_c]
reduced policy loss:  scalar
```

---

## Q82. INTERVIEW CLARIFICATION - Minimal PPO policy-loss code?

```python
# All tensors are [B, T_completion].
ratio = torch.exp(new_logp - old_logp)

unclipped = ratio * advantages
clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages

policy_objective = torch.minimum(unclipped, clipped)
policy_loss = -(policy_objective * completion_mask).sum() / completion_mask.sum()
```

`old_logp` and `advantages` should be treated as fixed targets during the policy update.

---

## Q83. MUST REMEMBER - What is PPO with a KL penalty?

A second lecture-level PPO formulation explicitly penalizes divergence:

$$
\text{objective}
\approx
\mathbb E[\rho_t A_t]
-
\beta D_{KL}(\pi_\theta\|\pi_{anchor}).
$$

The original PPO paper may compare to the old policy. LLM RLHF commonly also uses a frozen SFT reference as the anchor.

---

## Q84. SHOULD KNOW - Can clipping and reference KL be used together?

Yes. The lecture notes that practical LLM PPO recipes may combine:

```text
PPO ratio clipping
        +
KL penalty to frozen SFT reference
```

They solve related but different stability problems.

---

## Q85. MUST REMEMBER - Old policy versus reference policy?

| Model | Meaning | Updated? | Role |
|---|---|---:|---|
| `pi_theta` | Current trainable policy | Yes | Model being optimized |
| `pi_old` | Snapshot before PPO update epochs | Periodically refreshed | Probability-ratio denominator / local trust region |
| `pi_ref` | Frozen SFT checkpoint | No | Global behavioral anchor |

**Memory liner:**

> Old is yesterday's RL policy; reference is the pre-RL SFT model.

---

## Q86. SHOULD KNOW - What other loss must accompany the PPO policy objective?

The value head must learn to predict returns, so PPO training also requires a value-regression loss.

A full implementation therefore has at least:

```text
policy loss
+
value loss
+
reference KL control
```

Exact coefficients and full training recipe are not derived in the lecture.

---

## Q87. MUST REMEMBER - Why does PPO require multiple model states?

The lecture identifies:

- trainable policy;
- old-policy snapshot;
- frozen reference/SFT model;
- frozen reward model;
- value function/value head.

Some components can share a backbone, but conceptually PPO depends on all these signals.

**Memory liner:**

> PPO-RLHF is not one forward pass through one model; it is a coordinated multi-model training system.

---

## Q88. LECTURE BOUNDARY - Is GRPO covered here?

No. The lecture only names GRPO as a later RL variant and explicitly defers it to Lecture 6.

Do not add GRPO equations or group-normalized advantages to this Lecture 5 note.

---

# 10. Why PPO-based RLHF is difficult

## Q89. MUST REMEMBER - What is the two-stage dependency problem in RLHF?

```text
preference data
    ↓
train reward model
    ↓
run policy optimization against that reward model
```

If the reward model is later found to be flawed, the policy stage may need to be repeated after retraining the reward model.

**Memory liner:**

> Errors in the learned objective propagate into the expensive optimization that follows.

---

## Q90. MUST REMEMBER - Which hyperparameters make PPO-RLHF hard to tune?

The lecture explicitly mentions:

- `beta` for KL control;
- `epsilon` for PPO clipping;
- multiple GAE hyperparameters;
- additional optimization and normalization choices.

**Memory liner:**

> PPO has interacting knobs for reward scale, trust region, credit assignment, and optimization.

---

## Q91. MUST REMEMBER - Why is average reward an incomplete training metric?

A rising learned reward can mean:

- genuine behavioral improvement;
- exploitation of reward-model errors;
- reduced diversity;
- drift toward a narrow style;
- reward-scale artifacts.

Therefore average reward should be accompanied by held-out preference evaluation, capability tests, safety tests, and KL/drift monitoring.

The lecture directly warns that average reward is less transparent than SFT cross-entropy; the expanded monitoring list is an **INTERVIEW CLARIFICATION**.

---

## Q92. MUST REMEMBER - Why does PPO need exploration?

If repeated rollouts for a prompt are nearly identical, the model sees little variation and cannot discover which alternative completions receive better rewards.

```text
no diversity
   ↓
no meaningful preference contrast
   ↓
weak learning signal
```

**Memory liner:**

> The policy must explore enough to discover better behavior, but not so much that samples become useless.

---

## Q93. SHOULD KNOW - How is exploration introduced in language-model RL?

At the lecture's level, through stochastic generation, such as nonzero temperature, so the current policy produces varied rollouts.

The lecture does not derive entropy regularization or other formal exploration bonuses.

---

## Q94. MUST REMEMBER - What is the difference between PPO training data and SFT training data?

```text
SFT:
fixed prompt-response demonstrations supplied by dataset

PPO:
current policy generates new responses during training
```

This is why PPO is on-policy and SFT is not.

---

## Q95. SHOULD KNOW - What are the main PPO-RLHF failure modes?

- Reward-model overoptimization/reward hacking.
- Policy collapse or low diversity.
- Excessive KL drift.
- Too much KL, causing no improvement.
- Unstable value estimates.
- Poor advantage normalization.
- Noisy or biased preference labels.
- Distribution mismatch between reward training and policy rollouts.
- High systems cost and model-memory pressure.

The first, second-to-last, and systems issues are directly developed in the lecture; the detailed optimization symptoms are **INTERVIEW CLARIFICATIONS**.

---

## Q96. SHOULD KNOW - What should be logged during PPO-RLHF?

A sensible interview answer is:

```text
reward mean / variance
reference KL
PPO clip fraction
probability-ratio statistics
policy loss
value loss
value explained variance
completion length
entropy or diversity indicators
held-out preference win rate
capability/safety regressions
```

The lecture only explicitly discusses reward monitoring, KL, and instability; the rest is a standard implementation checklist.

---

# 11. Best-of-N inference

## Q97. MUST REMEMBER - What is Best-of-N?

Given one prompt:

```text
1. Generate N candidate completions from the SFT/policy model.
2. Score every candidate with a trained reward model.
3. Return the highest-scoring completion.
```

$$
\boxed{
y^*=
\arg\max_{y_i,\,i=1,\dots,N}
r_\phi(x,y_i)
}
$$

**Memory liner:**

> Do not retrain the policy; search over several samples and let the reward model choose.

---

## Q98. MUST REMEMBER - Does Best-of-N update model weights?

No. It is an inference-time selection strategy.

```text
SFT model: frozen
reward model: frozen
training: none
inference: N generations + N scores
```

---

## Q99. MUST REMEMBER - What problem does Best-of-N avoid?

It avoids PPO/RL training complexity:

- no value model;
- no policy-gradient loop;
- no PPO hyperparameters;
- no on-policy training instability.

---

## Q100. MUST REMEMBER - What cost does Best-of-N introduce?

It moves work from training to inference:

```text
single response cost      ≈ 1 generation
Best-of-N generation cost ≈ N generations + reward scoring
```

This can be unsuitable for high-traffic or low-latency serving.

**Memory liner:**

> Best-of-N buys quality with inference compute.

---

## Q101. SHOULD KNOW - Why can parallel Best-of-N still increase latency?

Even if candidates are generated in parallel, the system must wait for the slowest candidate before selecting the best.

```text
latency ≈ max(latency_1, ..., latency_N) + scoring
```

As `N` grows, the maximum of the generation-time distribution tends to increase.

---

## Q102. MUST REMEMBER - What if all N candidates are poor?

The reward model returns the **best among them**, not necessarily a genuinely good answer.

$$
\max_i r(x,y_i)
$$

can still correspond to poor absolute quality if the proposal model has weak support.

**Memory liner:**

> Reranking cannot select an answer the generator never produced.

---

## Q103. SHOULD KNOW - Why does reward-score scale not matter for pure Best-of-N ranking?

Any strictly increasing transformation preserves the argmax:

$$
\arg\max_i r_i
=
\arg\max_i f(r_i)
$$

for monotonically increasing `f`.

Absolute reward scale matters more when rewards enter a gradient-based optimization objective.

---

## Q104. SHOULD KNOW - How does `N` affect quality and cost?

Generally:

```text
N ↑
→ more chance of sampling a strong completion
→ more generation cost
→ potentially more tail latency
→ more opportunity to exploit reward-model errors
```

The final point is a standard inference from reward-model selection rather than an explicit lecture derivation.

---

## Q105. MUST REMEMBER - Best-of-N versus self-consistency?

Both generate multiple samples, but they select differently:

```text
Best-of-N:
score full candidates with a reward model and pick maximum

Self-consistency:
extract final answers and choose the majority answer
```

Self-consistency appeared in the previous lecture; Best-of-N is the reward-model-based method taught here.

---

# 12. Direct Preference Optimization (DPO)

## Q106. MUST REMEMBER - What does DPO stand for?

**Direct Preference Optimization.**

---

## Q107. MUST REMEMBER - What problem is DPO trying to solve?

DPO seeks preference alignment without:

- separately fitting and serving a reward model inside an RL loop;
- learning a value function;
- running on-policy PPO optimization;
- managing PPO instability and many RL hyperparameters.

**Memory liner:**

> DPO converts preference alignment into a direct supervised loss over chosen and rejected completions.

---

## Q108. MUST REMEMBER - What models does DPO require during training?

Conceptually:

1. `pi_theta`: trainable policy initialized from the SFT model.
2. `pi_ref`: frozen reference model, normally the original SFT checkpoint.

```text
preference pair
    ├── policy log-probs
    └── reference log-probs
          ↓
DPO loss
```

The lecture contrasts these two model copies with the larger PPO stack.

---

## Q109. MUST REMEMBER - What data does DPO use?

The same static preference triples:

$$
(x,y_w,y_l).
$$

No on-policy rollout is required by the basic DPO objective.

---

## Q110. MUST REMEMBER - Write the DPO loss.

Define the policy-versus-reference log-ratio for a completion:

$$
\ell_\theta(x,y)
=
\log\pi_\theta(y|x)
-
\log\pi_{ref}(y|x).
$$

Then:

$$
\boxed{
\mathcal L_{DPO}
=
-\mathbb E_{(x,y_w,y_l)}
\left[
\log\sigma
\left(
\beta
\left[
\ell_\theta(x,y_w)
-
\ell_\theta(x,y_l)
\right]
\right)
\right]
}
$$

Expanded:

$$
\boxed{
\mathcal L_{DPO}
=
-\mathbb E
\log\sigma
\left(
\beta
\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]
\right)
}
$$

---

## Q111. MUST REMEMBER - What is DPO trying to increase?

DPO wants the trainable policy to improve the chosen response **relative to the reference** more than it improves the rejected response:

```text
policy/reference gain for chosen
                  >
policy/reference gain for rejected
```

**Memory liner:**

> DPO learns a relative preference margin after subtracting what the reference model already believed.

---

## Q112. MUST REMEMBER - Why are reference log-probabilities subtracted?

Without the reference, the model would merely maximize the chosen-versus-rejected likelihood gap. The reference subtraction measures how preference tuning changes those likelihoods relative to the SFT starting point.

```text
current preference margin
      -
reference preference margin
```

This encodes the same “improve reward but stay anchored” idea that motivated KL regularization.

---

## Q113. MUST REMEMBER - What is the DPO sequence-log-probability shape flow?

For chosen completion:

```text
policy logits:          [B, T_w, V]
chosen token IDs:       [B, T_w]
chosen token log-probs: [B, T_w]
completion mask:        [B, T_w]
sequence log-prob:      [B]
```

The same is computed for:

- policy/chosen;
- policy/rejected;
- reference/chosen;
- reference/rejected.

Then:

```text
chosen log-ratio: [B]
rejected log-ratio:[B]
DPO margin:       [B]
losses:           [B]
mean loss:        scalar
```

---

## Q114. MUST REMEMBER - How is a completion's log-probability computed for DPO?

$$
\boxed{
\log\pi_\theta(y|x)
=
\sum_{t=1}^{T_c}
\log\pi_\theta(y_t|x,y_{<t})
}
$$

Prompt tokens are conditioning context and should not be included in the response log-probability sum.

**Memory liner:**

> Gather the log-probability of every response token, mask the prompt, then sum over the completion.

---

## Q115. INTERVIEW CLARIFICATION - Minimal DPO loss code?

```python
import torch
import torch.nn.functional as F

# Each input is [B]: sequence log-probability summed over completion tokens.
policy_chosen_logp = ...
policy_rejected_logp = ...
reference_chosen_logp = ...   # no gradient
reference_rejected_logp = ... # no gradient

chosen_log_ratio = policy_chosen_logp - reference_chosen_logp
rejected_log_ratio = policy_rejected_logp - reference_rejected_logp

preference_margin = beta * (chosen_log_ratio - rejected_log_ratio)
loss = -F.logsigmoid(preference_margin).mean()
```

---

## Q116. MUST REMEMBER - What does the DPO margin mean?

$$
m
=
\beta
\left[
\ell_\theta(x,y_w)-\ell_\theta(x,y_l)
\right].
$$

```text
m >> 0 → chosen strongly preferred by policy relative to reference
m = 0  → preference probability 0.5
m << 0 → rejected is relatively favored
```

---

## Q117. MUST REMEMBER - What does `beta` do in DPO?

From the KL-regularized derivation, `beta` controls the strength of the reference-model constraint/temperature of the preference model.

Conceptually:

```text
larger beta in underlying KL objective
→ stronger pressure to remain near reference

in the DPO logistic loss
→ also scales the preference margin and gradient sharpness
```

**Memory liner:**

> `beta` sets how aggressively preference evidence should move the policy relative to its reference anchor.

The lecture gives roughly `0.1` as an illustrative order of magnitude, not a universal value.

---

## Q118. MUST REMEMBER - Why is the DPO paper described as “Your language model is secretly a reward model”?

The KL-regularized optimal-policy equation lets the implicit reward be written as a function of policy/reference log-probability ratios.

Therefore an explicit neural reward model is not necessary in the final DPO training loss.

**Memory liner:**

> The policy's relative likelihood shift can play the role of a reward difference.

---

# 13. DPO derivation mental map

## Q119. MUST REMEMBER - What RL objective does DPO start from?

For each prompt `x`, consider:

$$
\boxed{
\max_\pi
\mathbb E_{y\sim\pi(\cdot|x)}[r(x,y)]
-
\beta D_{KL}
\big(\pi(\cdot|x)\|\pi_{ref}(\cdot|x)\big)
}
$$

This is the same reward-versus-reference trade-off introduced earlier in the lecture.

---

## Q120. MUST REMEMBER - What is the optimal policy form for this objective?

Solving the constrained optimization gives:

$$
\boxed{
\pi^*(y|x)
=
\frac{1}{Z(x)}
\pi_{ref}(y|x)
\exp\left(\frac{r(x,y)}{\beta}\right)
}
$$

where `Z(x)` is a prompt-dependent partition function ensuring probabilities sum to one.

**INTERVIEW CLARIFICATION:** The lecture presents this as the derived optimal form without walking through the Lagrange-multiplier calculus.

---

## Q121. MUST REMEMBER - Rearrange the optimal-policy equation to express reward.

Take logs:

$$
\log\pi^*(y|x)
=
\log\pi_{ref}(y|x)
+
\frac{r(x,y)}{\beta}
-
\log Z(x).
$$

Therefore:

$$
\boxed{
r(x,y)
=
\beta
\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}
+
\beta\log Z(x)
}
$$

---

## Q122. MUST REMEMBER - Why does the partition function disappear in pairwise preferences?

Bradley-Terry uses a reward difference for completions under the same prompt:

$$
r(x,y_w)-r(x,y_l).
$$

Both contain the same `beta log Z(x)`, so it cancels:

$$
\boxed{
r_w-r_l
=
\beta
\left[
\log\frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)}
-
\log\frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)}
\right]
}
$$

**Memory liner:**

> Same prompt means same normalization constant, so preference differences remove it.

---

## Q123. MUST REMEMBER - How does this produce the DPO objective?

1. Start with Bradley-Terry:

$$
P(y_w\succ y_l|x)=\sigma(r_w-r_l).
$$

2. Substitute the policy-based expression for `r_w-r_l`.
3. Replace the unknown optimal policy with trainable `pi_theta`.
4. Maximize preference likelihood, or minimize negative log-likelihood.

Result:

$$
\mathcal L_{DPO}
=-\mathbb E\log\sigma(\text{policy/reference preference margin}).
$$

**Memory liner:**

> DPO = Bradley-Terry preference likelihood after analytically replacing reward differences with policy log-ratios.

---

## Q124. MUST REMEMBER - Which DPO quantities receive gradients?

```text
pi_theta chosen/rejected log-probs → gradient
pi_ref chosen/rejected log-probs   → frozen / no gradient
preference labels                  → fixed
```

The reference model remains unchanged throughout preference tuning.

---

## Q125. SHOULD KNOW - Why initialize `pi_theta` from `pi_ref`?

They should represent the same SFT distribution before preference tuning. DPO then learns a controlled relative shift from that checkpoint.

Starting from an unrelated policy/reference pair would weaken the interpretation of the reference correction.

---

# 14. DPO versus PPO versus Best-of-N

## Q126. MUST REMEMBER - Compare the three methods.

| Property | PPO-based RLHF | DPO | Best-of-N |
|---|---|---|---|
| Changes model weights? | Yes | Yes | No |
| Data source | On-policy rollouts + reward model | Static preference pairs | Runtime samples |
| Separate reward model | Yes | No in training objective | Yes |
| Value model | Yes | No | No |
| Frozen reference | Usually yes | Yes | Not required for selection |
| Main cost | Complex RL training | Supervised preference training | Expensive inference |
| Negative signal | Through reward/advantage | Direct chosen-vs-rejected objective | Only candidate selection |
| Main lecture advantage | Stronger peak optimization | Simpler/stabler | No RL training |
| Main lecture drawback | Expensive and unstable | Distribution shift / sometimes lower peak quality | N-fold generation cost |

---

## Q127. MUST REMEMBER - Why can PPO outperform DPO?

The lecture's framing is that PPO can optimize against **fresh outputs from the current policy**, allowing on-policy improvement over the model's evolving distribution.

DPO trains on a fixed preference dataset, which may not match what the current policy produces after it moves.

**Memory liner:**

> PPO follows the current policy distribution; DPO is limited by the coverage of its static preference pairs.

---

## Q128. MUST REMEMBER - What distribution-shift issue can hurt DPO?

Preference pairs may have been generated by another model or an earlier checkpoint. The trainable policy may later visit different completion regions where the dataset supplies no direct ordering signal.

```text
preference-data distribution
          ≠
current policy's completion distribution
```

The lecture presents this as an intrinsic DPO challenge.

---

## Q129. SHOULD KNOW - How can DPO distribution mismatch be reduced?

The lecture suggests two broad routes:

- perform SFT on relevant preference-related data before DPO;
- generate preference candidates from the policy/model family being tuned and rate them.

Both require additional data work.

---

## Q130. MUST REMEMBER - When would the lecture favor DPO versus PPO?

### DPO

Use when:

- compute/engineering budget is limited;
- training stability and simplicity matter;
- good preference pairs already exist;
- near-top quality is sufficient.

### PPO

Use when:

- every increment of preference performance matters;
- the team has RL expertise;
- on-policy optimization is valuable;
- multi-model training cost is acceptable.

**Memory liner:**

> DPO is the practical supervised route; PPO is the heavier on-policy route for maximum control/performance.

---

## Q131. SHOULD KNOW - When would Best-of-N be reasonable?

Use it when:

- retraining is undesirable;
- request volume is low enough to afford multiple samples;
- the base policy can already generate some strong candidates;
- offline or high-value tasks justify extra inference compute.

---

## Q132. MUST REMEMBER - Why is ordinary SFT not equivalent to DPO?

SFT on the chosen response optimizes:

$$
-\log\pi_\theta(y_w|x).
$$

DPO optimizes a **relative chosen-versus-rejected shift corrected by the reference model**:

$$
-\log\sigma
\left(
\beta[(\log\pi_\theta(y_w)-\log\pi_\theta(y_l))
-(\log\pi_{ref}(y_w)-\log\pi_{ref}(y_l))]
\right).
$$

**Memory liner:**

> SFT raises the chosen answer; DPO raises it relative to both the rejected answer and the reference distribution.

---

# 15. End-to-end tensor-shape sheet

## Preference batch

```text
prompt IDs:             [B, T_p]
chosen completion IDs:  [B, T_w]
rejected completion IDs:[B, T_l]

prompt+chosen IDs:      [B, T_p + T_w]
prompt+rejected IDs:    [B, T_p + T_l]
```

Lengths may differ, so batches use padding and masks.

---

## Reward model

```text
chosen hidden states:   [B, T_p + T_w, D]
rejected hidden states: [B, T_p + T_l, D]
chosen pooled state:    [B, D]
rejected pooled state:  [B, D]
chosen reward:          [B]
rejected reward:        [B]
reward difference:      [B]
RM loss:                scalar
```

---

## Policy token probabilities

```text
policy logits:       [B, T_c, V]
log-softmax:         [B, T_c, V]
target token IDs:    [B, T_c]
gathered token logp: [B, T_c]
completion mask:     [B, T_c]
sequence logp:       [B]
```

Shape derivation:

```text
[B, T_c, V]
    gather target ID per position
→ [B, T_c]
    multiply response mask and sum over T_c
→ [B]
```

---

## Value and advantage

```text
policy hidden states: [B, T_c, D]
value predictions:    [B, T_c]
return/value targets: [B, T_c]
advantages:           [B, T_c]
```

---

## PPO

```text
new logp:           [B, T_c]
old logp:           [B, T_c]
ratio rho:          [B, T_c]
advantages:         [B, T_c]
unclipped objective:[B, T_c]
clipped objective:  [B, T_c]
masked mean:        scalar
```

---

## DPO

```text
policy chosen seq logp:   [B]
policy rejected seq logp: [B]
ref chosen seq logp:      [B]
ref rejected seq logp:    [B]
chosen log-ratio:          [B]
rejected log-ratio:        [B]
DPO preference margin:    [B]
DPO loss:                 scalar
```

---

# 16. Equations to memorize cold

## Autoregressive completion probability

$$
\boxed{
\pi_\theta(y|x)
=
\prod_t\pi_\theta(y_t|x,y_{<t})
}
$$

## Sequence log-probability

$$
\boxed{
\log\pi_\theta(y|x)
=
\sum_t\log\pi_\theta(y_t|x,y_{<t})
}
$$

## Sigmoid

$$
\boxed{
\sigma(z)=\frac{1}{1+e^{-z}}
}
$$

## Bradley-Terry preference probability

$$
\boxed{
P(y_w\succ y_l|x)
=
\sigma(r_w-r_l)
}
$$

## Reward-model loss

$$
\boxed{
\mathcal L_{RM}
=
-\mathbb E\log\sigma(r_w-r_l)
}
$$

## KL divergence

$$
\boxed{
D_{KL}(P\|Q)
=
\sum_iP(i)\log\frac{P(i)}{Q(i)}
}
$$

## Advantage mental equation

$$
\boxed{A_t\approx R_t-V(s_t)}
$$

## PPO ratio

$$
\boxed{
\rho_t
=
\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}
}
$$

## PPO clipped objective

$$
\boxed{
L^{clip}
=
\mathbb E_t
\min
\left(
\rho_tA_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
}
$$

## KL-regularized RL objective

$$
\boxed{
\max_\pi
\mathbb E[r(x,y)]
-
\beta D_{KL}(\pi\|\pi_{ref})
}
$$

## Best-of-N

$$
\boxed{
y^*=\arg\max_i r_\phi(x,y_i)}
$$

## DPO loss

$$
\boxed{
\mathcal L_{DPO}
=
-\mathbb E
\log\sigma
\left(
\beta
\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}
-
\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]
\right)
}
$$

## DPO optimal policy relation

$$
\boxed{
\pi^*(y|x)
=
\frac{1}{Z(x)}
\pi_{ref}(y|x)e^{r(x,y)/\beta}
}
$$

---

# 17. PyTorch-like implementation mental maps

## A. Compute completion sequence log-probabilities

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def completion_log_probs(
    logits: Tensor,
    input_ids: Tensor,
    completion_mask: Tensor,
) -> Tensor:
    """Return summed causal-LM log-probability for each completion.

    Args:
        logits: [B, T, V], where logits[:, t] predicts input_ids[:, t + 1].
        input_ids: [B, T].
        completion_mask: [B, T], equal to 1 on completion-token positions
            whose likelihood should be included and 0 on prompt/padding positions.

    Returns:
        sequence_logp: [B].
    """
    if logits.ndim != 3 or input_ids.ndim != 2 or completion_mask.ndim != 2:
        raise ValueError("Expected logits [B,T,V], IDs [B,T], mask [B,T].")
    if logits.shape[:2] != input_ids.shape or input_ids.shape != completion_mask.shape:
        raise ValueError("Batch and sequence dimensions must match.")

    # Position t predicts token at position t+1.
    next_token_logits = logits[:, :-1, :]       # [B, T-1, V]
    next_token_ids = input_ids[:, 1:]           # [B, T-1]
    next_token_mask = completion_mask[:, 1:]    # [B, T-1]

    log_probs = F.log_softmax(next_token_logits, dim=-1)
    token_logp = log_probs.gather(
        dim=-1,
        index=next_token_ids.unsqueeze(-1),
    ).squeeze(-1)                               # [B, T-1]

    return (token_logp * next_token_mask).sum(dim=-1)  # [B]
```

**Shape memory:**

```text
[B,T,V] --gather target token--> [B,T]
[B,T]   --mask prompt/padding--> [B,T]
[B,T]   --sum completion axis--> [B]
```

---

## B. Reward-model pairwise loss

```python
from torch import nn


class RewardModel(nn.Module):
    """Causal-transformer backbone plus scalar sequence reward head."""

    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state       # [B, T, D]

        # Index of final non-padding token for each sequence.
        last_index = attention_mask.long().sum(dim=-1) - 1  # [B]
        batch_index = torch.arange(
            hidden.size(0), device=hidden.device
        )
        pooled = hidden[batch_index, last_index]             # [B, D]
        reward = self.reward_head(pooled).squeeze(-1)        # [B]
        return reward


def reward_pair_loss(
    reward_model: RewardModel,
    chosen_ids: Tensor,
    chosen_mask: Tensor,
    rejected_ids: Tensor,
    rejected_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    chosen_reward = reward_model(chosen_ids, chosen_mask)       # [B]
    rejected_reward = reward_model(rejected_ids, rejected_mask) # [B]

    # -log sigmoid(r_chosen - r_rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    return loss, chosen_reward, rejected_reward
```

**Debugging invariant:**

```text
training is progressing when chosen_reward - rejected_reward
becomes positive on held-out preference pairs.
```

---

## C. Minimal DPO training step

```python
@torch.no_grad()
def reference_sequence_logps(
    reference_model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    completion_mask: Tensor,
) -> Tensor:
    out = reference_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    return completion_log_probs(out.logits, input_ids, completion_mask)


def policy_sequence_logps(
    policy: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    completion_mask: Tensor,
) -> Tensor:
    out = policy(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    return completion_log_probs(out.logits, input_ids, completion_mask)


def dpo_loss(
    policy_chosen_logp: Tensor,
    policy_rejected_logp: Tensor,
    ref_chosen_logp: Tensor,
    ref_rejected_logp: Tensor,
    beta: float,
) -> tuple[Tensor, Tensor]:
    """All log-probability tensors have shape [B]."""
    if beta <= 0:
        raise ValueError("beta must be positive")

    chosen_log_ratio = policy_chosen_logp - ref_chosen_logp
    rejected_log_ratio = policy_rejected_logp - ref_rejected_logp

    logits = beta * (chosen_log_ratio - rejected_log_ratio)  # [B]
    loss = -F.logsigmoid(logits).mean()

    # Useful metric: how often the learned margin has the desired sign.
    preference_accuracy = (logits > 0).float().mean()
    return loss, preference_accuracy
```

Full batch flow:

```text
prompt + chosen
    ├── policy → chosen policy sequence logp
    └── ref    → chosen reference sequence logp

prompt + rejected
    ├── policy → rejected policy sequence logp
    └── ref    → rejected reference sequence logp

four [B] tensors
    ↓
DPO margin [B]
    ↓
-logsigmoid
    ↓
scalar loss
```

---

## D. PPO conceptual training loop

This is intentionally high level because the lecture does not derive a production PPO implementation.

```python
for prompt_batch in prompt_loader:
    # 1. On-policy rollout from the current policy.
    with torch.no_grad():
        generated = policy.generate(
            **prompt_batch,
            do_sample=True,
            temperature=rollout_temperature,
        )

        # 2. Frozen evaluators/checkpoints.
        reward = reward_model.score(prompt_batch, generated)   # [B]
        old_logp = old_policy.token_logps(prompt_batch, generated)  # [B,Tc]
        ref_logp = reference_model.token_logps(prompt_batch, generated)  # [B,Tc]

        # 3. Value predictions and advantage/return targets.
        old_values = value_model(prompt_batch, generated)      # [B,Tc]
        advantages, returns = estimate_advantages(
            reward=reward,
            values=old_values,
            mask=completion_mask,
        )

    # 4. Several PPO optimization epochs on the collected rollout batch.
    for _ in range(num_ppo_epochs):
        new_logp = policy.token_logps(prompt_batch, generated) # [B,Tc]
        new_values = value_model(prompt_batch, generated)      # [B,Tc]

        ratio = torch.exp(new_logp - old_logp)
        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio, 1.0 - clip_eps, 1.0 + clip_eps
        ) * advantages

        policy_loss = -masked_mean(
            torch.minimum(unclipped, clipped), completion_mask
        )
        value_loss = masked_mean(
            (new_values - returns) ** 2, completion_mask
        )

        # Sampled reference drift penalty.
        kl_estimate = masked_mean(new_logp - ref_logp, completion_mask)

        total_loss = policy_loss + value_coef * value_loss + beta * kl_estimate
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # 5. Refresh old-policy snapshot for the next rollout/update cycle.
    old_policy.load_state_dict(policy.state_dict())
```

**Important:** Real PPO systems need careful distributed rollout collection, padding/masking, reward shaping, value clipping, numerical stabilization, and synchronization. The pseudocode is a conceptual map, not drop-in production code.

---

# 18. High-value distinctions interviewers test

## 1. SFT versus preference tuning

```text
SFT:
learn P(desired response | prompt)
from positive demonstrations

Preference tuning:
learn chosen > rejected
from relative judgments
```

---

## 2. Reward versus probability

```text
reward score r(x,y): unbounded scalar
preference probability: sigmoid(r_w - r_l)
```

A reward value such as `2.0` is not “200% probability.”

---

## 3. Pairwise training versus pointwise scoring

```text
train reward model using two candidates
use reward model on one candidate at a time
```

---

## 4. Reward versus value

```text
reward:
quality of completed response

value:
expected eventual quality from partial prefix
```

---

## 5. Reward versus advantage

```text
reward:
absolute learned score for outcome

advantage:
reward/return relative to expected value from that state
```

---

## 6. `pi_old` versus `pi_ref`

```text
pi_old:
previous PPO policy snapshot; local update constraint

pi_ref:
frozen SFT model; global alignment anchor
```

---

## 7. PPO clipping versus KL penalty

```text
clipping:
limits probability-ratio movement per PPO update

reference KL:
limits overall behavioral drift from SFT distribution
```

---

## 8. On-policy versus fixed preference training

```text
PPO:
current policy generates training rollouts

DPO:
training examples are fixed chosen/rejected pairs
```

---

## 9. DPO versus reward-model training

Both use a `-log sigmoid(difference)` shape, but the differences are different:

```text
Reward model:
r_phi(x,y_w) - r_phi(x,y_l)

DPO:
beta * [policy/reference log-ratio for y_w
        - policy/reference log-ratio for y_l]
```

---

## 10. Best-of-N versus policy improvement

Best-of-N improves the returned answer by selection, not by changing the policy's probability distribution.

---

# 19. Common interview traps and corrected answers

## Trap 1: “RLHF gives one reward to every token.”

**Correction:** The reward model in this lecture gives one completion-level score. Value/advantage machinery creates token-level training signals.

---

## Trap 2: “The reward model is trained together with PPO.”

**Correction:** In the lecture's two-stage RLHF recipe, reward-model training finishes first; the reward model is frozen during policy optimization.

---

## Trap 3: “PPO's ratio compares the policy with the SFT reference.”

**Correction:** The canonical clipped ratio compares current policy with `pi_old`, the previous policy snapshot. A separate KL term can compare current policy with the SFT reference.

---

## Trap 4: “KL is a symmetric distance.”

**Correction:** KL is directional and asymmetric.

---

## Trap 5: “If reward rises, alignment definitely improved.”

**Correction:** Reward can rise because of reward hacking or scale/normalization effects. External evaluation is required.

---

## Trap 6: “DPO has no reward concept.”

**Correction:** DPO removes the separately parameterized reward model from training, but its derivation expresses an implicit reward through policy/reference log-ratios.

---

## Trap 7: “DPO only maximizes the chosen answer.”

**Correction:** It increases the chosen response relative to the rejected response and corrects both using the frozen reference model.

---

## Trap 8: “Best-of-N is cheap if generations are parallel.”

**Correction:** It still consumes roughly `N` generations and waits for tail latency before reranking.

---

## Trap 9: “Preference tuning fixes missing knowledge.”

**Correction:** It mainly changes behavioral preference among model-supported completions. Broad knowledge/capability gaps may require better pre-training, mid-training, SFT, retrieval, or tools.

---

## Trap 10: “Reward scores have a universal calibrated meaning.”

**Correction:** Pairwise training primarily identifies score order/differences. Scale and offset are not automatically calibrated.

---

## Trap 11: “A large positive DPO chosen log-probability is sufficient.”

**Correction:** The relevant quantity is the chosen-versus-rejected **policy/reference margin**, not chosen likelihood alone.

---

## Trap 12: “PPO and DPO are parameter-efficient methods like LoRA.”

**Correction:** PPO/DPO are objectives/training algorithms. LoRA is a parameter-efficient update parameterization that may be combined with them.

---

# 20. Debugging questions worth preparing

## Q133. Reward-model accuracy rises, but PPO behavior gets worse. What might be wrong?

Potential explanations:

- reward-model validation data resembles training data but not policy rollouts;
- annotator shortcuts were learned;
- policy discovers adversarial high-reward outputs;
- reward scale is poorly normalized;
- KL penalty is too weak;
- capability regressions are not represented in the reward.

**Memory liner:**

> A good pairwise classifier is not automatically a robust objective under optimization.

---

## Q134. Reward-model loss is stuck near `log 2`. What does that suggest?

If `r_w-r_l≈0`, then:

$$
-\log\sigma(0)=\log 2.
$$

Possible issues:

- chosen/rejected labels are swapped or noisy;
- the pooling/head implementation is wrong;
- both sequences are accidentally identical after preprocessing;
- gradients are frozen;
- learning rate is too small;
- truncation removes the differentiating response content.

---

## Q135. Reward differences explode during training. Why is that risky?

The pairwise loss can be reduced by increasing score margins without producing calibrated rewards. Very large magnitudes can make later RL sensitive to reward scaling and numerical issues.

Check regularization, learning rate, normalization, and held-out ranking performance.

---

## Q136. PPO KL grows rapidly. What does it mean?

The policy is moving too far from the reference distribution.

Possible fixes:

- increase KL coefficient `beta`;
- reduce policy learning rate;
- reduce PPO epochs per rollout batch;
- tighten clip width `epsilon`;
- normalize/clip advantages;
- inspect reward hacking.

---

## Q137. PPO clip fraction is almost always zero. What does that suggest?

Updates may be too small:

- learning rate too low;
- advantages near zero;
- too few PPO epochs;
- excessive KL penalty;
- log-probability/mask bug.

---

## Q138. PPO clip fraction is almost always very high. What does that suggest?

Updates are too aggressive:

- learning rate too high;
- reward/advantage scale too large;
- too many PPO epochs;
- stale old-policy log-probabilities;
- incorrect ratio calculation.

---

## Q139. DPO loss decreases, but chosen responses become excessively verbose. Why?

Preference pairs may contain a verbosity bias. DPO faithfully learns the relative signal present in the data.

Possible actions:

- balance answer lengths;
- improve annotation rubric;
- evaluate length-controlled win rates;
- include concise chosen examples;
- inspect whether sequence log-probability summation introduces dataset correlations.

---

## Q140. DPO preference accuracy reaches 100% quickly, but downstream gains are small. Why?

- training pairs are too easy;
- preference dataset is narrow;
- model memorizes pair-specific artifacts;
- held-out prompts are out of distribution;
- reference already strongly prefers the winner;
- the important failures are absent from training data.

---

## Q141. All Best-of-N answers are bad. What is the bottleneck?

The proposal model lacks sufficient support/quality. Increase `N` only helps if the model sometimes produces a good candidate.

The solution may require better SFT/preference tuning, improved prompting, retrieval, or tools rather than more reranking.

---

## Q142. DPO chosen and rejected sequence log-probabilities look nearly identical. What should you inspect?

- completion masks;
- causal shift alignment;
- whether prompt tokens are mistakenly included;
- whether chosen and rejected IDs were mixed up;
- padding/truncation;
- frozen reference copy;
- tokenization differences.

---

# 21. Rapid-fire oral interview questions

You should be able to answer each in roughly 30-90 seconds.

1. Where does preference tuning sit relative to pre-training and SFT?
2. Why is SFT insufficient for all alignment goals?
3. What exactly is a preference pair?
4. Why is pairwise annotation often easier than writing demonstrations?
5. What negative signal does preference tuning provide?
6. Compare pointwise, pairwise, and listwise feedback.
7. How would you collect preference pairs from an SFT model?
8. Why should prompts match deployment traffic?
9. What annotation biases could corrupt preference data?
10. Map agent, state, action, policy, and reward to an LLM.
11. Why is the language-model policy a probability distribution over vocabulary tokens?
12. What is a rollout/completion?
13. Why is RLHF's reward signal sparse compared with SFT?
14. What are RLHF's two stages?
15. What does “human” mean in RLHF?
16. How is RLAIF different?
17. What does a reward model output?
18. Write the Bradley-Terry preference probability.
19. Derive the reward-model loss from maximum likelihood.
20. Why is a pairwise-trained reward model usable pointwise?
21. Why are absolute reward values not naturally calibrated?
22. How can a causal LM be converted into a reward model?
23. Why must reward be conditioned on the prompt?
24. What is reward hacking?
25. Why does policy optimization expose reward-model weaknesses?
26. Why keep the policy close to the SFT model?
27. Write the KL divergence equation.
28. Why is KL not a distance?
29. What does `beta` do in a KL-regularized objective?
30. Reward versus value function?
31. What is advantage?
32. Why does subtracting a value baseline reduce variance?
33. What sign of advantage increases token probability?
34. What does PPO stand for?
35. Define the PPO probability ratio.
36. Write the clipped PPO objective.
37. Explain clipping for positive advantage.
38. Explain clipping for negative advantage.
39. Distinguish `pi_old` from `pi_ref`.
40. Why does PPO need a value head?
41. Which model components are frozen in PPO-based RLHF?
42. Why is PPO called on-policy?
43. What makes PPO-RLHF difficult to tune?
44. Why is average reward not a sufficient metric?
45. Why is rollout diversity necessary?
46. What is Best-of-N?
47. What are Best-of-N's training and inference costs?
48. Why can parallel Best-of-N still hurt latency?
49. What happens if all candidates are poor?
50. What does DPO stand for?
51. What problem does DPO remove relative to PPO?
52. Which two models does DPO need?
53. Write the DPO loss.
54. How is completion sequence log-probability computed?
55. Why are prompt tokens masked from DPO sequence likelihood?
56. Why subtract reference-model log-probabilities?
57. What is the DPO preference margin?
58. What is `beta` doing in DPO?
59. Explain “the language model is secretly a reward model.”
60. State the KL-regularized optimal-policy equation.
61. Why does `Z(x)` cancel in a preference pair?
62. Explain the DPO derivation in four steps.
63. Why can PPO outperform DPO?
64. What distribution shift can hurt DPO?
65. DPO versus SFT on chosen answers?
66. PPO versus DPO versus Best-of-N?
67. Can LoRA be used with DPO?
68. Why does preference tuning not necessarily add factual knowledge?
69. What metrics would you monitor for reward-model training?
70. What metrics would you monitor for PPO?
71. What metrics would you monitor for DPO?
72. How would you detect reward hacking?

---

# 22. Memory liners

1. **Training stages:** Pre-training builds capability; SFT teaches instruction behavior; preference tuning chooses preferred behavior.
2. **Preference pair:** One prompt, one winner, one loser.
3. **Annotation:** Ranking is often easier than writing the perfect answer.
4. **Negative signal:** SFT says what to imitate; preferences also say what to avoid.
5. **RL mapping:** Prefix is state, next token is action, softmax is policy.
6. **Rollout:** A completion is a trajectory of token actions.
7. **Sparse reward:** One completion reward must supervise many token choices.
8. **RLHF:** Learn the judge, then optimize the actor against the judge.
9. **Reward model:** Prompt plus response maps to one scalar.
10. **Bradley-Terry:** Preference probability is sigmoid of reward difference.
11. **Reward loss:** Make chosen reward exceed rejected reward.
12. **Pairwise/pointwise:** Train with pairs; score one candidate later.
13. **Reward scale:** Ordering matters more naturally than absolute offset.
14. **Reward hacking:** High proxy score can coexist with failed true intent.
15. **Reference model:** Preserve what SFT already learned.
16. **KL:** Directional discrepancy between probability distributions.
17. **Reward vs value:** Reward scores the finish; value predicts the future from the prefix.
18. **Advantage:** Outcome minus what was expected.
19. **Positive advantage:** Raise sampled action probability.
20. **Negative advantage:** Lower sampled action probability.
21. **PPO ratio:** New probability divided by old probability.
22. **PPO clipping:** Improve probabilities, but not too much in one update.
23. **Old vs reference:** Old stabilizes each step; reference anchors the whole run.
24. **On-policy:** Train on what the current model just generated.
25. **Best-of-N:** Generate many, score many, return one.
26. **Best-of-N cost:** Skip RL training, pay at inference.
27. **DPO:** Direct supervised learning from preference pairs.
28. **DPO anchor:** Compare policy shifts against the frozen SFT model.
29. **DPO margin:** Chosen policy/reference gain minus rejected gain.
30. **DPO derivation:** Replace Bradley-Terry reward differences with policy/reference log-ratios.
31. **DPO limitation:** Static preference data may not cover the current policy.
32. **PPO limitation:** Better on-policy control, much harder systems and optimization problem.
33. **LoRA:** Parameter-efficient update mechanism, not a preference objective.
34. **Alignment boundary:** Preference tuning selects behavior; it does not automatically create missing capabilities.

---

# 23. Final previous-day checklist

## Preference-tuning fundamentals

- [ ] Explain pre-training → SFT → preference tuning.
- [ ] Distinguish imitation from preference optimization.
- [ ] Define a preference triple `(x, y_w, y_l)`.
- [ ] Explain why preference labels are easier to collect than perfect demonstrations.
- [ ] Compare pointwise, pairwise, and listwise feedback.
- [ ] Explain how prompts and candidate responses are collected.
- [ ] Explain why annotation guidelines and prompt distribution matter.

## RLHF and reward modelling

- [ ] Map RL agent/state/action/policy/reward to an LLM.
- [ ] Explain why completion reward is sparse.
- [ ] State RLHF's two stages.
- [ ] Write Bradley-Terry in exponential and sigmoid form.
- [ ] Derive `-log sigmoid(r_w-r_l)`.
- [ ] Explain pairwise training but pointwise reward inference.
- [ ] Derive reward-model tensor shapes.
- [ ] Explain reward dimensions, calibration, and noisy labels.
- [ ] Explain reward hacking with an intuitive example.

## KL, value, and advantage

- [ ] Write `D_KL(P||Q)` and state its properties.
- [ ] Explain why the SFT reference model is needed.
- [ ] Distinguish completion reward from token-level value.
- [ ] Explain advantage as reward/return minus baseline.
- [ ] Explain why baselines reduce variance.
- [ ] State that exact GAE is outside this lecture.

## PPO

- [ ] Define `pi_theta`, `pi_old`, and `pi_ref`.
- [ ] Write the PPO probability ratio.
- [ ] Write the clipped PPO objective.
- [ ] Explain positive- and negative-advantage clipping.
- [ ] Explain PPO clipping versus reference KL.
- [ ] List the policy, old policy, reference, reward, and value components.
- [ ] Explain why PPO is on-policy.
- [ ] List PPO-RLHF's engineering and optimization challenges.
- [ ] State that GRPO is deferred to Lecture 6.

## Best-of-N

- [ ] Explain the generate-score-select pipeline.
- [ ] State that it changes no model weights.
- [ ] Explain inference cost and tail latency.
- [ ] Explain why it fails if all proposals are poor.

## DPO

- [ ] Write the DPO loss without notes.
- [ ] Compute sequence log-probability only over completion tokens.
- [ ] Explain policy/reference log-ratios.
- [ ] Explain the role of `beta`.
- [ ] State the optimal KL-regularized policy.
- [ ] Rearrange reward as a policy/reference log-ratio.
- [ ] Explain why the partition function cancels.
- [ ] Compare DPO with chosen-only SFT.
- [ ] Compare DPO, PPO, and Best-of-N.
- [ ] Explain DPO's static-data distribution-shift limitation.

---

# 24. One-minute recap

```text
A base model learns language.
SFT teaches it how to answer instructions.
Preference tuning teaches it which plausible answers are more desirable.

Preference data usually contains:
    prompt x
    chosen response y_w
    rejected response y_l

A reward model is trained with Bradley-Terry:
    P(y_w > y_l | x) = sigmoid(r_w - r_l)
    L_RM = -log sigmoid(r_w - r_l)

RLHF then samples a completion from the current policy,
scores it with the frozen reward model, estimates token-level values
and advantages, and updates the policy with PPO.

PPO uses:
    rho_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t)

and clips probability updates so positive-advantage actions become
more likely and negative-advantage actions become less likely,
but neither changes too far in one step.

A frozen SFT reference and KL penalty reduce reward hacking,
catastrophic drift, and instability.

Best-of-N avoids RL:
    generate N → reward-score N → return argmax
but pays N-fold inference cost.

DPO avoids explicit reward-model/PPO training:
    compare chosen and rejected policy/reference log-ratios
    optimize a direct logistic preference loss.

PPO is powerful but complex and on-policy.
DPO is simpler and supervised but depends heavily on static
preference-data coverage.
```

---

# 25. Explicit lecture boundaries

The following are mentioned or naturally adjacent but are **not fully taught in this lecture**:

- Exact Generalized Advantage Estimation equations.
- Full policy-gradient theorem derivation.
- Production PPO value clipping, entropy bonuses, and distributed rollout systems.
- GRPO equations and reasoning-model training; deferred to Lecture 6.
- Detailed LLM-as-a-judge methodology; deferred to a later lecture.
- Constitutional AI and detailed RLAIF pipelines.
- ORPO, IPO, KTO, SimPO, and other DPO-family variants.
- Formal multi-objective reward aggregation.

Keep those topics in separate lecture notes rather than silently merging them into Lecture 5.
