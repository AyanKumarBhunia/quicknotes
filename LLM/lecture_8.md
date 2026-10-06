# CME 295 Lecture 8 - LLM Evaluation, LLM-as-a-Judge, Agent Evaluation, and Benchmarks

> **Scope:** This note is grounded in the supplied **CME 295 Lecture 8 transcript**. It adds standard equations, implementation mental models, and intuitive explanations only when they directly clarify a topic taught in this lecture. Details not explicitly derived in class are labelled **INTERVIEW CLARIFICATION**.
>
> **Goal:** Previous-day revision for ML/AI Research Scientist interviews: important questions, distinctions, equations, evaluation dataflows, failure modes, benchmark mechanics, implementation ideas, and memory liners.
>
> **Transcript normalization:** The automatic transcript occasionally renders technical names incorrectly. This note uses the standard terms **Cohen's kappa**, **Fleiss' kappa**, **Krippendorff's alpha**, **METEOR**, **BLEU**, **LLM-as-a-judge**, **factuality**, **AIME**, **PIQA**, **SWE-bench**, **HarmBench**, **tau-bench**, **Pareto frontier**, and **Goodhart's law** while preserving the lecture's intended content.

**Source transcript:** Stanford CME 295, Lecture 8 - LLM Evaluation  
`https://www.youtube.com/watch/8fNP4N46RRo`

---

## Lecture spine

```text
WHAT DOES "EVALUATE AN LLM" MEAN?
    many possible dimensions
        output quality
        factuality / relevance / usefulness
        tone / format / safety
        latency / price / availability
    lecture focus: output quality

HUMAN EVALUATION
    prompt -> model response -> human rating
    challenges
        subjectivity
        inter-rater disagreement
        cost and latency
    monitor agreement beyond chance
        Cohen's kappa
        Fleiss' kappa
        Krippendorff's alpha

REFERENCE-BASED METRICS
    candidate response vs fixed reference
        METEOR
        BLEU
        ROUGE
    strengths
        cheap and repeatable after references exist
    weaknesses
        paraphrase / style sensitivity
        imperfect human correlation
        references still cost money

LLM-AS-A-JUDGE
    prompt + response + rubric
        -> rationale
        -> score / pass-fail
    or pairwise
        prompt + response A + response B
        -> preferred response
    use structured output for parseability
    failure modes
        position bias
        verbosity bias
        self-enhancement bias
        mismatch with human preferences
    best practices
        crisp criteria
        simple scales
        rationale before verdict
        low temperature
        calibrate against humans

FACTUALITY EVALUATION
    response
        -> extract atomic facts
        -> retrieve evidence / search
        -> verify each fact
        -> weighted aggregation

AGENT EVALUATION
    user query
        -> tool selection + arguments
        -> tool execution
        -> result synthesis
    evaluate and debug every stage, not only the final answer

BENCHMARK FAMILIES
    knowledge      -> MMLU
    math reasoning -> AIME
    commonsense    -> PIQA
    coding         -> SWE-bench
    safety         -> HarmBench
    tool agents    -> tau-bench

INTERPRETING BENCHMARKS
    models have profiles, not one universal quality number
    compare quality with cost / safety / context
    Pareto frontier
    benchmark contamination
    Goodhart's law
    use-case-specific evaluation remains necessary
```

---

## Notation

```text
N             number of evaluated examples
R             number of human raters
C             number of answer classes / choices
K             number of repeated attempts
n             number of sampled attempts used for an estimator
c             number of successful attempts among n
P_o           observed inter-rater agreement
P_e           expected agreement by chance
kappa         chance-corrected agreement
q             user prompt / query
y             candidate model response
y_A, y_B      two responses in a pairwise evaluation
J             judge model
s             judge score or binary verdict
m             number of matched unigrams
ch            number of matched contiguous chunks in METEOR
F             number of extracted factual claims
z_i           correctness indicator for fact i, in {0,1}
alpha_i       importance weight for fact i
S             number of steps in an agent trace
N_tools       number of available tools
K_tools       number of tools retained by a router
```

### Priority legend

- **MUST REMEMBER:** answer immediately and derive the main equation or dataflow.
- **SHOULD KNOW:** explain the trade-off or failure mode clearly.
- **LECTURE BOUNDARY:** mentioned but not fully derived in this lecture.
- **INTERVIEW CLARIFICATION:** standard detail added to make the lecture interview-ready.

---

# 1. The evaluation mental model

## Q1. MUST REMEMBER - What is the central message of Lecture 8?

If you cannot measure model behavior, you cannot reliably decide what to improve.

```text
model change
    -> evaluation
    -> evidence about what improved or regressed
    -> next model / data / prompt change
```

**Memory liner:**

> Evaluation closes the loop between model development and evidence-based improvement.

---

## Q2. MUST REMEMBER - What does "LLM evaluation" potentially include?

The lecture lists two broad families:

1. **Output-quality evaluation:** coherence, factuality, relevance, usefulness, tone, safety, and task success.
2. **System evaluation:** latency, price, uptime, and related serving properties.

This lecture mainly studies the first family.

**Memory liner:**

> Model quality and system quality are both evaluation, but this lecture focuses on the quality of the response.

---

## Q3. MUST REMEMBER - Why is LLM output evaluation intrinsically difficult?

An LLM produces free-form outputs across very different domains:

```text
natural-language answers
code
mathematical reasoning
summaries
creative text
tool calls and agent traces
```

A metric that works for one output type may be meaningless for another.

**Memory liner:**

> Free-form, multi-domain output prevents one universal metric from capturing all desirable behavior.

---

## Q4. MUST REMEMBER - What are the three evaluator families developed in the lecture?

| Evaluator | Main idea | Main strength | Main weakness |
|---|---|---|---|
| Human | Ask people to rate responses | Closest to intended human value | Slow, expensive, subjective |
| Reference/rule metric | Compare against a fixed answer | Cheap and repeatable | Brittle to valid paraphrases |
| LLM-as-a-judge | Ask another LLM to apply a rubric | Scalable and provides a rationale | Judge bias and calibration risk |

**Memory liner:**

> Humans are the target, rules are cheap proxies, and LLM judges are flexible learned proxies.

---

## Q5. SHOULD KNOW - What exactly is the unit being evaluated?

The basic lecture setup is:

```text
prompt q
   +
model response y
   +
evaluation criterion c
   -> evaluator
   -> verdict / score
```

A response cannot generally be judged without the prompt because correctness and usefulness are conditional on what the user asked.

**Memory liner:**

> Evaluate a response in the context of its prompt and a clearly stated criterion.

---

## Q6. MUST REMEMBER - Why must evaluation criteria be explicit?

The same response can score differently along different dimensions:

```text
factually correct but unfriendly
helpful but verbose
well formatted but irrelevant
safe but overly refusing
```

A rating is uninterpretable unless the dimension is specified.

**Memory liner:**

> "Good" is not a metric until you define what good means.

---

## Q7. SHOULD KNOW - What is the difference between an evaluation and a benchmark?

- An **evaluation** is the general procedure for measuring behavior on a chosen dataset and criterion.
- A **benchmark** packages prompts, expected outputs or verifiers, and a scoring protocol so that systems can be compared consistently.

The lecture later groups benchmarks by knowledge, reasoning, coding, safety, and tool use.

**Memory liner:**

> An evaluation is a measurement process; a benchmark is a standardized evaluation package.

---

## Q8. MUST REMEMBER - Why should a model not be summarized by one number?

Different models can have different strength profiles:

```text
model A: stronger coding, slower and expensive
model B: weaker reasoning, cheap and fast
model C: safer, but refuses more often
```

The useful model depends on the target use case.

**Memory liner:**

> Model quality is a vector of capabilities and costs, not a single scalar.

---

# 2. Human evaluation

## Q9. MUST REMEMBER - Why is human evaluation the idealized evaluator?

The ultimate product target is often human satisfaction or human-defined correctness. Therefore the most direct procedure is:

```text
prompt -> response -> human judgment
```

Collecting judgments over many prompts gives an empirical estimate of model quality according to people.

**Memory liner:**

> Human evaluation directly measures the behavior the product is trying to serve.

---

## Q10. MUST REMEMBER - What are the two main problems with human evaluation in the lecture?

1. **Subjectivity:** different raters can interpret usefulness or quality differently.
2. **Operational cost:** rating many outputs is slow and expensive.

**Memory liner:**

> Humans are valuable evaluators, but they are neither perfectly consistent nor cheap to scale.

---

## Q11. MUST REMEMBER - Why are detailed rater guidelines necessary?

Without explicit criteria, raters can apply different private definitions of quality. Guidelines should define:

```text
what counts as success
what counts as failure
how to treat partial correctness
how to handle uncertainty
examples near the decision boundary
```

**Memory liner:**

> Clear rubrics reduce label noise by aligning the mental task performed by different raters.

---

## Q12. MUST REMEMBER - What is observed agreement?

For two raters over `N` examples:

$$
P_o
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[r_{A,i}=r_{B,i}]
$$

It is simply the fraction of examples on which the raters produce the same label.

**Memory liner:**

> Observed agreement counts how often the raters match.

---

## Q13. MUST REMEMBER - Why is raw agreement insufficient?

Some agreement occurs by chance, especially when both raters frequently choose the same majority label.

For binary labels, if rater A says positive with probability `p_A` and rater B says positive with probability `p_B`, chance agreement is:

$$
P_e
=
p_Ap_B
+
(1-p_A)(1-p_B)
$$

The two terms mean:

```text
both randomly say positive
+
both randomly say negative
```

**Memory liner:**

> Raw agreement is meaningful only relative to how much agreement the label marginals would create by chance.

---

## Q14. MUST REMEMBER - Why can two random binary raters already agree 50% of the time?

If:

$$
p_A=p_B=0.5
$$

then:

$$
P_e
=0.5^2+0.5^2
=0.5
$$

Therefore an observed agreement of 50% can contain no real coordination at all.

**Memory liner:**

> Fifty-percent binary agreement can be pure chance.

---

## Q15. MUST REMEMBER - What is Cohen's kappa?

For two raters:

$$
\boxed{
\kappa
=
\frac{P_o-P_e}{1-P_e}
}
$$

Interpretation:

```text
kappa = 1   perfect observed agreement
kappa = 0   no better than chance agreement
kappa < 0   worse than the chance baseline
```

**Memory liner:**

> Kappa measures the fraction of possible beyond-chance agreement that was actually achieved.

---

## Q16. MUST REMEMBER - Derive why kappa equals one under perfect agreement.

If `P_o=1`:

$$
\kappa
=
\frac{1-P_e}{1-P_e}
=1
$$

If `P_o=P_e`:

$$
\kappa=0
$$

**Memory liner:**

> Kappa subtracts the chance floor and normalizes by the remaining headroom.

---

## Q17. SHOULD KNOW - Cohen's kappa vs Fleiss' kappa vs Krippendorff's alpha?

The lecture names all three as chance-aware agreement families.

**INTERVIEW CLARIFICATION:**

| Metric | Typical use |
|---|---|
| Cohen's kappa | Two raters assigning categorical labels |
| Fleiss' kappa | More than two raters assigning categorical labels |
| Krippendorff's alpha | Flexible rater counts, missing ratings, and multiple measurement scales |

The exact formulas for Fleiss' kappa and Krippendorff's alpha are outside the lecture.

**Memory liner:**

> Choose the agreement statistic to match the number of raters and type of labels.

---

## Q18. MUST REMEMBER - How is inter-rater agreement used operationally?

It is a health metric for the annotation process. If agreement is too low:

```text
inspect disagreements
    -> clarify ambiguous rubric language
    -> run rater calibration sessions
    -> add examples
    -> relabel or continue only after alignment improves
```

**Memory liner:**

> Low agreement is often a data-process problem, not merely a bad score to report.

---

## Q19. SHOULD KNOW - Pointwise human rating vs pairwise preference?

- **Pointwise:** score one response on a scale.
- **Pairwise:** compare two responses and choose the better one.

The lecture's judge discussion later emphasizes that binary or pairwise decisions are often easier and less noisy than fine-grained scales.

**Memory liner:**

> Comparing two concrete responses is often easier than assigning an absolute numerical quality score.

---

## Q20. MUST REMEMBER - Does the lecture abandon human evaluation after introducing automated evaluators?

No. Human ratings remain the target used to:

- design rubrics;
- calibrate LLM judges;
- measure agreement;
- check that automated improvements correspond to real user value.

**Memory liner:**

> Automation reduces human volume; it does not remove the need for human grounding.

---

# 3. Fixed-reference and rule-based metrics

## Q21. MUST REMEMBER - What is the fixed-reference evaluation setup?

Humans write ideal outputs once for a fixed prompt set. Every model version is then compared against those references using an automatic metric.

```text
prompt q_i
reference y_i^*
candidate y_i
    -> overlap / ordering metric
    -> score
```

**Memory liner:**

> Pay for references once, then reuse them for cheap repeated model evaluation.

---

## Q22. SHOULD KNOW - What advantage do fixed references give during model iteration?

They make evaluation repeatable across checkpoints because every model is scored against the same target set rather than requiring new human judgments after every change.

**Memory liner:**

> Fixed references turn evaluation into a reproducible regression test.

---

## Q23. MUST REMEMBER - What is METEOR trying to capture?

METEOR combines:

1. unigram matching between prediction and reference;
2. precision and recall of those matches;
3. a penalty when matched words appear in fragmented or different order.

The lecture also notes that implementations may allow stemming and synonym matches.

**Memory liner:**

> METEOR rewards matched content and penalizes fragmented ordering.

---

## Q24. MUST REMEMBER - What are unigram precision and recall in METEOR?

If `m` unigrams match:

$$
P=\frac{m}{|y|}
$$

$$
R=\frac{m}{|y^*|}
$$

where `y` is the candidate and `y^*` is the reference.

**Memory liner:**

> Precision normalizes matches by candidate length; recall normalizes them by reference length.

---

## Q25. SHOULD KNOW - What is the weighted harmonic term used by METEOR?

**INTERVIEW CLARIFICATION:** A common form consistent with the lecture is:

$$
F_{mean}
=
\frac{PR}{\alpha P+(1-\alpha)R}
$$

The hyperparameter `alpha` controls the relative emphasis on precision and recall.

**Memory liner:**

> METEOR first turns overlap precision and recall into one weighted harmonic score.

---

## Q26. MUST REMEMBER - What is METEOR's fragmentation intuition?

Let:

- `m` be the number of matched unigrams;
- `ch` be the number of contiguous matched chunks.

Many short chunks indicate that the same words exist but their ordering is fragmented. Fewer, longer chunks indicate better ordering alignment.

**INTERVIEW CLARIFICATION:** A standard penalty form is:

$$
Penalty
=
\gamma
\left(\frac{ch}{m}\right)^{\beta}
$$

**Memory liner:**

> The more scattered the matched words are, the larger the ordering penalty.

---

## Q27. MUST REMEMBER - What is the overall METEOR structure?

$$
\boxed{
METEOR
=
(1-Penalty)F_{mean}
}
$$

Higher is better.

The lecture emphasizes that the exact formula contains hand-chosen hyperparameters and is therefore partly a designed recipe.

**Memory liner:**

> Content match gives the base score; fragmentation reduces it.

---

## Q28. MUST REMEMBER - What is BLEU conceptually?

BLEU is primarily a modified n-gram precision metric for translation:

```text
how many candidate n-grams
are supported by the reference(s)?
```

It also applies a brevity penalty so a very short candidate cannot obtain a deceptively high precision score.

**Memory liner:**

> BLEU rewards reference-supported n-grams and penalizes under-length translations.

---

## Q29. SHOULD KNOW - Why does BLEU need a brevity penalty?

A candidate containing only one highly reliable word may have high n-gram precision while failing to translate most of the source. The brevity penalty discourages this loophole.

**INTERVIEW CLARIFICATION:** For candidate length `c` and effective reference length `r`:

$$
BP=
\begin{cases}
1,&c>r\\
\exp(1-r/c),&c\le r
\end{cases}
$$

**Memory liner:**

> Precision alone rewards saying too little; the brevity penalty charges for missing length.

---

## Q30. SHOULD KNOW - What is the standard BLEU equation?

**INTERVIEW CLARIFICATION:** If `p_n` is modified precision for n-grams of order `n`:

$$
\boxed{
BLEU
=
BP\exp\left(
\sum_{n=1}^{N}w_n\log p_n
\right)
}
$$

The lecture does not derive this full equation; it focuses on precision and brevity intuition.

**Memory liner:**

> BLEU is a brevity-adjusted geometric mean of modified n-gram precisions.

---

## Q31. MUST REMEMBER - What is ROUGE at the lecture level?

ROUGE is a family of reference-overlap metrics often used for summarization.

**INTERVIEW CLARIFICATION:** Common variants include:

- `ROUGE-N`: n-gram overlap, often recall-oriented;
- `ROUGE-L`: overlap based on the longest common subsequence.

**Memory liner:**

> BLEU is classically translation-oriented; ROUGE is a family commonly used for summarization.

---

## Q32. MUST REMEMBER - Why do reference metrics fail on valid paraphrases?

These sentences can express the same meaning with little lexical overlap:

```text
A plush teddy bear can comfort a child during bedtime.
Soft stuffed bears often help children feel safe as they fall asleep.
```

A lexical metric may score the second poorly even though a human sees equivalent meaning.

**Memory liner:**

> Semantic equivalence does not imply n-gram overlap.

---

## Q33. MUST REMEMBER - Why can a carefully designed rule metric still correlate poorly with humans?

It optimizes a manually chosen proxy such as word overlap or order, while humans evaluate broader semantics, usefulness, factuality, tone, and context.

**Memory liner:**

> Hand-designed overlap features capture only a narrow slice of human judgment.

---

## Q34. MUST REMEMBER - Do rule-based metrics remove human cost entirely?

No. Humans are still needed to write or validate the references, and metric hyperparameters were often designed or calibrated with human judgments.

**Memory liner:**

> Reference metrics amortize human effort; they do not make the human target disappear.

---

## Q35. SHOULD KNOW - When are reference metrics still useful?

They remain useful when:

- the acceptable output is tightly constrained;
- lexical overlap is genuinely important;
- exact or near-exact answers exist;
- you need a fast, deterministic regression signal.

**Memory liner:**

> Brittle metrics are valuable when the task itself is narrow and deterministic.


# 4. LLM-as-a-judge

## Q36. MUST REMEMBER - What is LLM-as-a-judge?

A separate LLM evaluates a candidate response according to an explicit criterion.

```text
user prompt q
candidate response y
evaluation rubric c
        -> judge LLM J
        -> rationale + score
```

**Memory liner:**

> Use a capable language model as a flexible learned evaluator of another model's response.

---

## Q37. MUST REMEMBER - What information should be given to the judge?

At minimum:

1. the original user prompt;
2. the model response;
3. the criterion or rubric;
4. the required output format.

A judge that sees only the response may not know whether it actually answered the request.

**Memory liner:**

> The judge needs the task, the answer, the standard, and the response schema.

---

## Q38. MUST REMEMBER - Why ask for the rationale before the score?

The lecture reports an empirical benefit from letting the model analyze the response before committing to a verdict.

```text
evidence / critique first
        ->
final score second
```

This resembles the reasoning-before-answer pattern discussed in the reasoning lecture.

**Memory liner:**

> Let the judge inspect and explain before it commits to the label.

---

## Q39. MUST REMEMBER - Why is ordinary prompting not enough for a production judge?

Free-form generation does not guarantee a parseable score. The model may:

- omit the score;
- change field names;
- add unexpected prose;
- return an invalid JSON object.

The lecture recommends constrained or guided decoding, commonly exposed as **structured output**.

**Memory liner:**

> A judge is useful to software only when its verdict is machine-parseable.

---

## Q40. MUST REMEMBER - What is a pointwise judge?

It evaluates one response independently:

$$
J(q,y,c)
\rightarrow
(rationale, score)
$$

Example outputs:

```json
{
  "rationale": "The response answers the question but contains one unsupported claim.",
  "pass": false
}
```

**Memory liner:**

> Pointwise judging asks: how good is this one response?

---

## Q41. MUST REMEMBER - What is a pairwise judge?

It compares two responses for the same prompt:

$$
J(q,y_A,y_B,c)
\rightarrow
\{A,B,tie\}
$$

Pairwise evaluation is useful when an absolute score is difficult but relative preference is easier.

**Memory liner:**

> Pairwise judging asks: which of these two responses better satisfies the rubric?

---

## Q42. SHOULD KNOW - How can pairwise LLM judging create preference data?

Generate two completions for a prompt, ask a judge which is better, and record:

```text
(prompt, chosen_response, rejected_response)
```

Such synthetic preference pairs can support reward-model training or direct preference optimization, subject to judge-quality checks.

**Memory liner:**

> A pairwise judge can turn model samples into chosen-versus-rejected training examples.

---

## Q43. MUST REMEMBER - What are the two main benefits emphasized for LLM judges?

1. They do not require a reference answer for every example.
2. They can provide a natural-language rationale explaining the score.

**Memory liner:**

> LLM judges are reference-free and interpretable relative to opaque overlap scores.

---

## Q44. MUST REMEMBER - Is an LLM judge a ground-truth oracle?

No. It is another model and can be wrong, biased, inconsistent, or poorly aligned with human preferences.

**Memory liner:**

> A judge model is a scalable proxy, not truth itself.

---

## Q45. SHOULD KNOW - What model should be used as a judge?

The lecture's practical guidance is usually to choose a model that:

- is not exactly the same generator being evaluated;
- has enough capacity;
- has strong reasoning and instruction-following ability.

A larger judge is a common choice, not a mathematical requirement.

**Memory liner:**

> The judge should be capable enough to distinguish quality differences and sufficiently independent from the generator.

---

# 5. Judge biases and failure modes

## Q46. MUST REMEMBER - What is position bias?

In pairwise evaluation, the judge may prefer the response shown first or second because of ordering rather than quality.

```text
Judge(q, A, B) -> A
Judge(q, B, A) -> B
```

The preference changed when only presentation order changed.

**Memory liner:**

> Position bias means ordering affects the verdict.

---

## Q47. MUST REMEMBER - How can position bias be mitigated?

Run the comparison in both orders:

```text
A vs B
B vs A
```

Then retain order-invariant decisions, aggregate several runs, or mark inconsistent cases for additional review.

**Memory liner:**

> Swap the candidates; real quality should survive the swap.

---

## Q48. MUST REMEMBER - What is verbosity bias?

The judge may prefer a longer response merely because it contains more detail, even when the concise response is equally or more correct.

**Memory liner:**

> More words can look more impressive without adding more value.

---

## Q49. MUST REMEMBER - How can verbosity bias be mitigated?

The lecture gives three strategies:

1. explicitly tell the judge not to reward length by itself;
2. provide in-context examples where concise answers win;
3. compare pointwise quality and optionally apply a length-related penalty.

**Memory liner:**

> Make concision part of the rubric rather than letting length silently dominate.

---

## Q50. MUST REMEMBER - What is self-enhancement bias?

A model may prefer responses produced by itself because those responses resemble sequences its own probability distribution considers likely.

**Memory liner:**

> A model can mistake stylistic familiarity with correctness.

---

## Q51. MUST REMEMBER - How can self-enhancement bias be reduced?

Use a different judge from the generator, preferably one with sufficient capacity and independently validate its agreement with human labels.

This does not eliminate shared-data or shared-style biases across model families.

**Memory liner:**

> Separate generation from evaluation, then calibrate the evaluator.

---

## Q52. SHOULD KNOW - Are position, verbosity, and self-enhancement the only judge biases?

No. The lecture explicitly says the list is not exhaustive. A judge can also have a systematic mismatch with the target human preference or policy.

**Memory liner:**

> Judge bias includes any systematic difference between the automated verdict and the intended evaluation target.

---

## Q53. MUST REMEMBER - Why should judge guidelines be crisp?

A vague criterion such as "is this good?" permits the judge to choose its own hidden priorities. A crisp rubric should spell out:

```text
criterion
success conditions
failure conditions
edge cases
what not to reward
```

**Memory liner:**

> Reduce judge variance by reducing rubric ambiguity.

---

## Q54. MUST REMEMBER - Why does the lecture often prefer binary scores?

Pass/fail or A/B decisions are simpler for both humans and LLM judges than fine-grained scales. Fewer categories reduce ambiguity between neighboring ratings.

**Memory liner:**

> Use the simplest scale that preserves the decision you need.

---

## Q55. MUST REMEMBER - What is the rationale-first judge template?

```text
1. Read the prompt, response, and rubric.
2. Identify evidence for success and failure.
3. Produce a concise rationale.
4. Emit the final structured verdict.
```

**Memory liner:**

> Analyze first, label last.

---

## Q56. MUST REMEMBER - Why use low temperature for evaluation?

Evaluation should be reproducible. A low temperature reduces sampling variability so rerunning the same evaluation is less likely to produce unrelated verdicts.

The lecture gives values around `0.1` or `0.2` as common examples.

**Memory liner:**

> Creativity helps generation; stability helps measurement.

---

## Q57. MUST REMEMBER - How should an LLM judge be calibrated against humans?

On a representative subset:

```text
collect human ratings
collect judge ratings
compare disagreements / correlation
inspect systematic failure slices
revise rubric or judge
repeat
```

The goal is not merely a high judge score; it is alignment between judge scores and the human target.

**Memory liner:**

> Validate the proxy against the people it is supposed to approximate.

---

## Q58. MUST REMEMBER - What is the danger of overoptimizing against an LLM judge?

A model can learn to exploit patterns that increase judge score without improving human-perceived quality. This is proxy overoptimization.

```text
optimize model for judge score
        ->
judge score rises
        ->
human value may plateau or fall
```

**Memory liner:**

> Improving a proxy is useful only while the proxy remains aligned with the real objective.

---

## Q59. SHOULD KNOW - Can a plausible rationale make an incorrect judge score trustworthy?

No. A rationale improves inspectability but can itself be fluent and wrong. The verdict and rationale should be checked against human labels and known examples.

**Memory liner:**

> Explanation increases debuggability, not guaranteed correctness.

---

## Q60. MUST REMEMBER - Compare human, rule-based, and LLM judging.

| Property | Human | Reference/rule | LLM judge |
|---|---:|---:|---:|
| Handles semantics | Strong | Weak to medium | Stronger, but imperfect |
| Needs a reference | No | Usually yes | No |
| Gives rationale | Yes | Usually no | Yes |
| Scales cheaply | No | Yes | More than humans |
| Subject to bias | Yes | Yes | Yes |
| Deterministic | Not necessarily | Usually | Only approximately; use low temperature |

**Memory liner:**

> Every evaluator trades fidelity, cost, flexibility, and failure modes differently.

---

# 6. Evaluation dimensions and factuality

## Q61. MUST REMEMBER - What two broad response-quality groups does the lecture introduce?

### Task-performance dimensions

- usefulness;
- factuality;
- relevance;
- success on the requested task.

### Alignment and presentation dimensions

- tone;
- style;
- output format;
- safety.

**Memory liner:**

> Evaluate both whether the model did the task and how it behaved while doing it.

---

## Q62. SHOULD KNOW - Why should usefulness, relevance, and factuality be separate criteria?

A response can be:

- relevant but factually wrong;
- factual but not useful;
- useful in general but unrelated to the prompt.

Combining them prematurely hides the reason for failure.

**Memory liner:**

> Separate dimensions make evaluation actionable.

---

## Q63. MUST REMEMBER - Why is factuality harder than a simple whole-response pass/fail label?

A long response can contain many claims. Some may be correct and others incorrect. Labelling the entire response as false discards this nuance.

**Memory liner:**

> Factuality is naturally claim-level before it is response-level.

---

## Q64. MUST REMEMBER - What is the lecture's factuality pipeline?

```text
model response
    -> extract a list of factual claims
    -> verify each claim against evidence
    -> optionally weight claims by importance
    -> aggregate into one score
```

**Memory liner:**

> Decompose, verify, weight, aggregate.

---

## Q65. MUST REMEMBER - Why extract atomic facts?

A verifier should answer one focused question at a time. Atomic claims reduce ambiguity and make it possible to identify exactly which statement failed.

Example:

```text
"Teddy bears were created in the early 1900s and named after Theodore Roosevelt."

claim 1: Teddy bears were created in the early 1900s.
claim 2: Teddy bears were named after Theodore Roosevelt.
```

**Memory liner:**

> Atomic claims turn vague response-level factuality into inspectable units.

---

## Q66. SHOULD KNOW - What makes a claim suitably atomic?

A useful claim should be independently verifiable and should not silently combine multiple facts connected by words such as "and," "because," or "after."

**Memory liner:**

> One claim should require one evidence-backed verdict.

---

## Q67. MUST REMEMBER - How is each fact checked?

The lecture proposes retrieving evidence through mechanisms such as:

- RAG over a trusted knowledge base;
- web search;
- additional LLM calls grounded in retrieved evidence.

Then classify the claim as supported or unsupported.

**Memory liner:**

> Factuality judging needs evidence, not only another model's memory.

---

## Q68. MUST REMEMBER - What is the per-fact representation?

For fact `i`:

$$
z_i=
\begin{cases}
1,&\text{fact i is supported}\\
0,&\text{fact i is unsupported}
\end{cases}
$$

The lecture favors a simple binary decision at this stage even though real evidence can sometimes be uncertain.

**Memory liner:**

> Make individual verification simple, then recover nuance through aggregation.

---

## Q69. SHOULD KNOW - Why assign importance weights to facts?

Not every error has equal impact. A central answer claim may matter more than a peripheral historical detail.

Let `alpha_i >= 0` represent the importance of fact `i`.

**Memory liner:**

> Weight errors by how much they affect the answer's core meaning.

---

## Q70. MUST REMEMBER - What weighted factuality score follows from the lecture?

$$
\boxed{
Factuality(y)
=
\frac{\sum_{i=1}^{F}\alpha_i z_i}
{\sum_{i=1}^{F}\alpha_i}
}
$$

If all weights are equal:

$$
Factuality(y)
=
\frac{1}{F}\sum_{i=1}^{F}z_i
$$

**Memory liner:**

> Response factuality is the weighted fraction of supported claims.

---

## Q71. MUST REMEMBER - What can fail inside a factuality evaluator?

The final score inherits errors from several stages:

```text
claim extractor misses or merges facts
retriever fails to find evidence
evidence source is wrong or stale
verifier misreads evidence
importance weights are poorly chosen
```

**Memory liner:**

> A factuality score is only as reliable as its decomposition, evidence, and verification pipeline.

---

## Q72. SHOULD KNOW - What is the implementation mental map for factuality evaluation?

```python
from dataclasses import dataclass
from typing import Sequence


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool
    importance: float
    evidence: Sequence[str]


def factuality_score(verdicts: Sequence[ClaimVerdict]) -> float:
    total_weight = sum(v.importance for v in verdicts)
    if total_weight <= 0:
        raise ValueError("At least one positive importance weight is required")

    supported_weight = sum(
        v.importance for v in verdicts if v.supported
    )
    return supported_weight / total_weight
```

This code implements only the final aggregation. Claim extraction and evidence-grounded verification are separate stages.

**Memory liner:**

> Keep extraction, retrieval, verification, and aggregation modular so each source of error can be debugged.


# 7. Evaluating tool calls and agentic workflows

## Q73. MUST REMEMBER - Why is final-answer evaluation insufficient for agents?

An agent can fail before the final response at any of these stages:

```text
understand request
    -> select tool
    -> predict arguments
    -> execute tool
    -> interpret tool result
    -> decide whether to continue
    -> synthesize final response
```

A bad final answer does not reveal which stage caused the problem.

**Memory liner:**

> Evaluate agent traces stage by stage, not only the text shown at the end.

---

## Q74. MUST REMEMBER - What three-stage tool-call decomposition does the lecture use?

```text
1. TOOL PREDICTION
   choose a tool and its arguments

2. TOOL EXECUTION
   run backend code and obtain a structured result

3. RESULT SYNTHESIS
   ground on the result and answer the user
```

A multi-step agent may repeat this structure several times.

**Memory liner:**

> Predict, execute, synthesize - and repeat when the task requires more steps.

---

## Q75. SHOULD KNOW - What information should an agent evaluator log?

**INTERVIEW CLARIFICATION:** To support the failure analysis taught in the lecture, record:

```text
user prompt
tools exposed to the model
router shortlist
selected tool name
predicted arguments
backend status and result
intermediate observations
final response
expected tool / state change, when available
```

**Memory liner:**

> If a stage is not logged, it is difficult to distinguish reasoning failure from infrastructure failure.

---

## Q76. MUST REMEMBER - What is a punt?

A punt is a non-answer such as:

```text
"Sorry, I cannot do that."
"I do not know."
```

In the lecture's tool example, the model punts even though a suitable tool exists.

**Memory liner:**

> A punt abandons the task instead of using an available path to complete it.

---

## Q77. MUST REMEMBER - How can a tool-router recall error cause a punt?

A router first reduces a large tool catalog to a small shortlist. If the needed tool is missing from that shortlist, the main model cannot call it.

```text
all tools
   -> router
   -> selected tools
          X needed tool absent
   -> task fails
```

**Memory liner:**

> Tool routing should be recall-oriented: include the needed tool even if a few extras survive.

---

## Q78. MUST REMEMBER - What if the correct tool is present but the LLM does not use it?

Then the failure is not router recall. It is tool-use behavior in the main model or prompt.

Possible remedies from the lecture:

- add SFT examples showing that pattern;
- improve tool-use instructions;
- improve the tool description;
- use a more capable model if grounding is broadly weak.

**Memory liner:**

> First ask whether the model could see the tool; only then ask whether it knew to use it.

---

## Q79. MUST REMEMBER - What is tool hallucination?

The model emits a function name that is not defined, for example calling `find_bear` when only `find_teddy_bear` exists.

Potential causes:

- weak grounding;
- unclear global tool-use instructions;
- poorly named functions or arguments;
- insufficient model capability.

**Memory liner:**

> Tool hallucination is inventing an API instead of selecting from the available schema.

---

## Q80. MUST REMEMBER - What is wrong-tool selection?

The model chooses a real but inappropriate tool because several tools have overlapping or ambiguous scopes.

The lecture recommends making tool descriptions explicit about when each tool should and should not be used.

**Memory liner:**

> Distinct tools need distinct semantic contracts.

---

## Q81. MUST REMEMBER - What is the right-tool, wrong-arguments failure?

The tool name is correct, but one or more arguments are missing, fabricated, malformed, or semantically wrong.

Example:

```text
user: find a bear near me
model: find_teddy_bear(latitude=0, longitude=0)
```

The model invented coordinates because the actual location was unavailable.

**Memory liner:**

> Selecting the capability is only half the job; arguments must be grounded in available context.

---

## Q82. MUST REMEMBER - How should missing required information be handled?

Do not silently fabricate a default. Instead:

```text
check context or permissions
    -> call another information-gathering tool
    -> or ask the user a clarifying question
    -> or return an actionable structured error
```

**Memory liner:**

> Missing context should trigger information acquisition, not argument hallucination.

---

## Q83. MUST REMEMBER - What is a backend tool error?

The model predicts the correct tool and arguments, but the implementation returns the wrong value or crashes because of ordinary software bugs.

The remedy is software engineering, not necessarily model retraining.

**Memory liner:**

> A correct tool call can still fail because the tool itself is broken.

---

## Q84. MUST REMEMBER - Why is returning no tool response dangerous?

For action tools, silence prevents the model from knowing whether the state actually changed. It may falsely tell the user that the action succeeded.

```text
action requested
    -> tool executes
    -> no status returned
    -> model invents success / failure
```

**Memory liner:**

> Every action tool should return an explicit, machine-readable outcome.

---

## Q85. SHOULD KNOW - Why can an empty structured result be better than `None`?

An empty object or list can explicitly mean "the search succeeded but found zero items." A null or missing response may instead mean timeout, crash, absent data, or no result.

```json
{
  "status": "success",
  "items": []
}
```

**Memory liner:**

> Empty is a valid result; silence is ambiguous.

---

## Q86. MUST REMEMBER - What is a synthesis or grounding failure?

The tool result is correct, but the model's final response contradicts, ignores, or misreads it.

Example:

```text
tool result: one bear found
final answer: no bears were found
```

**Memory liner:**

> Correct computation does not help if the model fails to ground its final text in the returned observation.

---

## Q87. MUST REMEMBER - How can excessive tool output hurt synthesis?

A backend may return a large amount of irrelevant data, causing the important field to be lost in noise. The lecture recommends trimming outputs to only what the next model step needs.

**Memory liner:**

> Tool outputs should be informative, structured, and minimal enough to ground on reliably.

---

## Q88. SHOULD KNOW - Why are named structured fields helpful?

Compare:

```text
raw tuple: ["Teddy", 1.0, 37.4, -122.1]
```

with:

```json
{
  "name": "Teddy",
  "distance_miles": 1.0,
  "latitude": 37.4,
  "longitude": -122.1
}
```

Named fields reduce ambiguity during synthesis.

**Memory liner:**

> Structure converts backend bytes into semantic observations the model can use.

---

## Q89. MUST REMEMBER - What root-cause buckets summarize the agent failures?

The lecture groups remedies around:

1. **Model capability and grounding**;
2. **Context relevance and router recall**;
3. **Tool-use prompting or SFT**;
4. **API names, arguments, and descriptions**;
5. **Backend implementation quality**;
6. **Tool-output structure and concision**.

**Memory liner:**

> Diagnose whether the failure belongs to the model, context, schema, or software before changing anything.

---

## Q90. SHOULD KNOW - What stage-level metrics follow naturally from the lecture?

**INTERVIEW CLARIFICATION:**

| Stage | Possible metric |
|---|---|
| Router | recall of required tool in shortlist |
| Tool selection | exact tool-name accuracy |
| Arguments | schema validity and field-level accuracy |
| Execution | success rate / expected state change |
| Synthesis | groundedness to tool result |
| Full task | end-to-end task success |

**Memory liner:**

> Stage metrics explain end-to-end success instead of merely reporting it.

---

## Q91. MUST REMEMBER - Why build an error taxonomy rather than inspect failures one by one?

Agent systems can fail in many repeated patterns. Grouping failures lets the team measure prevalence and apply one targeted fix to an entire category.

```text
collect traces
   -> label failure type
   -> count by category
   -> prioritize largest / highest-risk category
   -> apply focused remedy
```

**Memory liner:**

> A taxonomy converts agent debugging from anecdotes into an engineering process.

---

## Q92. SHOULD KNOW - What is the end-to-end agent evaluation mental map?

```python
def evaluate_trace(trace, expected):
    return {
        "router_has_required_tool": expected.tool in trace.exposed_tools,
        "selected_correct_tool": trace.tool_name == expected.tool,
        "arguments_valid": validate_args(trace.arguments, expected.schema),
        "backend_succeeded": trace.tool_status == "success",
        "state_is_correct": check_state(trace.final_state, expected.final_state),
        "answer_grounded": judge_grounding(trace.tool_result, trace.final_answer),
    }
```

The lecture does not prescribe this exact API; it motivates the decomposition.

**Memory liner:**

> End-to-end evaluation should preserve the causal chain from decision to action to answer.

---

# 8. Benchmark design and benchmark families

## Q93. MUST REMEMBER - What is the purpose of a benchmark suite?

A suite characterizes multiple capabilities so models can be compared on a standardized set of tasks.

**Memory liner:**

> Benchmarks produce a capability profile, not a universal truth about the model.

---

## Q94. MUST REMEMBER - Why do many benchmarks use constrained outputs?

Multiple-choice letters, fixed numeric answers, and executable tests enable deterministic scoring without adding another uncertain LLM judge.

**Memory liner:**

> Constrain the answer when possible so correctness can be checked directly.

---

## Q95. MUST REMEMBER - What is MMLU?

MMLU stands for **Massive Multitask Language Understanding**. In the lecture it contains roughly sixty diverse subject areas and uses four-option multiple-choice questions.

```text
question + choices A/B/C/D
    -> model emits one option
    -> exact comparison with answer key
```

**Memory liner:**

> MMLU measures broad knowledge through standardized multiple-choice tasks.

---

## Q96. SHOULD KNOW - What part of model training does MMLU mostly probe according to the lecture?

It largely probes whether broad information from pre-training can be recalled and applied at inference time, though reasoning may also be needed for some questions.

**Memory liner:**

> Knowledge benchmarks test what the model retained from broad training corpora.

---

## Q97. MUST REMEMBER - Why is AIME convenient for reasoning evaluation?

AIME contains difficult competition-math problems but requires a tightly constrained three-digit answer. The model may reason freely, while the evaluator parses and checks one numeric result.

**Memory liner:**

> AIME combines difficult reasoning with an objective final-answer verifier.

---

## Q98. MUST REMEMBER - What is PIQA?

PIQA is **Physical Interaction: Question Answering**, a two-choice benchmark for physical commonsense and everyday interactions.

The transcript's carpet example asks which vacuum setup can retrieve a lost object without swallowing it irretrievably.

**Memory liner:**

> PIQA tests practical physical reasoning rather than memorized academic facts.

---

## Q99. MUST REMEMBER - What is SWE-bench?

SWE-bench constructs software-engineering tasks from real repository issues and associated code changes/tests.

```text
repository + issue
    -> model generates patch
    -> apply patch
    -> run tests
    -> score success
```

**Memory liner:**

> SWE-bench evaluates whether a model can repair real codebases, not merely write isolated snippets.

---

## Q100. MUST REMEMBER - Why are tests powerful coding verifiers?

They convert an open-ended patch into observable behavior. A patch is useful only if the relevant tests pass without unacceptable regressions.

**Memory liner:**

> Executable tests turn code generation into a verifiable task.

---

## Q101. MUST REMEMBER - Why are safety benchmarks hard to compare across providers?

Safety policies are partly normative. Different providers may intentionally draw different boundaries, so a lower refusal or attack-success rate is meaningful only relative to the target policy.

**Memory liner:**

> Safety evaluation must be interpreted against the policy the system is supposed to follow.

---

## Q102. MUST REMEMBER - What is HarmBench at the lecture level?

HarmBench evaluates harmful behavior across categories including standard harmful requests, copyright, contextual text cases, and multimodal cases.

Unlike multiple-choice or exact-answer tasks, open-ended harmful outputs often require a learned classifier.

**Memory liner:**

> HarmBench uses a classifier because unsafe behavior cannot be captured reliably by exact string matching.

---

## Q103. SHOULD KNOW - What caveat comes with classifier-scored safety benchmarks?

The benchmark score inherits classifier errors. A response may be misclassified as safe or unsafe, so the evaluator itself must be validated.

The lecture also distinguishes intent/attempt from capability: an unsafe attempt can count as a safety failure even if the model executes it poorly.

**Memory liner:**

> Safety quality and task-execution quality are different axes.

---

## Q104. MUST REMEMBER - What is tau-bench?

Tau-bench evaluates tool-using agents in domains such as airline and retail. It provides:

- domain tools;
- policies governing allowed behavior;
- user tasks;
- multi-turn interaction;
- a final reward based on database state or required actions.

**Memory liner:**

> Tau-bench tests whether a conversational tool agent can complete policy-constrained real-world tasks.

---

## Q105. MUST REMEMBER - Why does tau-bench use a simulated user model?

The next user message depends on what the agent just did. A fixed script cannot cover all possible agent trajectories, so another model dynamically plays the user while following a task specification.

**Memory liner:**

> Interactive agents require an evaluator that can continue the conversation conditionally.

---

## Q106. MUST REMEMBER - Why is database state a strong agent verifier?

The final conversational text may claim success falsely. The database provides an external ground truth about whether the flight, order, refund, or other state actually changed as required.

**Memory liner:**

> For action agents, verify the world state, not merely the assistant's confirmation.

---

## Q107. MUST REMEMBER - What is `pass^k` in the lecture?

`pass^k` measures reliability: the probability that **all** of `k` attempts succeed.

If a single-run success probability is `p` and runs are independent:

$$
pass^k=p^k
$$

**INTERVIEW CLARIFICATION:** Given `n` observed attempts with `c` successes, the sampling-without-replacement estimator analogous to `pass@k` is:

$$
\boxed{
\widehat{pass^k}
=
\frac{\binom{c}{k}}
{\binom{n}{k}}
}
$$

**Memory liner:**

> `pass^k` asks whether the system succeeds every time, not whether it gets lucky once.

---

## Q108. MUST REMEMBER - `pass@k` vs `pass^k`?

### `pass@k`

Probability that **at least one** of `k` attempts succeeds:

$$
\widehat{pass@k}
=
1-
\frac{\binom{n-c}{k}}
{\binom{n}{k}}
$$

### `pass^k`

Probability that **all** `k` attempts succeed. **INTERVIEW CLARIFICATION:** using the sampling-without-replacement estimator analogous to the prior-lecture `pass@k` derivation:

$$
\widehat{pass^k}
=
\frac{\binom{c}{k}}
{\binom{n}{k}}
$$

**Memory liner:**

> `pass@k` measures possibility with repeated tries; `pass^k` measures consistency across repeated tries.

---

# 9. Interpreting benchmarks correctly

## Q109. MUST REMEMBER - Why should benchmarks be reported as a suite?

A model can improve on one capability while regressing on another. Reporting only the best number hides trade-offs.

**Memory liner:**

> A benchmark suite is a capability profile, not a beauty contest with one score.

---

## Q110. SHOULD KNOW - What does it mean to profile a model?

Plot or tabulate its behavior across dimensions such as:

```text
knowledge
reasoning
coding
safety
tool use
cost
latency
context length
```

Then choose the model whose profile matches the application.

**Memory liner:**

> The best model is conditional on the workload and constraints.

---

## Q111. MUST REMEMBER - Why combine quality with price or latency?

A small quality gain can be unjustified if it requires much higher serving cost or response time. Production choice is therefore multi-objective.

**Memory liner:**

> Accuracy without cost is a research number; accuracy under constraints is a deployment decision.

---

## Q112. MUST REMEMBER - What is a Pareto frontier?

A model lies on the Pareto frontier if no alternative is better on one target dimension without being worse on another.

For quality and cost:

```text
model is Pareto-dominated
    if another model is at least as good in quality
    and no more expensive,
    with one strict improvement
```

**Memory liner:**

> The Pareto frontier contains the non-dominated trade-off choices.

---

## Q113. MUST REMEMBER - What is benchmark contamination?

Contamination occurs when benchmark questions, answers, or near-duplicates appear in training data or are otherwise accessible during evaluation.

The score may then reflect memorization or answer lookup rather than general capability.

**Memory liner:**

> A test is meaningful only if the model did not train on its answers.

---

## Q114. SHOULD KNOW - What contamination defenses does the lecture mention?

- use hash values to detect known benchmark items;
- block access to websites containing answers during tool evaluation;
- evaluate on newly released exams or tasks that predate neither training nor testing;
- inspect the training/evaluation overlap where possible.

**Memory liner:**

> Protect the boundary between learning data and evaluation data.

---

## Q115. MUST REMEMBER - What is Goodhart's law and why does it matter here?

The lecture states the idea:

> When a measure becomes a target, it ceases to be a good measure.

Once teams optimize directly for a benchmark, systems may exploit benchmark-specific patterns rather than improve the real capability.

**Memory liner:**

> Optimizing a metric changes the system and can destroy the metric's validity as a proxy.

---

## Q116. MUST REMEMBER - Why should every application maintain custom evaluations?

Public benchmarks may not represent:

- the application's prompt distribution;
- domain terminology;
- policy requirements;
- tool catalog;
- user tolerance for latency, verbosity, or refusals.

**Memory liner:**

> Public benchmarks compare models broadly; custom evals decide whether a model works for your product.

---

## Q117. SHOULD KNOW - Where do arenas or broad user preferences fit?

User preference systems can complement hard-coded benchmarks by measuring real-world response preference, but they also inherit user-population bias, subjective taste, and weak factual verification.

**Memory liner:**

> Preference data captures lived experience but does not automatically establish truth or safety.

---

## Q118. MUST REMEMBER - How should hard verifiers, LLM judges, and humans be combined?

Use the strongest available source of supervision:

```text
exact answer / executable test / database state
    -> use hard verifier

open-ended semantic criterion at scale
    -> use calibrated LLM judge

high-stakes or ambiguous slice
    -> use expert human review
```

**Memory liner:**

> Prefer deterministic evidence when it exists; use learned judgment where the task is genuinely open-ended.

---

## Q119. SHOULD KNOW - What makes an evaluation reproducible?

**INTERVIEW CLARIFICATION consistent with the lecture's low-temperature guidance:** record:

- model and judge versions;
- prompts and rubrics;
- temperature and sampling settings;
- tool versions and knowledge-base snapshot;
- parser/schema;
- benchmark version;
- random seeds when supported.

**Memory liner:**

> An evaluation result without its configuration is difficult to reproduce or interpret.

---

## Q120. MUST REMEMBER - What is the complete evaluation recipe from this lecture?

```text
1. Define the real use case and quality dimensions.
2. Build a representative prompt set.
3. Use direct verifiers wherever possible.
4. Use human labels to define and calibrate subjective criteria.
5. Use a structured LLM judge for scalable open-ended evaluation.
6. Test judge biases and order sensitivity.
7. Decompose factuality and agent traces into checkable stages.
8. Report a benchmark profile plus cost / latency constraints.
9. monitor contamination and proxy overoptimization.
10. Inspect failures by category and iterate on the right component.
```

**Master memory liner:**

> Define the target, choose the strongest verifier, calibrate proxies, decompose failures, and report trade-offs rather than one flattering number.


---

# 10. Equations to memorize cold

## 10.1 Observed agreement

$$
\boxed{
P_o
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[r_{A,i}=r_{B,i}]
}
$$

```text
number of matching ratings / total examples
```

---

## 10.2 Binary chance agreement

$$
\boxed{
P_e
=
p_Ap_B+(1-p_A)(1-p_B)
}
$$

```text
both say positive + both say negative
```

---

## 10.3 Cohen's kappa

$$
\boxed{
\kappa
=
\frac{P_o-P_e}{1-P_e}
}
$$

```text
observed agreement - chance agreement
-------------------------------------
maximum possible - chance agreement
```

---

## 10.4 METEOR overlap precision and recall

$$
\boxed{
P=\frac{m}{|y|},
\qquad
R=\frac{m}{|y^*|}
}
$$

where `m` is the number of matched unigrams.

---

## 10.5 METEOR weighted harmonic term

**INTERVIEW CLARIFICATION:**

$$
\boxed{
F_{mean}
=
\frac{PR}{\alpha P+(1-\alpha)R}
}
$$

---

## 10.6 METEOR fragmentation penalty

**INTERVIEW CLARIFICATION:**

$$
\boxed{
Penalty
=
\gamma
\left(\frac{ch}{m}\right)^{\beta}
}
$$

---

## 10.7 METEOR score

$$
\boxed{
METEOR
=
(1-Penalty)F_{mean}
}
$$

---

## 10.8 BLEU brevity penalty

**INTERVIEW CLARIFICATION:**

$$
\boxed{
BP=
\begin{cases}
1,&c>r\\
\exp(1-r/c),&c\le r
\end{cases}
}
$$

---

## 10.9 BLEU

**INTERVIEW CLARIFICATION:**

$$
\boxed{
BLEU
=
BP\exp\left(
\sum_{n=1}^{N}w_n\log p_n
\right)
}
$$

---

## 10.10 Weighted factuality

$$
\boxed{
Factuality(y)
=
\frac{\sum_i\alpha_i z_i}
{\sum_i\alpha_i}
}
$$

---

## 10.11 Pass at k

**LECTURE RECALL:** Lecture 8 references the `pass@k` result derived in the preceding reasoning lecture.

$$
\boxed{
\widehat{pass@k}
=
1-
\frac{\binom{n-c}{k}}
{\binom{n}{k}}
}
$$

```text
1 - probability all k selected attempts are failures
```

---

## 10.12 Pass to the power k

**INTERVIEW CLARIFICATION:** This is the all-success counterpart obtained by the same sampling-without-replacement logic.

$$
\boxed{
\widehat{pass^k}
=
\frac{\binom{c}{k}}
{\binom{n}{k}}
}
$$

```text
probability all k selected attempts are successes
```

Under independent runs with success probability `p`:

$$
\boxed{pass^k=p^k}
$$

---

# 11. Evaluation data and tensor shapes

The lecture is primarily about systems and metrics rather than neural tensor algebra. These shapes are implementation-oriented representations of the evaluation objects it describes.

## 11.1 Human-rating matrix

```text
ratings: [N, R]

N = evaluated prompt-response pairs
R = number of raters
```

For binary labels:

```text
ratings[i, r] in {0, 1}
```

For two raters:

```text
rater_A: [N]
rater_B: [N]
```

---

## 11.2 Pointwise judge batch

```text
prompt token IDs:       [B, T_q]
response token IDs:     [B, T_y]
rubric token IDs:       [B, T_c]
combined judge input:   [B, T_total]
structured score:       [B]
rationale token IDs:    [B, T_r]
```

Conceptual function:

$$
J(q,y,c)\rightarrow(r,s)
$$

---

## 11.3 Pairwise judge batch

```text
prompts:                [B, T_q]
response A:             [B, T_A]
response B:             [B, T_B]
pairwise label:         [B]
```

Possible encoded labels:

```text
0 = A wins
1 = B wins
2 = tie / abstain
```

For order swapping, construct two evaluations per pair:

```text
ordered pairs: [B, 2, ...]
```

where index 0 is `(A,B)` and index 1 is `(B,A)`.

---

## 11.4 Claim-level factuality batch

After extracting up to `F_max` claims per response:

```text
claim mask:             [B, F_max]
correctness z:          [B, F_max]
importance alpha:       [B, F_max]
weighted score:         [B]
```

Aggregation:

```python
score = (alpha * z * mask).sum(-1) / (alpha * mask).sum(-1)
```

---

## 11.5 Multiple-choice benchmark batch

```text
prompt IDs:             [B, T]
number of choices:      C
choice scores/logits:   [B, C]
predicted choice:       [B]
ground-truth choice:    [B]
correct indicator:      [B]
```

Accuracy:

$$
Accuracy
=
\frac{1}{B}
\sum_{i=1}^{B}
\mathbf{1}[\hat y_i=y_i]
$$

---

## 11.6 Agent-trace representation

For at most `S` tool steps:

```text
tool IDs:               [B, S]
tool-step mask:         [B, S]
execution success:      [B, S]
argument-valid flags:   [B, S]
step rewards:           [B, S]
end-to-end success:     [B]
```

The actual argument objects and tool outputs are usually variable-size structured records rather than dense tensors.

---

# 12. Implementation mental maps

## 12.1 Cohen's kappa from binary labels

```python
from __future__ import annotations

from collections.abc import Sequence


def cohens_kappa_binary(
    rater_a: Sequence[int],
    rater_b: Sequence[int],
) -> float:
    if len(rater_a) != len(rater_b):
        raise ValueError("Rater vectors must have the same length")
    if not rater_a:
        raise ValueError("At least one rating is required")
    if any(x not in (0, 1) for x in (*rater_a, *rater_b)):
        raise ValueError("This helper expects binary labels")

    n = len(rater_a)
    p_observed = sum(a == b for a, b in zip(rater_a, rater_b)) / n

    p_a = sum(rater_a) / n
    p_b = sum(rater_b) / n
    p_expected = p_a * p_b + (1.0 - p_a) * (1.0 - p_b)

    denominator = 1.0 - p_expected
    if denominator == 0.0:
        # Both raters have degenerate marginals. Kappa is undefined.
        raise ValueError("Kappa is undefined when chance agreement is 1")

    return (p_observed - p_expected) / denominator
```

**Mental map:**

```text
measure actual agreement
    -
measure agreement caused by label frequencies
    ->
normalize by remaining possible agreement
```

---

## 12.2 Structured pointwise judge schema

```python
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class JudgeResult:
    rationale: str
    verdict: Literal["pass", "fail"]
```

Conceptual judge prompt:

```text
You are evaluating one response.

Criterion:
The response must be factually supported, directly relevant, and concise.
Do not reward verbosity by itself.

User prompt:
{prompt}

Candidate response:
{response}

First provide a concise evidence-based rationale.
Then return a verdict of exactly "pass" or "fail".
```

Use a structured-output mechanism so the returned fields are guaranteed to be parseable.

---

## 12.3 Position-robust pairwise judging

```python
from dataclasses import dataclass
from typing import Literal


Preference = Literal["A", "B", "tie"]


@dataclass(frozen=True)
class OrderedJudgment:
    winner: Preference
    rationale: str


def reconcile_swapped_judgments(
    ab: OrderedJudgment,
    ba: OrderedJudgment,
) -> Preference:
    """
    In the BA run, label A refers to the original response B,
    and label B refers to the original response A.
    """
    ba_in_original_order: Preference
    if ba.winner == "A":
        ba_in_original_order = "B"
    elif ba.winner == "B":
        ba_in_original_order = "A"
    else:
        ba_in_original_order = "tie"

    if ab.winner == ba_in_original_order:
        return ab.winner
    return "tie"
```

**Memory liner:**

> Reverse presentation order and convert the second verdict back to the original candidate identities.

---

## 12.4 Batched factuality aggregation in PyTorch

```python
import torch


def weighted_factuality(
    supported: torch.Tensor,   # [B, F], bool or 0/1
    importance: torch.Tensor,  # [B, F], non-negative
    mask: torch.Tensor,        # [B, F], bool
) -> torch.Tensor:
    if supported.shape != importance.shape or supported.shape != mask.shape:
        raise ValueError("supported, importance, and mask must share shape [B, F]")

    supported_f = supported.to(dtype=importance.dtype)
    mask_f = mask.to(dtype=importance.dtype)

    weights = importance * mask_f
    numerator = (weights * supported_f).sum(dim=-1)       # [B]
    denominator = weights.sum(dim=-1).clamp_min(1e-12)    # [B]
    return numerator / denominator                         # [B]
```

---

## 12.5 Agent trace as a typed record

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolStep:
    exposed_tools: list[str]
    selected_tool: str | None
    arguments: dict[str, Any] | None
    tool_status: str | None
    tool_result: Any | None


@dataclass
class AgentTrace:
    user_prompt: str
    steps: list[ToolStep] = field(default_factory=list)
    final_answer: str | None = None
    final_state: dict[str, Any] | None = None
```

Recommended evaluation outputs:

```python
@dataclass
class TraceEvaluation:
    router_recall_ok: bool
    tool_selection_ok: bool
    arguments_ok: bool
    backend_ok: bool
    state_change_ok: bool
    final_answer_grounded: bool
    end_to_end_success: bool
```

---

## 12.6 Pass metrics

```python
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n:
        raise ValueError("Require 0 <= c <= n")
    if not 1 <= k <= n:
        raise ValueError("Require 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_power_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n:
        raise ValueError("Require 0 <= c <= n")
    if not 1 <= k <= n:
        raise ValueError("Require 1 <= k <= n")
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)
```

---

## 12.7 Simple Pareto-front computation

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPoint:
    name: str
    quality: float   # higher is better
    cost: float      # lower is better


def pareto_front(points: list[ModelPoint]) -> list[ModelPoint]:
    result: list[ModelPoint] = []
    for candidate in points:
        dominated = any(
            other.name != candidate.name
            and other.quality >= candidate.quality
            and other.cost <= candidate.cost
            and (
                other.quality > candidate.quality
                or other.cost < candidate.cost
            )
            for other in points
        )
        if not dominated:
            result.append(candidate)
    return sorted(result, key=lambda p: p.cost)
```

**Memory liner:**

> Remove every model for which another option is both no worse and strictly better somewhere.

---

# 13. Comparison tables

## 13.1 Choosing an evaluator

| Situation | Preferred evaluator | Why |
|---|---|---|
| Exact numeric answer | Exact match / verifier | Direct and deterministic |
| Code patch | Compile and run tests | Measures executable correctness |
| Tool action | Check backend/database state | Verifies real-world effect |
| Translation regression | BLEU/METEOR plus human slices | Fast reference signal |
| Open-ended helpfulness | Calibrated LLM judge | Semantic and scalable |
| High-stakes ambiguous output | Expert humans | Requires domain judgment |
| Factual paragraph | Claim decomposition + evidence | Localizes supported and unsupported claims |

---

## 13.2 Judge-bias matrix

| Bias | Symptom | Diagnostic | Mitigation |
|---|---|---|---|
| Position | Winner changes when A/B order swaps | Evaluate both orders | Swap, aggregate, or abstain |
| Verbosity | Longer answer wins without more correctness | Compare equal-quality short/long controls | Explicit rubric, examples, length awareness |
| Self-enhancement | Judge favors its own model family | Compare against independent human labels | Different judge, stronger judge, calibration |
| Human-policy mismatch | Stable judge score disagrees with target users | Slice by human subgroup / criterion | Rewrite rubric and recalibrate |
| Parse failure | Missing or malformed score | Schema validation | Structured output / guided decoding |

---

## 13.3 Benchmark-family matrix

| Family | Example | Output verifier | Main capability |
|---|---|---|---|
| Knowledge | MMLU | Multiple-choice answer key | Broad retained knowledge |
| Math reasoning | AIME | Exact numeric answer | Multi-step mathematical reasoning |
| Physical commonsense | PIQA | Two-choice answer key | Everyday physical reasoning |
| Coding | SWE-bench | Repository tests | Software-engineering task completion |
| Safety | HarmBench | Learned safety classifier | Harmful-behavior compliance/resistance |
| Agent/tool use | tau-bench | State change / action reward | Reliable policy-constrained tool use |

---

## 13.4 `pass@k` vs `pass^k`

| Metric | Event | Product interpretation |
|---|---|---|
| `pass@k` | At least one of k runs succeeds | Can I get one valid answer with repeated attempts? |
| `pass^k` | All k runs succeed | Can I trust the system repeatedly? |

```text
coding exploration        -> pass@k can be useful
customer-facing automation -> pass^k exposes reliability
```

---

## 13.5 Agent-stage failure matrix

| Stage | Failure | Likely owner |
|---|---|---|
| Router | Required tool absent | Router/retriever design |
| Main model | Tool visible but not selected | Prompt/SFT/model capability |
| Tool name | Invented tool | Grounding/schema/model capability |
| Tool choice | Wrong real tool | Overlapping API descriptions |
| Arguments | Missing or hallucinated values | Context/permissions/schema |
| Backend | Error or wrong result | Software implementation |
| Backend response | No status / ambiguous null | Tool contract |
| Synthesis | Final answer ignores result | Grounding/context/model capability |

---

# 14. Debugging questions for failed evaluations

## Human evaluation

1. Are raters applying the same criterion?
2. Is the guideline ambiguous near the decision boundary?
3. Is one label overwhelmingly common?
4. Is raw agreement high only because of label imbalance?
5. Does disagreement concentrate on one prompt category?
6. Do raters need a calibration session?

## Reference metrics

7. Is the reference set too narrow for valid paraphrases?
8. Is the metric rewarding lexical overlap rather than meaning?
9. Is a short candidate exploiting precision?
10. Are score changes correlated with human judgment?
11. Did tokenization or preprocessing change?

## LLM-as-a-judge

12. Does the rubric define one criterion or several mixed criteria?
13. Is the output schema guaranteed?
14. Does swapping candidate order change the winner?
15. Does the judge prefer longer responses?
16. Is the judge evaluating outputs from itself?
17. Does the judge agree with humans on a calibration set?
18. Is the temperature low enough for repeatability?
19. Did the judge model or prompt version change?
20. Are you optimizing the generator against this same proxy too aggressively?

## Factuality

21. Did claim extraction miss important assertions?
22. Were compound claims split?
23. Did retrieval find authoritative evidence?
24. Is the evidence current for the claim?
25. Is the verifier distinguishing unsupported from contradicted?
26. Are importance weights masking serious errors?

## Agents

27. Was the necessary tool included by the router?
28. Was the tool included but ignored?
29. Did the model invent a tool?
30. Were API descriptions mutually exclusive enough?
31. Were required arguments present in context?
32. Did permissions prevent a grounded argument?
33. Did the backend actually execute successfully?
34. Did it return a meaningful structured result?
35. Did the agent verify the external state?
36. Did the final answer faithfully summarize the tool result?

## Benchmarks

37. Could the test be present in training data?
38. Is answer extraction correct?
39. Is the benchmark measuring the desired use case?
40. Is the score dominated by one easy category?
41. Is a classifier-based evaluator introducing its own error?
42. Is the model improvement worth the added cost or latency?

---

# 15. Oral interview questions - practise answering aloud

## Core evaluation

1. What does it mean to evaluate an LLM?
2. Why is free-form generation harder to evaluate than classification?
3. Why should output-quality dimensions be separated?
4. Compare human evaluation, fixed-reference metrics, and LLM-as-a-judge.
5. Why is no evaluator a perfect oracle?
6. What is the difference between an evaluation and a benchmark?
7. Why is one aggregate score often misleading?

## Human ratings and agreement

8. Why is human evaluation both valuable and difficult?
9. What is observed agreement?
10. Why can observed agreement be high by chance?
11. Derive binary chance agreement for two raters.
12. Derive Cohen's kappa.
13. What does negative kappa mean?
14. Cohen's kappa vs Fleiss' kappa?
15. What role does Krippendorff's alpha play?
16. How would you react to low inter-rater agreement?
17. Why are examples in annotation guidelines useful?
18. Pointwise ratings vs pairwise preferences?

## Reference metrics

19. What problem do fixed-reference metrics solve?
20. Explain METEOR intuitively.
21. Define METEOR precision and recall.
22. What is the fragmentation penalty?
23. Explain BLEU intuitively.
24. Why does BLEU need a brevity penalty?
25. What is ROUGE used for at a high level?
26. Why can a semantically correct paraphrase receive a poor overlap score?
27. When would you still use BLEU or ROUGE?

## LLM-as-a-judge

28. What is LLM-as-a-judge?
29. What inputs must the judge receive?
30. Why ask for a rationale before the verdict?
31. Why use structured output?
32. Pointwise vs pairwise judging?
33. How can a judge generate preference data?
34. What is position bias?
35. How do you test and mitigate position bias?
36. What is verbosity bias?
37. What is self-enhancement bias?
38. Should the judge be larger than the generator?
39. Why prefer a binary scale?
40. Why use low temperature?
41. How do you calibrate a judge against humans?
42. What is proxy overoptimization?
43. Can a good rationale accompany a bad score?

## Factuality

44. Why not evaluate factuality only at the response level?
45. What is an atomic claim?
46. Describe the claim-level factuality pipeline.
47. Why does the verifier need retrieved evidence?
48. Derive weighted factuality aggregation.
49. What failure modes exist inside the factuality evaluator itself?

## Agent evaluation

50. Why is final-answer accuracy insufficient for agents?
51. Describe the predict-execute-synthesize decomposition.
52. What is a punt?
53. What is a tool-router recall failure?
54. Tool present but unused: what would you change?
55. What is tool hallucination?
56. Why do overlapping tool descriptions cause errors?
57. What is a right-tool, wrong-arguments failure?
58. How should missing location or permission be handled?
59. Why must action tools return explicit status?
60. Why is an empty structured list better than no response?
61. What causes result-synthesis failures?
62. How would you build an agent error taxonomy?
63. What metrics would you log at every tool stage?

## Benchmarks

64. What does MMLU measure and how is it scored?
65. Why is AIME convenient for automated reasoning evaluation?
66. What does PIQA measure?
67. How does SWE-bench verify a generated patch?
68. Why are safety benchmarks policy-dependent?
69. Why can HarmBench's classifier be wrong?
70. What does tau-bench evaluate?
71. Why use a simulated user in an agent benchmark?
72. Why verify database state instead of trusting final text?
73. Derive `pass@k`.
74. Derive `pass^k`.
75. When is `pass^k` more important than `pass@k`?

## Benchmark interpretation

76. What is a Pareto frontier?
77. What is benchmark contamination?
78. How can contamination be reduced?
79. Explain Goodhart's law in the context of LLM evaluation.
80. Why do public benchmarks not replace custom product evals?
81. How would you choose between a more accurate expensive model and a slightly weaker cheap one?
82. What information must be versioned for reproducible evaluation?
83. Give an end-to-end evaluation plan for a new LLM feature.

---

# 16. Memory liners

```text
1. Evaluation closes the model-development feedback loop.

2. "Good" is meaningless until the criterion is explicit.

3. LLM quality is multi-dimensional, not one scalar.

4. Human judgments are closest to the target but costly and noisy.

5. Raw agreement must be compared against chance agreement.

6. Kappa subtracts chance and normalizes the remaining headroom.

7. Low agreement often means the annotation protocol needs repair.

8. Fixed references are reusable regression tests.

9. METEOR = overlap quality minus fragmentation penalty.

10. BLEU rewards n-gram precision and penalizes short outputs.

11. Semantic equivalence does not require lexical overlap.

12. An LLM judge is a scalable proxy, not an oracle.

13. Judge input = prompt + response + rubric + output schema.

14. Analyze first, score second.

15. Structured output makes evaluation machine-readable.

16. Pointwise asks "how good?"; pairwise asks "which is better?"

17. Swap A/B order to expose position bias.

18. Verbosity is not correctness.

19. A model may prefer its own stylistic distribution.

20. Use low temperature for reproducible judging.

21. Calibrate judges against humans before trusting them at scale.

22. Do not optimize a proxy past the point where it tracks the goal.

23. Factuality = decompose, retrieve, verify, aggregate.

24. One atomic claim should receive one evidence-backed verdict.

25. For agents, evaluate the trace, not only the final sentence.

26. Tool routing should maximize recall of the needed capability.

27. Missing context should not become fabricated arguments.

28. Every action tool should return explicit structured status.

29. Empty is meaningful; silence is ambiguous.

30. Verify world state rather than trusting the assistant's claim.

31. Constrained benchmark outputs reduce evaluator uncertainty.

32. pass@k measures one success; pass^k measures repeated reliability.

33. Public benchmarks profile models; custom evals select products.

34. Pareto-optimal means not dominated across the chosen trade-offs.

35. A benchmark seen during training is no longer a clean test.

36. Goodhart: once the measure becomes the target, the proxy can break.

37. Use the strongest verifier available: tests, exact answers, state, judge, then humans.
```

---

# 17. Previous-day interview checklist

## Must derive without notes

- [ ] Observed agreement `P_o`.
- [ ] Binary chance agreement `P_e`.
- [ ] Cohen's kappa.
- [ ] METEOR precision, recall, and fragmentation intuition.
- [ ] BLEU precision and brevity-penalty intuition.
- [ ] Weighted factuality score.
- [ ] `pass@k` and `pass^k`.
- [ ] Pareto domination definition.

## Must explain clearly

- [ ] Why LLM evaluation has no universal metric.
- [ ] Human vs reference vs LLM judge.
- [ ] Why inter-rater agreement needs a chance baseline.
- [ ] Pointwise vs pairwise judging.
- [ ] Structured output for judge parseability.
- [ ] Position, verbosity, and self-enhancement bias.
- [ ] Why low temperature is used for evaluation.
- [ ] How to calibrate a judge with human labels.
- [ ] Why factuality should be decomposed into claims.
- [ ] Why agent traces require stage-level evaluation.
- [ ] The seven major tool/agent failure patterns.
- [ ] MMLU, AIME, PIQA, SWE-bench, HarmBench, and tau-bench.
- [ ] Benchmark contamination and Goodhart's law.

## Must be able to design on a whiteboard

- [ ] A structured pointwise judge.
- [ ] An order-robust pairwise judge.
- [ ] A claim-level factuality pipeline using RAG/search.
- [ ] A tool-agent trace logger and error taxonomy.
- [ ] A benchmark suite for a domain-specific assistant.
- [ ] A quality-cost Pareto comparison.

## Final 60-second lecture answer

> Lecture 8 is about turning LLM quality into measurable evidence. Human ratings are the closest target but are expensive and subjective, so we monitor chance-corrected agreement such as Cohen's kappa. Reference metrics such as METEOR, BLEU, and ROUGE are cheap but brittle to paraphrases. LLM-as-a-judge provides scalable semantic scoring and rationales, but must use crisp rubrics, structured output, low temperature, bias checks, and calibration against humans. Factuality should be decomposed into atomic claims and verified against evidence. Agent evaluation must inspect tool routing, tool choice, arguments, backend execution, state change, and final grounding. Public benchmarks such as MMLU, AIME, PIQA, SWE-bench, HarmBench, and tau-bench measure different capability slices; they should be interpreted as a profile alongside cost and safety, with contamination and Goodhart's law kept in mind.

---

# 18. Lecture boundaries

The following topics are related to LLM evaluation but are **not developed in this lecture** and should remain in separate notes:

- formal confidence intervals and statistical power calculations;
- online A/B testing and causal experimentation;
- detailed Elo or Bradley-Terry leaderboard estimation;
- exact formulas for Fleiss' kappa and Krippendorff's alpha;
- learned semantic metrics beyond the rule metrics discussed;
- full red-teaming methodology;
- adversarial judge attacks and judge fine-tuning;
- process-supervision evaluation of hidden reasoning traces;
- exact provider-specific structured-output APIs;
- detailed benchmark dataset sizes and all benchmark variants;
- system-level latency, throughput, uptime, and serving-cost measurement;
- multimodal evaluation beyond the brief HarmBench mention.

**Final lecture memory line:**

> Evaluation is not one metric: it is a calibrated system of direct verifiers, human judgments, learned judges, failure taxonomies, and use-case-specific benchmarks.
