# CME 295 Lecture 7 - RAG, Tool Calling, MCP, ReAct, and Agentic LLMs

> **Scope:** This note is grounded in the supplied **CME 295 Lecture 7 transcript**. It adds standard equations, tensor-shape derivations, implementation mental models, and intuitive explanations only when they directly clarify a topic taught in this lecture. Additions not explicitly derived in class are labelled **INTERVIEW CLARIFICATION**.
>
> **Goal:** Previous-day revision for ML/AI Research Scientist interviews: important questions, distinctions, equations, tensor shapes, system dataflows, implementation ideas, failure modes, evaluation metrics, and memory liners.
>
> **Transcript normalization:** The automatic transcript occasionally renders **RAG** as “rack/drag,” **bi-encoder** as “by encoder,” **BERT** as “bird,” **HyDE** as “height,” and contains minor transcription errors around technical names. This note uses the standard terminology while preserving the lecture’s intended content.

**Source transcript:** Stanford CME 295, Lecture 7 — Agentic LLMs  
`https://www.youtube.com/watch/h-7S6HNq0Vg`

---

## Lecture spine

```text
LIMITATIONS OF A STANDALONE LLM
    static parametric knowledge
    finite context window
    distraction from irrelevant context
    no direct ability to act on external systems

RAG: RETRIEVAL-AUGMENTED GENERATION
    offline knowledge-base construction
        documents
          -> structure-aware chunking
          -> optional overlap / contextualization
          -> chunk embeddings
          -> ANN index

    online question answering
        query
          -> candidate retrieval
              dense semantic retrieval
              lexical retrieval / BM25
              or hybrid retrieval
          -> optional reranking
          -> top-k chunks
          -> augment prompt
          -> LLM generation

RETRIEVAL IMPROVEMENTS
    Sentence-BERT / bi-encoder embeddings
    approximate nearest-neighbour search
    HyDE for query-document mismatch
    contextual retrieval for decontextualized chunks
    prompt caching to reduce repeated-prefix cost
    cross-encoder reranking

RETRIEVAL EVALUATION
    precision@k
    recall@k
    reciprocal rank / MRR
    DCG / NDCG@k

TOOL CALLING
    expose tool name + description + argument schema
        -> LLM selects tool and predicts arguments
        -> application executes tool
        -> tool result returns to conversation
        -> LLM writes final user-facing response

SCALING TOOL USE
    SFT or in-context instruction
    tool selector / router
    only load full schemas for selected tools
    MCP standardizes tool exposure

AGENTS
    goal
      -> observe
      -> plan / reason
      -> act through tools
      -> observe result
      -> repeat until done or stopped

MULTI-AGENT SYSTEMS
    specialized agents
      -> communicate through a protocol such as A2A

SAFETY AND ENGINEERING
    data exfiltration and unsafe actions
    training-time harmlessness data
    inference-time safety checks
    bounded loops, validation, logging, and human approval
```

---

## Notation

```text
B             batch size / number of user queries
M             number of chunks in the knowledge base
K_cand        number of first-stage retrieval candidates
K             number of final chunks inserted into the prompt
T_q           query-token length
T_c           chunk-token length
T_ctx         total retrieved-context length
T_prompt      final augmented-prompt length
D_e           retrieval-embedding dimension
N_tools       number of available tools
K_tools       number of selected tools
S             maximum number of agent-loop steps
V             LLM vocabulary size

q             user query
c_j           chunk j
E_q           query embedding
E_c           matrix of chunk embeddings
s(q,c_j)      retrieval score
rel_j         ground-truth relevance label or grade
r_j           rank position of a document
```

### Priority legend

- **MUST REMEMBER:** answer immediately and derive the equation, tensor shape, or system flow.
- **SHOULD KNOW:** explain the mechanism, design trade-off, or failure mode clearly.
- **LECTURE BOUNDARY:** mentioned but not fully derived in this lecture.
- **INTERVIEW CLARIFICATION:** standard detail added to make the lecture implementation-ready.

---

# 1. What this lecture adds to the LLM mental map

## Q1. MUST REMEMBER — What are the three main topics of Lecture 7?

```text
1. Retrieval-augmented generation (RAG)
2. Tool / function calling
3. Agentic LLM workflows
```

**Memory liner:**

> RAG gives the model relevant knowledge; tools give it capabilities; agents repeatedly combine reasoning and tools to pursue a goal.

---

## Q2. MUST REMEMBER — Which limitations of a standalone LLM motivate this lecture?

The lecture focuses on two practical gaps:

1. **Knowledge is static:** the model’s parameters reflect training data available only up to a cutoff.
2. **The model cannot directly act:** ordinary text generation cannot itself query a live service, send an email, change a thermostat, or execute a computation.

A third issue appears in the RAG motivation: even a large context window is finite and can become less effective when overloaded with irrelevant information.

**Memory liner:**

> Parameters are stale, context is scarce, and text alone cannot change the world.

---

## Q3. MUST REMEMBER — How do RAG, tool calling, and agents differ?

| Mechanism | Main purpose | Typical control flow |
|---|---|---|
| RAG | Retrieve relevant unstructured information | retrieve → augment → generate |
| Tool calling | Invoke a structured capability | choose tool/arguments → execute → interpret result |
| Agent | Pursue a goal through multiple reasoning/action steps | observe → plan → act → observe → repeat |

**Memory liner:**

> RAG is contextual knowledge injection; a tool is one capability invocation; an agent is an iterative controller.

---

## Q4. SHOULD KNOW — Are these mechanisms mutually exclusive?

No. An agent may:

- retrieve documents through RAG;
- select and call tools;
- reason over the returned observations;
- call additional tools;
- finally answer the user.

**Memory liner:**

> RAG and tools are components; an agent can orchestrate both.

---

# 2. Why RAG is needed

## Q5. MUST REMEMBER — What is a knowledge cutoff?

A knowledge cutoff is the latest point represented by the model’s training data. Information occurring afterwards is not automatically encoded in the base model’s parameters.

```text
pre-training data ends at time t_cutoff
              ↓
model parameters reflect data up to t_cutoff
              ↓
event after t_cutoff is not known from parameters alone
```

**Memory liner:**

> Parametric knowledge is a frozen snapshot of the training distribution.

---

## Q6. MUST REMEMBER — Why not continually update the model weights whenever knowledge changes?

The lecture gives two major reasons:

1. **Knowledge editing can cause regressions:** changing weights to insert one fact may damage unrelated capabilities.
2. **Operational maintenance is expensive:** every downstream fine-tuned model or use case may need to be updated again.

**Memory liner:**

> Updating weights for every new fact is risky, expensive, and hard to propagate across model variants.

---

## Q7. MUST REMEMBER — Why not place the entire knowledge base in the prompt?

Three constraints:

1. **Finite context:** the input cannot contain an unlimited number of tokens.
2. **Retrieval degradation:** irrelevant or distant context can make the relevant fact harder to use.
3. **Cost and latency:** processing more input tokens costs more and takes longer.

**Memory liner:**

> More context is not free, not infinite, and not always easier for the model to use.

---

## Q8. MUST REMEMBER — What does the needle-in-a-haystack test measure?

It inserts a target fact—the **needle**—inside a long prompt—the **haystack**—and asks the model to recover it.

The lecture emphasizes that performance can depend on:

- total context length;
- where the relevant fact occurs;
- the amount of irrelevant information around it.

**Memory liner:**

> Context capacity is not the same as reliable retrieval from every position in that context.

---

## Q9. MUST REMEMBER — What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**.

```text
user query
   ↓
retrieve relevant external information
   ↓
append that information to the prompt
   ↓
generate an answer grounded in the augmented prompt
```

**Memory liner:**

> Do not put everything in context; retrieve only what is likely to matter.

---

## Q10. MUST REMEMBER — What are the three steps hidden in the acronym RAG?

```text
Retrieve
    find relevant chunks

Augment
    insert retrieved chunks into the prompt

Generate
    ask the LLM to answer using the augmented prompt
```

**Memory liner:**

> RAG = retrieve, augment, generate.

---

## Q11. MUST REMEMBER — Does RAG change the model weights?

Not necessarily. In the basic setup, RAG changes the **input context**, not the model parameters.

```text
same frozen LLM
+ different retrieved context
= different grounded answer
```

**Memory liner:**

> Fine-tuning changes the model; RAG changes what the model sees.

---

## Q12. MUST REMEMBER — Why is retrieval quality the central bottleneck?

The generator can only use evidence that reaches its prompt. If the relevant information is not retrieved, later stages cannot reliably recover it.

```text
bad retrieval
   ↓
missing or misleading context
   ↓
answer failure even with a strong LLM
```

**Memory liner:**

> Generation cannot compensate for evidence that never entered the context.

---

## Q13. INTERVIEW CLARIFICATION — What are the three broad failure locations in a RAG system?

```text
1. Retrieval failure
   relevant evidence is absent

2. Context-construction failure
   evidence is truncated, duplicated, poorly ordered, or mixed with noise

3. Generation failure
   the evidence is present but the model ignores, misreads, or contradicts it
```

This decomposition is useful for debugging even though the lecture mainly emphasizes the retrieval stage.

**Memory liner:**

> Diagnose RAG stage by stage; do not label every wrong answer “an LLM failure.”

---

# 3. Building the knowledge base

## Q14. MUST REMEMBER — What is the offline RAG indexing pipeline?

```text
collect documents
    ↓
clean / parse document structure
    ↓
split documents into chunks
    ↓
optionally add overlap or document-level context
    ↓
encode every chunk into an embedding
    ↓
store text + embedding in a searchable index
```

**Memory liner:**

> Build the searchable memory before queries arrive.

---

## Q15. MUST REMEMBER — What is a chunk?

A chunk is a bounded piece of a source document that acts as one retrieval unit.

```text
large document
    ↓ split
chunk_1, chunk_2, ..., chunk_M
```

The lecture describes chunk lengths on the order of hundreds of tokens as a common scale, not a universal rule.

**Memory liner:**

> Retrieval normally ranks document pieces, not entire corpora at once.

---

## Q16. MUST REMEMBER — What is the chunk-size trade-off?

### Too small

- loses surrounding context;
- ambiguous references may become uninterpretable;
- answer evidence may be split across chunks.

### Too large

- one embedding must summarize multiple topics;
- retrieval becomes less specific;
- retrieved context consumes more prompt tokens.

**Memory liner:**

> Small chunks are precise but context-poor; large chunks are coherent but semantically diluted.

---

## Q17. MUST REMEMBER — Why use overlapping chunks?

Overlap preserves context around arbitrary boundaries.

```text
chunk 1: tokens   1 ... 500
chunk 2: tokens 401 ... 900
                  ↑ shared region
```

**Benefit:** facts crossing the boundary can still appear together.  
**Cost:** more chunks, more storage, and more duplicate retrieval results.

**Memory liner:**

> Overlap protects boundary context but introduces redundancy.

---

## Q18. SHOULD KNOW — Why should chunking respect document structure?

A fixed token cut can split:

- a Markdown section from its heading;
- a JSON object in the middle;
- a code function from its signature;
- a table row from its column meaning.

The lecture explicitly notes that file structure matters even though it does not prescribe one universal chunker.

**Memory liner:**

> Chunk according to semantic and structural boundaries whenever possible.

---

## Q19. MUST REMEMBER — What does the embedding model do?

It maps a query or chunk into a vector space where relevant query–chunk pairs should have high similarity.


after encoding:

```text
query q   → E_q ∈ R^{D_e}
chunk c_j → E_j ∈ R^{D_e}
```

**Memory liner:**

> The retriever turns relevance into geometric closeness.

---

## Q20. SHOULD KNOW — Pre-trained embedding model or custom model?

The lecture gives both options:

- use a pre-trained retrieval embedding model, which is common;
- train or adapt a custom model when the domain or relevance definition is specialised.

**Memory liner:**

> Start with a strong pre-trained retriever; customise only when your data shows a domain gap.

---

## Q21. SHOULD KNOW — What is the embedding-dimension trade-off?

A larger embedding can provide more representational capacity, but increases:

- vector-index storage;
- similarity-search compute;
- memory bandwidth;
- sometimes latency.

The lecture gives dimensions around the low thousands as an illustrative order of magnitude, not a fixed recommendation.

**Memory liner:**

> More dimensions can represent more nuance, but every stored and compared vector becomes more expensive.

---

## Q22. MUST REMEMBER — What are the core indexing tensor shapes?

Let there be `M` chunks, each represented by `D_e` features.

```text
chunk token IDs:      [M, T_c]
chunk embeddings E_c:[M, D_e]
```

For a batch of queries:

```text
query token IDs:      [B, T_q]
query embeddings E_q:[B, D_e]
```

**Memory liner:**

> Retrieval compares a batch of query vectors against a matrix of stored chunk vectors.

---

## Q23. MUST REMEMBER — Offline versus online work in RAG?

### Offline

- document ingestion;
- chunking;
- chunk embedding;
- index construction;
- optional contextualization.

### Online

- query embedding;
- nearest-neighbour retrieval;
- reranking;
- prompt augmentation;
- answer generation.

**Memory liner:**

> Precompute stable document work; reserve query-dependent work for serving time.

---

## Q24. SHOULD KNOW — What must be stored besides the vector?

At minimum, the system needs a way to recover the original chunk text from a retrieved vector ID.

```text
vector ID
   -> chunk text
   -> source document / position
```

The lecture focuses on embeddings and chunks; rich metadata design is outside its detailed scope.

---

# 4. Candidate retrieval with dense embeddings

## Q25. MUST REMEMBER — What are the two retrieval stages?

```text
Stage 1: candidate retrieval
    huge corpus -> manageable candidate set
    optimise recall and speed

Stage 2: ranking / reranking
    candidate set -> final top-k
    use a more expensive, precise scorer
```

**Memory liner:**

> Retrieve broadly first; judge carefully second.

---

## Q26. MUST REMEMBER — Why optimise recall in the first stage?

If a relevant chunk is discarded during candidate retrieval, the reranker never gets a chance to recover it.

**Memory liner:**

> First-stage false negatives are unrecoverable downstream.

---

## Q27. MUST REMEMBER — What is a bi-encoder retriever?

The query and chunk are encoded independently:

```text
query  -> encoder -> query vector
chunk  -> encoder -> chunk vector
                     ↓
              similarity score
```

The same encoder weights may be shared, as in many Sentence-BERT-style setups, though asymmetric encoders are also possible.

**Memory liner:**

> Bi-encoder = encode separately, compare cheaply.

---

## Q28. MUST REMEMBER — Why are bi-encoders scalable?

Chunk embeddings can be precomputed once. At query time, only the query must be encoded, followed by vector search.

```text
expensive document encoder work: offline once
query encoder work: online per request
vector comparison: highly optimized
```

**Memory liner:**

> Independence enables precomputation.

---

## Q29. MUST REMEMBER — What is cosine similarity?


after L2 normalization or in its general form:

$$
\boxed{
\operatorname{cos}(q,c)
=
\frac{q^\top c}{\|q\|_2\,\|c\|_2}
}
$$

Higher values indicate vectors pointing in more similar directions.

**Memory liner:**

> Cosine measures directional alignment between query and chunk embeddings.

---

## Q30. MUST REMEMBER — What is the batched dense-retrieval shape flow?

```text
E_q: [B, D_e]
E_c: [M, D_e]

scores = E_q @ E_c^T
scores: [B, M]

top-k indices: [B, K_cand]
top-k scores:  [B, K_cand]
```

Equation:

$$
S=E_qE_c^\top
$$

**Memory liner:**

> Query matrix times transposed chunk matrix gives every query–chunk score.

---

## Q31. INTERVIEW CLARIFICATION — When are dot product, cosine, and L2 ranking equivalent?

If all embeddings are unit-normalized:

$$
\|q-c\|_2^2
=
\|q\|_2^2+\|c\|_2^2-2q^\top c
=
2-2q^\top c
$$

Therefore:

```text
max dot product
= max cosine similarity
= min squared L2 distance
```

for unit-norm embeddings.

**Memory liner:**

> Once vectors are normalized, several common similarity measures induce the same ordering.

---

## Q32. MUST REMEMBER — Why not compare the query against every vector exactly?

A linear scan over millions or billions of vectors can be too slow and expensive.

The lecture therefore introduces **approximate nearest-neighbour (ANN)** search, where the index narrows the search space and trades a small amount of exactness for much lower latency.

**Memory liner:**

> ANN avoids scoring the entire corpus for every query.

---

## Q33. SHOULD KNOW — What does ANN change and what does it not change?

ANN changes **how candidate vectors are found efficiently**. It does not change the conceptual relevance objective:

```text
find chunks with high similarity to the query embedding
```

The lecture does not derive a particular ANN indexing algorithm.

---

## Q34. MUST REMEMBER — What role does Sentence-BERT play?

Sentence-BERT-style models produce one vector per sequence and are trained so semantically related pairs are close enough for efficient similarity search.

**Memory liner:**

> Standard BERT contextualizes tokens; Sentence-BERT is adapted to produce useful sequence-level retrieval embeddings.

---

## Q35. INTERVIEW CLARIFICATION — What does a contrastive retrieval objective look like?

The lecture mentions contrastive training but does not derive it. A standard in-batch form is:

$$
\mathcal L_i
=
-\log
\frac{\exp(s(q_i,c_i^+)/\tau)}
{\sum_j \exp(s(q_i,c_j)/\tau)}
$$

where:

- `c_i^+` is the relevant chunk;
- other chunks in the batch act as negatives;
- `tau` is a temperature.

**Memory liner:**

> Pull relevant query–chunk pairs together and push competing chunks apart.

---

## Q36. SHOULD KNOW — What is a hard negative?

**INTERVIEW CLARIFICATION:** A hard negative is a chunk that looks plausible to the retriever but is not actually relevant. Such negatives provide a stronger learning signal than obviously unrelated text.

The lecture does not discuss hard-negative mining in detail.

---

## Q37. MUST REMEMBER — What is semantic retrieval?

Semantic retrieval looks for similar **meaning**, not necessarily overlapping surface words.

```text
query and chunk may share no exact keyword
but still receive high embedding similarity
```

**Memory liner:**

> Dense retrieval can match paraphrases.

---

## Q38. MUST REMEMBER — What is the query–document distribution mismatch?

Queries are often short questions, while indexed chunks are longer declarative passages. Encoding both with the same representation function may make their geometry less directly comparable.

**Memory liner:**

> “Question language” and “document language” are different input distributions.

---

## Q39. MUST REMEMBER — What is HyDE?

**HyDE** means **Hypothetical Document Embeddings**.

```text
user question
   ↓
LLM generates a hypothetical answer/document
   ↓
embed the hypothetical document
   ↓
retrieve real documents near that embedding
```

The generated document need not be factually correct; its purpose is to transform the query into document-like language for retrieval.

**Memory liner:**

> Convert the query into the kind of text the document encoder expects.

---

## Q40. SHOULD KNOW — Is HyDE guaranteed to help?

No. The lecture presents it as a technique to try, not a universal default. It adds:

- one extra LLM call;
- extra latency and cost;
- dependence on the hypothetical text being directionally useful.

**Memory liner:**

> HyDE is a retrieval heuristic, not a correctness guarantee.

---

## Q41. SHOULD KNOW — What is another solution to query–document mismatch?

Use separate encoders:

```text
query encoder     f_q(q)
document encoder  f_d(c)
```

This can specialize each side, but introduces additional training and maintenance complexity. The lecture notes that shared encoders are operationally simpler.

---

# 5. Lexical, hybrid, and contextual retrieval

## Q42. MUST REMEMBER — Dense retrieval versus lexical retrieval?

| Dense / semantic retrieval | Lexical retrieval |
|---|---|
| Matches meaning | Matches terms |
| Handles paraphrases | Handles exact names, identifiers, and rare keywords |
| Uses learned embeddings | Uses term-frequency-style statistics |
| May miss exact-token requirements | May miss semantic equivalence |

**Memory liner:**

> Dense retrieval asks “does it mean the same?”; lexical retrieval asks “does it contain the same terms?”

---

## Q43. MUST REMEMBER — What is BM25 at the level taught here?

BM25 is a heuristic lexical relevance score based on query-term overlap, while adjusting for effects such as term frequency and document length.

The lecture does not derive the exact BM25 equation.

**Memory liner:**

> BM25 is a strong exact-term retriever, not an embedding model.

---

## Q44. MUST REMEMBER — When is lexical retrieval especially valuable?

Examples include queries containing:

- a person or product name;
- an error code;
- an identifier;
- a rare acronym;
- an exact quoted phrase.

The lecture’s “Cuddly” versus “Huggy” example illustrates why semantic similarity alone may confuse related names.

**Memory liner:**

> Exact entities often require exact lexical evidence.

---

## Q45. MUST REMEMBER — What is hybrid retrieval?

Hybrid retrieval combines dense and lexical signals.

```text
semantic score
      +
lexical / BM25 score
      ↓
combined candidate ranking
```

**Memory liner:**

> Use embeddings for meaning and BM25 for exact terms.

---

## Q46. INTERVIEW CLARIFICATION — How can dense and lexical scores be combined?

Common options include:

1. normalize both scores and compute a weighted sum;
2. retrieve candidates from each system and take their union;
3. fuse rank positions rather than raw scores.

A simple weighted form is:

$$
s_{hybrid}(q,c)
=
\alpha s_{dense}(q,c)
+
(1-\alpha)s_{lexical}(q,c)
$$

The lecture teaches the hybrid idea but not one required fusion formula.

---

## Q47. MUST REMEMBER — What problem does contextual retrieval solve?

A chunk may be locally ambiguous when detached from its source document.

```text
raw chunk:
“Revenue increased by 12%.”

missing context:
which company, period, currency, and report section?
```

Contextual retrieval prepends a short document-aware description to the chunk before indexing it.

**Memory liner:**

> Give each chunk enough global context to make sense on its own.

---

## Q48. MUST REMEMBER — What is the contextual-chunking dataflow?

```text
whole document + target chunk
          ↓
LLM generates a short contextual prefix
          ↓
contextual prefix + original chunk
          ↓
embed and index this enriched chunk
```

**Memory liner:**

> Use the whole document to explain the local chunk before embedding it.

---

## Q49. SHOULD KNOW — What is the main cost of contextual retrieval?

It may require one LLM call per chunk, which can be expensive for a large corpus.

The lecture connects this directly to **prompt caching**, because the full document prefix may repeat across many chunk-contextualization calls.

---

## Q50. MUST REMEMBER — What is prompt caching?

If many LLM requests share the same long prefix, the provider or serving system can reuse the prefix’s previously computed internal state instead of recomputing it for every request.

```text
shared document prefix
       ↓ compute once
cached prefix state
       ├── chunk 1 suffix -> response 1
       ├── chunk 2 suffix -> response 2
       └── chunk 3 suffix -> response 3
```

**Memory liner:**

> Put repeated content first, compute its prefix once, and reuse it.

---

## Q51. INTERVIEW CLARIFICATION — Prompt caching versus ordinary KV caching?

- **KV caching:** reuse past token keys/values while decoding one sequence.
- **Prompt caching:** reuse a common prefix across separate requests or continuations.

Both exploit the deterministic left-to-right computation of the shared prefix, but they solve different serving problems.

---

## Q52. SHOULD KNOW — What should be placed early in a prompt to benefit from prompt caching?

Stable, repeated material such as:

- system instructions;
- a long source document;
- tool specifications;
- fixed task guidelines.

Per-request variables should follow the shared prefix.

**Memory liner:**

> Cache-friendly prompts place invariant text before variable text.

---

# 6. Cross-encoder reranking

## Q53. MUST REMEMBER — What is a cross-encoder reranker?

A cross-encoder processes the query and candidate chunk **jointly** and outputs a relevance score.

```text
[query ; separator ; chunk]
              ↓
       Transformer encoder
              ↓
       scalar relevance score
```

Because query and chunk tokens can directly attend to one another, the scorer can model fine-grained interactions that a bi-encoder may miss.

**Memory liner:**

> Cross-encoder = read the query and document together before judging relevance.

---

## Q54. MUST REMEMBER — Bi-encoder versus cross-encoder?

| Property | Bi-encoder | Cross-encoder |
|---|---|---|
| Query and chunk encoded | Independently | Jointly |
| Chunk vectors precomputable | Yes | No, not for a new query |
| Query–chunk token interactions | No direct cross-attention | Full interaction |
| Serving cost | Low | High |
| Typical role | Candidate retrieval | Reranking |

**Memory liner:**

> Bi-encoder is fast enough to search; cross-encoder is precise enough to rerank.

---

## Q55. MUST REMEMBER — Why not use a cross-encoder over the entire knowledge base?

For every query, every chunk would require a full joint forward pass:

```text
M chunks × one expensive Transformer pass per pair
```

This is usually infeasible at large `M`. The candidate retriever reduces the set first.

**Memory liner:**

> Spend expensive interaction modelling only after cheap retrieval has shrunk the search space.

---

## Q56. MUST REMEMBER — What are the reranking tensor shapes?

Assume `B` queries and `K_cand` candidates per query.

```text
paired token IDs:    [B, K_cand, T_pair]
flattened for model: [B * K_cand, T_pair]
encoder outputs:     [B * K_cand, T_pair, D]
scalar scores:       [B * K_cand]
reshaped scores:     [B, K_cand]
final top-k indices: [B, K]
```

**Memory liner:**

> Reranking scores every retained query–candidate pair, then sorts within each query.

---

## Q57. SHOULD KNOW — What does the reranker output?

Typically one relevance score per query–chunk pair. The score need not be a calibrated probability; it mainly needs to rank candidates correctly.

**Memory liner:**

> Retrieval needs a useful ordering, not necessarily a perfectly calibrated score.

---

## Q58. SHOULD KNOW — How is the final `K` chosen?

`K` trades off:

- chance of including all necessary evidence;
- prompt length and generation cost;
- amount of irrelevant context;
- duplication among overlapping chunks.

There is no universal best `K`; tune it on the target task.

---

## Q59. INTERVIEW CLARIFICATION — What failure modes can reranking introduce?

- the first-stage candidate set omitted the evidence;
- the reranker overfits lexical style rather than true relevance;
- individually relevant chunks are redundant;
- the best evidence requires combining multiple moderately ranked chunks;
- score calibration differs across query types.

**Memory liner:**

> Reranking improves ordering, but it cannot recover missing candidates.

---

## Q60. LECTURE BOUNDARY — How are rerankers trained?

The lecture states that a reranker can be pre-trained or custom-trained and mentions that multiple loss functions are possible, including contrastive-style approaches. It does not derive one required reranker loss.

---

# 7. Retrieval evaluation

## Q61. MUST REMEMBER — What labels are needed to evaluate a retriever?

For each query, you need to know which chunks are actually relevant, possibly with binary or graded relevance.

```text
query q
    -> relevant chunk set R_q
    -> retrieved ranked list L_q
```

The metrics compare `L_q` against `R_q`.

---

## Q62. MUST REMEMBER — What is precision@K?

$$
\boxed{
\operatorname{Precision@K}
=
\frac{|\operatorname{TopK}(q)\cap R_q|}{K}
}
$$

It asks:

> Of the `K` retrieved chunks, how many are actually relevant?

**Memory liner:**

> Precision@K measures purity of the retrieved context.

---

## Q63. MUST REMEMBER — What is recall@K?

$$
\boxed{
\operatorname{Recall@K}
=
\frac{|\operatorname{TopK}(q)\cap R_q|}{|R_q|}
}
$$

It asks:

> Of all relevant chunks, how many appeared in the top `K`?

**Memory liner:**

> Recall@K measures coverage of available evidence.

---

## Q64. MUST REMEMBER — Precision@K versus recall@K?

```text
precision@K:
    avoid irrelevant chunks

recall@K:
    avoid missing relevant chunks
```

Increasing `K` often raises recall while reducing or diluting precision.

**Memory liner:**

> Precision controls noise; recall controls missing evidence.

---

## Q65. MUST REMEMBER — What is reciprocal rank?

Let `rank_first` be the position of the highest-ranked relevant result:

$$
\boxed{
RR=
\frac{1}{\operatorname{rank}_{first\ relevant}}
}
$$

Examples:

```text
first relevant at rank 1 -> RR = 1
first relevant at rank 2 -> RR = 1/2
first relevant at rank 5 -> RR = 1/5
```

**Memory liner:**

> Reciprocal rank rewards finding one relevant item early.

---

## Q66. MUST REMEMBER — What is mean reciprocal rank?

Across `N` queries:

$$
\boxed{
MRR=
\frac{1}{N}
\sum_{i=1}^{N}
\frac{1}{\operatorname{rank}_{i,first\ relevant}}
}
$$

**Memory liner:**

> MRR averages how early the first useful result appears.

---

## Q67. MUST REMEMBER — What is DCG@K?

A common form is:

$$
\boxed{
DCG@K
=
\sum_{i=1}^{K}
\frac{2^{rel_i}-1}{\log_2(i+1)}
}
$$

For binary relevance, a simpler equivalent intuition is:

$$
DCG@K
\propto
\sum_{i=1}^{K}
\frac{rel_i}{\log_2(i+1)}
$$

Relevant items contribute more when placed near the top.

**Memory liner:**

> DCG accumulates relevance while discounting lower ranks.

---

## Q68. MUST REMEMBER — Why is it called “discounted cumulative gain”?

- **Gain:** relevant results add value.
- **Cumulative:** add contributions across the top `K`.
- **Discounted:** divide by a rank-dependent term, so later results contribute less.

---

## Q69. MUST REMEMBER — What is ideal DCG?

`IDCG@K` is the DCG obtained from the best possible ordering of the known relevant items.

```text
put highest-relevance chunks first
    ↓
compute DCG
    ↓
IDCG@K
```

---

## Q70. MUST REMEMBER — What is NDCG@K?

$$
\boxed{
NDCG@K
=
\frac{DCG@K}{IDCG@K}
}
$$

It normally lies in `[0,1]`, where `1` means the ranking is ideal under the chosen relevance labels.

**Memory liner:**

> NDCG says how close your ranking is to the best possible ranking for that query.

---

## Q71. SHOULD KNOW — Why normalize DCG?

Different queries may have different numbers and grades of relevant chunks. Dividing by the query-specific ideal score makes results more comparable across queries.

---

## Q72. MUST REMEMBER — What does NDCG capture that recall@K does not?

Recall@K only asks whether relevant chunks are somewhere in the top `K`. NDCG also cares **where** they appear and can handle graded relevance.

```text
same relevant set in top K
but better ordering
    -> same recall@K
    -> higher NDCG@K
```

---

## Q73. MUST REMEMBER — What does MRR ignore?

After the first relevant result, later relevant results do not affect reciprocal rank.

Therefore MRR is useful when one early answer is enough, but less informative when the system needs several complementary evidence chunks.

---

## Q74. MUST REMEMBER — Which retrieval metric should you use?

| Goal | Useful metric |
|---|---|
| Retrieve as much relevant evidence as possible | Recall@K |
| Keep augmented context clean | Precision@K |
| Put at least one answer-bearing result early | MRR |
| Reward the full ordering, possibly with graded relevance | NDCG@K |

**Memory liner:**

> The metric must match how retrieved chunks will be consumed downstream.

---

## Q75. SHOULD KNOW — What is MTEB?

The lecture mentions the **Massive Text Embedding Benchmark** as a broad benchmark suite used to compare embedding models across retrieval and related tasks.

**Memory liner:**

> MTEB evaluates embedding quality across many datasets and task families.

---

## Q76. LECTURE BOUNDARY — Do retrieval metrics fully evaluate a RAG system?

No. They evaluate whether the retriever ranks labelled chunks well. The lecture does not develop end-to-end metrics for:

- factual answer correctness;
- citation faithfulness;
- whether the model used the retrieved evidence;
- answer usefulness.

Keep retrieval evaluation and answer evaluation conceptually separate.

---

## Q77. SHOULD KNOW — How would you perform an offline retriever evaluation?

```text
for every labelled query:
    retrieve top K
    compare IDs against relevance labels
    compute precision@K, recall@K, RR, NDCG@K
average metrics across queries
slice results by query type
```

**Memory liner:**

> Evaluate on held-out queries that reflect the serving distribution.

---

## Q78. SHOULD KNOW — What does high recall but low precision mean?

The retriever finds most useful evidence but also inserts many irrelevant chunks. This can increase token cost and distract the generator.

---

## Q79. SHOULD KNOW — What does high NDCG but low recall mean?

The retrieved relevant chunks are ordered well, but some required evidence never appears in the top `K`. This may be acceptable for single-fact questions but harmful for multi-document reasoning.

---

## Q80. SHOULD KNOW — Why evaluate candidate retrieval and reranking separately?

```text
candidate retriever:
    did the relevant item enter K_cand?

reranker:
    given that it entered K_cand, was it moved into final K?
```

This tells you which stage is responsible for a miss.

---

# 8. End-to-end RAG implementation mental map

## Q81. MUST REMEMBER — What is the complete RAG dataflow?

```text
OFFLINE
documents
  -> parse
  -> chunk
  -> optional contextual prefix
  -> embed chunks
  -> build vector / lexical index

ONLINE
user query
  -> encode query
  -> dense / lexical / hybrid retrieval
  -> K_cand candidates
  -> optional cross-encoder reranking
  -> K final chunks
  -> construct augmented prompt
  -> LLM generation
  -> final answer
```

---

## Q82. MUST REMEMBER — PyTorch-like dense retrieval pseudocode?

```python
# query_ids: [B, T_q]
# chunk_vectors: [M, D_e], precomputed offline

query_vectors = query_encoder(query_ids)       # [B, D_e]
query_vectors = normalize(query_vectors)       # [B, D_e]
chunk_vectors = normalize(chunk_vectors)       # [M, D_e]

scores = query_vectors @ chunk_vectors.T       # [B, M]
top_scores, top_ids = scores.topk(K_cand, dim=-1)
```

In production, an ANN index normally replaces the explicit `[B,M]` matrix computation.

---

## Q83. MUST REMEMBER — Cross-encoder reranking pseudocode?

```python
candidate_texts = lookup_chunks(top_ids)            # B groups of K_cand texts
paired_ids = tokenize_query_chunk_pairs(
    queries,
    candidate_texts,
)                                                    # [B, K_cand, T_pair]

flat_ids = paired_ids.reshape(B * K_cand, T_pair)
flat_scores = reranker(flat_ids)                    # [B * K_cand]
scores = flat_scores.reshape(B, K_cand)             # [B, K_cand]

reranked_scores, order = scores.sort(descending=True)
final_ids = gather(top_ids, order[:, :K])            # [B, K]
```

---

## Q84. MUST REMEMBER — How is the final prompt constructed?

Conceptually:

```text
system instruction
+ explicit rule to use provided evidence
+ retrieved chunk 1
+ retrieved chunk 2
+ ...
+ user question
```

The lecture’s key idea is augmentation with relevant chunks. Exact formatting and ordering are implementation choices.

---

## Q85. SHOULD KNOW — What happens to prompt length?

If each of `K` chunks has about `T_c` tokens:

$$
T_{ctx}\approx K T_c
$$

and:

$$
T_{prompt}
=
T_{system}+T_q+T_{ctx}+T_{formatting}
$$

This exposes the key design trade-off: more evidence versus cost and distraction.

---

## Q86. MUST REMEMBER — RAG versus fine-tuning?

| RAG | Fine-tuning |
|---|---|
| Changes context | Changes parameters |
| Easy to update documents | Updating requires training |
| Can expose sources at inference | Knowledge is implicit in weights |
| Good for current/private knowledge | Good for behavior, style, or task adaptation |
| Retrieval can fail | Training can regress or overfit |

**Memory liner:**

> Use RAG to supply facts; use fine-tuning to change behavior.

---

## Q87. MUST REMEMBER — RAG versus long-context prompting?

```text
long-context prompting:
    put a large corpus directly in the prompt

RAG:
    search first, then insert a small relevant subset
```

RAG usually saves tokens and reduces irrelevant context, but adds a retrieval subsystem that can miss evidence.

---

## Q88. MUST REMEMBER — RAG versus tool calling?

```text
RAG:
    retrieve unstructured text chunks
    add them as evidence

Tool call:
    invoke a structured function/API
    receive a structured result or perform an action
```

A search API can blur the boundary, but the control interfaces are still different.

---

## Q89. SHOULD KNOW — How do you debug a wrong RAG answer?

```text
1. Was the correct source ingested?
2. Did chunking preserve the needed evidence?
3. Did candidate retrieval include it?
4. Did reranking keep it in final K?
5. Was the chunk truncated or duplicated in the prompt?
6. Did the LLM use the provided evidence correctly?
```

**Memory liner:**

> Inspect the retrieval trace before changing the generator.

---

## Q90. SHOULD KNOW — What is a practical RAG design checklist?

```text
corpus freshness
chunk boundaries
chunk size and overlap
embedding model
lexical versus dense versus hybrid retrieval
ANN configuration
candidate count
reranker
final top-k
prompt budget
retrieval metrics
end-to-end answer tests
```

---

# 9. Tool and function calling fundamentals

## Q91. MUST REMEMBER — What is tool calling?

At the lecture’s level, tool calling lets an LLM-driven system dynamically access or act upon an external resource in order to complete a task.

```text
natural-language intent
   ↓
structured tool invocation
   ↓
external computation / information / action
   ↓
structured observation
   ↓
user-facing response
```

**Memory liner:**

> Tool calling turns language intent into a structured external operation.

---

## Q92. MUST REMEMBER — Function calling versus tool calling?

The lecture uses function calling as the concrete programming framing of tool calling:

- **tool:** the capability exposed to the model;
- **function call:** one structured invocation of that capability.

The concepts are closely related, though a tool need not literally be implemented as a local Python function.

---

## Q93. MUST REMEMBER — How does tool calling differ from RAG?

RAG primarily inserts unstructured retrieved evidence. Tool calling invokes a structured interface whose inputs and outputs follow an API contract.

**Memory liner:**

> RAG returns passages; tools return results or side effects.

---

## Q94. MUST REMEMBER — What information about a function does the LLM need?

The lecture emphasizes:

```text
function / tool name
human-readable description
argument names and meanings
input schema / types
output structure or semantics
```

The model does **not** need the private implementation code.

**Memory liner:**

> Expose the contract, not the implementation.

---

## Q95. MUST REMEMBER — Why is the tool description important?

The model uses the description to decide:

- when the tool is relevant;
- which user information maps to each argument;
- what kind of result to expect.

Poor descriptions create tool-selection and argument-grounding failures.

---

## Q96. MUST REMEMBER — What are the three main stages of a tool call?

```text
Stage 1: LLM predicts tool name + arguments
Stage 2: application/runtime executes the tool
Stage 3: tool result is returned to the LLM, which writes the final response
```

**Memory liner:**

> The model proposes; the runtime executes; the model interprets.

---

## Q97. MUST REMEMBER — Does the LLM execute the function?

No. The model emits a structured request. Application code validates and executes it.

```text
LLM output:
    {tool_name, arguments}

runtime:
    validate -> call implementation -> collect result
```

**Memory liner:**

> A tool call is data emitted by the model, not execution performed inside the model.

---

## Q98. MUST REMEMBER — Why feed the tool result back to the model?

Tool output is often structured and not user-friendly. The model combines:

- the original user request;
- conversation history;
- tool invocation;
- tool result;

and produces a natural-language answer.

---

## Q99. INTERVIEW CLARIFICATION — What can a tool schema look like?

```json
{
  "name": "find_teddy_bear",
  "description": "Find available teddy bears near a geographic location.",
  "parameters": {
    "type": "object",
    "properties": {
      "latitude": {"type": "number"},
      "longitude": {"type": "number"},
      "max_distance_km": {"type": "number"}
    },
    "required": ["latitude", "longitude"]
  }
}
```

The exact schema format depends on the tool platform; the lecture’s essential idea is explicit documentation and structured arguments.

---

## Q100. MUST REMEMBER — What might the model emit?

```json
{
  "name": "find_teddy_bear",
  "arguments": {
    "latitude": 37.4275,
    "longitude": -122.1697,
    "max_distance_km": 10
  }
}
```

The application should parse and validate this object before execution.

---

## Q101. MUST REMEMBER — What might a structured tool result look like?

```json
{
  "items": [
    {
      "name": "Cuddly",
      "distance_km": 1.8,
      "store": "Campus Toy Shop"
    }
  ]
}
```

The model can then produce: “Cuddly is available 1.8 km away at Campus Toy Shop.”

---

## Q102. SHOULD KNOW — Where can tool arguments come from?

Arguments may be grounded in:

- the current user message;
- earlier conversation turns;
- system-provided context such as location or current date;
- results from previous tool calls.

**Memory liner:**

> Tool argument prediction is structured grounding over the entire available context.

---

## Q103. MUST REMEMBER — What three broad tool categories does the lecture give?

```text
Information tools
    search, weather, market data, live availability

Computation tools
    calculators, code execution, database queries

Action tools
    send email, change a device setting, perform a transaction
```

**Memory liner:**

> Tools can read, compute, or act.

---

## Q104. MUST REMEMBER — Why use a computation tool when the model can reason?

A deterministic calculator or code executor can be more reliable for exact arithmetic or program execution. The LLM can translate the natural-language problem into a structured computation, execute it, and explain the result.

**Memory liner:**

> Let the LLM decide what to compute; let the tool perform the exact computation.

---

## Q105. SHOULD KNOW — Why should tool output be structured?

Structured output makes it easier to:

- parse reliably;
- validate types;
- distinguish fields;
- feed the result back into the conversation;
- write deterministic tests.

---

## Q106. INTERVIEW CLARIFICATION — What validation should happen before execution?

```text
schema validation
required-field validation
type and range checks
authorization check
confirmation for sensitive actions
normalization / escaping
```

The lecture’s safety section motivates these controls, although it does not enumerate this exact validation list.

---

# 10. Teaching a model to use tools

## Q107. MUST REMEMBER — What two model behaviours may need supervision?

The lecture identifies two mappings:

```text
Mapping 1:
conversation + tool specification
    -> correct tool name and arguments

Mapping 2:
conversation + tool call + tool result
    -> correct final natural-language response
```

**Memory liner:**

> Learn both how to call the tool and how to communicate what came back.

---

## Q108. MUST REMEMBER — What does the first SFT example teach?

It teaches tool selection and argument prediction.

Example:

```text
input:
    system tool schema
    user: “Find a teddy bear near me.”
    location context: Stanford coordinates

supervised target:
    find_teddy_bear(latitude=..., longitude=...)
```

---

## Q109. MUST REMEMBER — What does the second SFT example teach?

It teaches the model to use the tool observation in the context of the full conversation.

```text
input:
    user request
    assistant tool call
    tool result

supervised target:
    final helpful response to the user
```

**Memory liner:**

> The second target is not “JSON to English” in isolation; it is conversation-grounded response generation.

---

## Q110. MUST REMEMBER — Why include full conversation history?

Without history, the model may not know:

- why the tool was called;
- which user constraint matters;
- how to phrase the answer;
- whether more work is needed.

**Memory liner:**

> A tool result is meaningful only relative to the goal that produced it.

---

## Q111. SHOULD KNOW — Why must tool-use examples be diverse?

The lecture stresses that examples should reflect the serving distribution:

- explicit location in the user message;
- implicit location from system context;
- multi-turn references;
- paraphrased requests;
- different argument combinations.

**Memory liner:**

> Train on the ways users actually express intent, not one canonical sentence.

---

## Q112. INTERVIEW CLARIFICATION — What can the training data format look like?

```python
messages = [
    {"role": "system", "content": tool_schema_and_instructions},
    {"role": "user", "content": "Find a teddy bear near me."},
    {
        "role": "assistant",
        "tool_call": {
            "name": "find_teddy_bear",
            "arguments": {"latitude": 37.42, "longitude": -122.17},
        },
    },
    {
        "role": "tool",
        "name": "find_teddy_bear",
        "content": {"name": "Cuddly", "distance_km": 1.8},
    },
    {
        "role": "assistant",
        "content": "Cuddly is available about 1.8 km away.",
    },
]
```

The exact serialization depends on the model family and API.

---

## Q113. INTERVIEW CLARIFICATION — What are the SFT tensor shapes?

After serializing each conversation:

```text
input_ids:      [B, T]
attention_mask: [B, T]
labels:         [B, T]
logits:         [B, T, V]
```

A loss mask can select assistant/tool-call target tokens while excluding fixed system and user tokens.

```text
loss_mask[t] = 1 for supervised assistant output
loss_mask[t] = 0 for conditioning context
```

This is standard SFT machinery; the lecture explains the two supervised mappings rather than deriving the tensorized loss.

---

## Q114. MUST REMEMBER — Is SFT the only way to teach a new tool?

No. The lecture gives alternatives:

- place the schema and clear instructions in the prompt;
- use few-shot tool-call demonstrations;
- use a stronger reasoning model to improve the tool-use instructions offline.

**Memory liner:**

> Strong models can often learn a tool from its contract and examples without weight updates.

---

## Q115. SHOULD KNOW — What is the benefit and limitation of few-shot tool examples?

### Benefit

- quick to add;
- no model training;
- concretely demonstrates the expected format.

### Limitation

- consumes context tokens;
- may overfit to the demonstrated phrasings;
- finite examples may not cover the full language distribution.

---

## Q116. MUST REMEMBER — What offline prompt-optimization loop does the lecture describe?

```text
1. Start with a draft tool-use explanation.
2. Build an evaluation set of query -> expected tool call pairs.
3. Run the current prompt on the evaluation set.
4. Collect successes and failures.
5. Give failures and current instructions to a strong reasoning model.
6. Ask it to improve the instructions.
7. Repeat offline.
8. Freeze the resulting instruction for serving.
```

**Memory liner:**

> Treat tool instructions as a program that can be evaluated and iteratively improved.

---

## Q117. MUST REMEMBER — Is that prompt-optimization loop training or inference?

It is **offline development-time optimization**. The final fixed instruction and tool schema are then used at inference time.

**Memory liner:**

> Optimise the instruction before deployment; do not rewrite it for every user request.

---

## Q118. SHOULD KNOW — When would you prefer prompting over SFT?

Prompting is attractive when:

- tools change frequently;
- the base model already follows schemas well;
- there is limited labelled tool-use data;
- fast iteration matters more than minimum prompt length.

SFT becomes more attractive when:

- tool use is frequent and stable;
- strict format reliability is needed;
- prompt cost is material;
- the model repeatedly fails from instructions alone.

---

## Q119. SHOULD KNOW — What is argument grounding?

Argument grounding is mapping natural-language information to the exact structured fields required by a tool.

```text
“near Stanford, within five miles”
      ↓
latitude, longitude, max_distance_km
```

Typical failures include missing fields, wrong units, stale context, and confusing two entities.

---

## Q120. INTERVIEW CLARIFICATION — What should happen when required information is missing?

A robust system can:

- ask a clarifying question;
- infer only when permitted and reliable;
- call another tool to obtain the missing value;
- refuse execution if a required or sensitive field is unresolved.

The lecture does not prescribe one error-recovery policy, but its multi-step agent discussion motivates this behaviour.

---

# 11. Scaling from one tool to many

## Q121. MUST REMEMBER — What goes wrong when every tool schema is placed in context?

- context window consumption;
- higher input cost and latency;
- tool descriptions become needles in a haystack;
- conflicting or similar APIs increase selection errors;
- the model may become mediocre across too many possibilities.

**Memory liner:**

> A large tool catalog creates the same retrieval problem as a large document corpus.

---

## Q122. MUST REMEMBER — What is a tool selector or router?

A tool selector is a first-stage system that chooses a small subset of potentially relevant tools from a large catalog.

```text
user query + short tool summaries
              ↓
        tool selector/router
              ↓
       K_tools selected tools
              ↓
load their full schemas for tool calling
```

**Memory liner:**

> Route first, expose detailed tool contracts second.

---

## Q123. MUST REMEMBER — What are the two stages of scalable tool selection?

```text
Stage 1:
    query + compact list of tool names/descriptions
    -> select relevant tool IDs

Stage 2:
    query + full schemas only for selected tools
    -> choose tool and arguments
```

This parallels candidate retrieval followed by precise ranking or execution.

---

## Q124. MUST REMEMBER — Is tool selection itself RAG?

It can be implemented using retrieval, but does not have to be. The lecture gives two broad possibilities:

- an LLM directly selects tools from short descriptions;
- a retrieval system matches the query to tool descriptions.

**Memory liner:**

> Tool routing is a selection problem; retrieval is one possible implementation.

---

## Q125. INTERVIEW CLARIFICATION — What are useful tool-selector shapes?

For `B` queries and `N_tools` tool descriptors:

```text
query embeddings: [B, D_e]
tool embeddings:  [N_tools, D_e]
selector scores:  [B, N_tools]
selected IDs:     [B, K_tools]
```

An LLM-based selector may instead emit a structured list of tool IDs.

---

## Q126. SHOULD KNOW — What should a compact tool catalog contain?

The lecture suggests a minimal representation such as:

```text
tool name
+ one- or two-line purpose description
```

Full argument schemas are loaded only after selection.

---

## Q127. SHOULD KNOW — What are tool-selector failure modes?

```text
false negative:
    needed tool not selected -> task becomes impossible

false positive:
    irrelevant tools selected -> extra context and confusion

ambiguous selection:
    several near-duplicate tools compete
```

As with RAG, first-stage recall is important.

---

# 12. Model Context Protocol (MCP)

## Q128. MUST REMEMBER — What problem does MCP address?

Without a standard, every model host and tool provider may define bespoke integration formats. MCP aims to standardize how tools and related context are exposed to an LLM application.

**MCP** stands for **Model Context Protocol**.

**Memory liner:**

> MCP is an interoperability layer between model hosts and external capabilities/context.

---

## Q129. MUST REMEMBER — What is an MCP server at the lecture’s level?

An MCP server exposes capabilities and context, including tools that a model host can use.

```text
book-provider MCP server
    -> search_books tool
    -> recommend_books tool
    -> prompts / resources
```

---

## Q130. MUST REMEMBER — What is an MCP client?

The MCP client is the component inside or connected to the LLM host that communicates with an MCP server.

```text
LLM host
   └── MCP client <---- protocol ----> MCP server
```

---

## Q131. MUST REMEMBER — What MCP concepts are introduced?

| Concept | Lecture-level meaning |
|---|---|
| Tools | Callable functions/capabilities |
| Prompts | Templates showing or supporting use of capabilities |
| Resources | External information or data the server can expose |
| Server | Provider of tools/prompts/resources |
| Client | Host-side connection to the server |

**Memory liner:**

> Tools act, prompts guide, resources inform, servers expose, and clients connect.

---

## Q132. SHOULD KNOW — What practical benefit does standardization provide?

- avoid rewriting the same tool integration for every model host;
- let specialist providers expose capabilities once;
- improve portability and ecosystem reuse;
- make discovery and integration more systematic.

---

## Q133. LECTURE BOUNDARY — What MCP details are outside this lecture?

The lecture gives the architecture vocabulary but does not derive:

- the exact wire protocol;
- transport mechanisms;
- authentication and permission models;
- lifecycle and capability-negotiation details;
- production deployment architecture.

Do not confuse the high-level mental map with the complete specification.

---

## Q134. MUST REMEMBER — MCP versus RAG?

```text
RAG:
    retrieve text evidence from a corpus

MCP:
    standard protocol for exposing tools, prompts, and resources
```

An MCP server could expose a search/retrieval tool, but MCP itself is not a retrieval algorithm.

---

## Q135. MUST REMEMBER — MCP versus tool selector?

```text
tool selector:
    decide which tools are relevant

MCP:
    standardize how tools/context are exposed and connected
```

They solve complementary problems.

---

# 13. What makes a system an agent?

## Q136. MUST REMEMBER — How does the lecture define an agent?

An agent is a system that autonomously pursues a goal and completes tasks on a user’s behalf.

The defining practical feature is not simply one tool call, but an iterative reasoning-and-action loop.

**Memory liner:**

> An agent repeatedly decides what to do next until the goal is reached or execution stops.

---

## Q137. MUST REMEMBER — Tool call versus agent?

| Tool call | Agent |
|---|---|
| Usually one structured operation | Potentially many operations |
| Caller already knows the immediate action | System decides successive actions |
| Limited local state | Maintains task state/history |
| No inherent loop | Repeats observation, reasoning, and action |

**Memory liner:**

> A tool is an action primitive; an agent is the controller that chooses and sequences primitives.

---

## Q138. MUST REMEMBER — What is ReAct?

**ReAct** combines **reasoning** and **acting**. A model alternates between thinking about the task, invoking tools, and incorporating the observations returned by those tools.

```text
reason / plan
    ↓
action / tool call
    ↓
observation
    ↓
reason again
```

**Memory liner:**

> Think, act, observe, and repeat.

---

## Q139. SHOULD KNOW — Why do papers use different loop terminology?

The lecture uses **observe → plan → act**, while ReAct is often framed with terms such as **thought → action → observation**. Names and exact ordering vary, but the core pattern is iterative state interpretation and action selection.

---

## Q140. MUST REMEMBER — What does the observe stage do?

It interprets the current situation:

- what the user wants;
- what is already known;
- what the latest tool result means;
- whether the goal is satisfied;
- what remains unknown.

**Memory liner:**

> Observation turns raw input or tool output into task state.

---

## Q141. MUST REMEMBER — What does the plan stage do?

It chooses the next subgoal or action needed to make progress.

```text
unknown room temperature
    ↓
plan: measure current room temperature
```

**Memory liner:**

> Planning converts the remaining gap into an actionable next step.

---

## Q142. MUST REMEMBER — What does the act stage do?

It invokes the selected tool with grounded arguments and receives an external result or side effect.

```text
plan: measure temperature
    ↓
act: get_room_temperature()
    ↓
observation: 65°F
```

---

## Q143. MUST REMEMBER — Walk through the thermostat example.

```text
User goal:
    “My teddy bear is cold. Do something.”

Observe:
    cold may be caused by low room temperature;
    current temperature is unknown.

Plan:
    obtain room temperature.

Act:
    call get_current_room_temperature().

Observe:
    room is 65°F, colder than desired.

Plan:
    increase temperature.

Act:
    call adjust_temperature(+5°F).

Observe:
    target state reached.

Finish:
    tell the user what was done.
```

**Memory liner:**

> An agent converts a vague goal into a sequence of grounded, testable actions.

---

## Q144. MUST REMEMBER — What is the agent stopping condition?

The loop exits when the system judges that:

- the goal has been achieved;
- no further action is needed;
- execution cannot continue safely or successfully;
- a configured budget or limit has been reached.

The lecture explicitly focuses on checking whether the goal has been reached; budgets and safeguards are natural engineering controls.

---

## Q145. INTERVIEW CLARIFICATION — What state does an agent typically maintain?

```text
user goal
conversation history
current plan / subgoal
available tools
past tool calls
observations / results
step count and token budget
permissions and safety state
```

**Memory liner:**

> An agent is an LLM plus state, tools, a loop, and termination rules.

---

## Q146. MUST REMEMBER — High-level agent pseudocode?

```python
def run_agent(goal, tools, max_steps):
    state = initialize_state(goal)

    for step in range(max_steps):
        observation = summarize_current_state(state)

        decision = llm_choose_next_step(
            goal=goal,
            observation=observation,
            tools=tools,
        )

        if decision.type == "finish":
            return decision.final_answer

        validate_tool_call(decision.tool_name, decision.arguments)
        result = execute_tool(decision.tool_name, decision.arguments)
        state.add(decision, result)

    return safe_incomplete_response(state)
```

The lecture presents the loop conceptually rather than prescribing one code framework.

---

## Q147. MUST REMEMBER — Why are agents more capable than one-shot tool use?

They can:

- discover missing information;
- react to tool results;
- revise a plan;
- combine multiple tools;
- check whether a goal has been achieved;
- recover from some intermediate failures.

---

## Q148. MUST REMEMBER — Why are agents less reliable than a single deterministic workflow?

Every step introduces another chance to:

- misunderstand the state;
- select the wrong tool;
- predict invalid arguments;
- misread an observation;
- choose an unnecessary action;
- fail to stop.

**Memory liner:**

> Agentic flexibility increases both capability and compounding error risk.

---

## Q149. INTERVIEW CLARIFICATION — Deterministic workflow versus agent?

```text
deterministic workflow:
    developer fixes the control graph in advance

agent:
    model decides at runtime which branch/action comes next
```

Use an agent only when runtime uncertainty and task variation justify model-controlled branching.

---

## Q150. MUST REMEMBER — How do reasoning models and agents overlap?

An agent can use a reasoning model as its controller. The model reasons about observations and selects tools, while the environment provides new evidence between reasoning steps.

```text
reasoning without tools:
    internal token trajectory

agentic reasoning:
    internal reasoning interleaved with external observations/actions
```

---

# 14. Multi-agent systems and A2A

## Q151. MUST REMEMBER — Why use multiple agents?

Different agents can specialise in different goals or tool sets, for example:

- thermostat control;
- energy management;
- air-quality management.

A higher-level user request may require coordination among them.

---

## Q152. SHOULD KNOW — Is every agent necessarily a different foundation model?

No. In the lecture’s example, an agent can be understood as an LLM configured with its own context, skills, tools, and reasoning loop. Multiple agents may share a base model while operating with different roles and state.

---

## Q153. MUST REMEMBER — What is the Agent-to-Agent protocol at the lecture’s level?

The lecture introduces **A2A** as a standardization effort for communication between agents, analogous in spirit to MCP standardizing model-to-tool/context connections.

**Memory liner:**

> MCP connects hosts to tools; A2A helps agents communicate with agents.

---

## Q154. MUST REMEMBER — What does an agent expose in the A2A mental map?

The lecture mentions:

- a set of **skills** or capabilities;
- examples/descriptions that let other agents understand those skills;
- execution status;
- mechanisms such as task cancellation.

---

## Q155. SHOULD KNOW — Why expose execution status?

Long-running actions may be:

- queued;
- running;
- completed;
- failed;
- cancelled.

Other agents need this state to coordinate rather than assume instant synchronous completion.

---

## Q156. SHOULD KNOW — What new failures appear in multi-agent systems?

```text
misrouted tasks
incompatible assumptions
conflicting goals
stale shared state
circular delegation
unbounded cost
partial failure and cancellation problems
```

The lecture focuses on the need for standardized communication and budget awareness rather than deriving a full coordination algorithm.

---

# 15. Agent and tool safety

## Q157. MUST REMEMBER — Why do tools and agents raise the safety stakes?

A text-only failure may produce a bad answer. A tool-enabled failure can cause an external side effect:

- disclose data;
- send a message;
- modify a system;
- spend money;
- execute harmful code.

**Memory liner:**

> Once the model can act, errors become state changes rather than only words.

---

## Q158. MUST REMEMBER — What is data exfiltration?

Data exfiltration is unauthorized movement of sensitive information out of a protected context.

The lecture’s example is a tool-enabled system that gains access to a secret and sends it through an external channel such as email.

---

## Q159. MUST REMEMBER — What two broad safety layers does the lecture describe?

```text
Training-time safeguards
    SFT / RL data teaching harmless behavior

Inference-time safeguards
    classifier or policy checks over the conversation and proposed action
```

**Memory liner:**

> Train safe tendencies, then enforce runtime gates.

---

## Q160. SHOULD KNOW — What is an inference-time safety classifier doing?

It examines the current conversation, proposed tool call, or response and predicts whether execution is safe enough to permit.

```text
proposed action
      ↓
safety classifier / policy engine
      ├── allow
      ├── block
      └── require human approval
```

---

## Q161. INTERVIEW CLARIFICATION — What is least privilege?

Give the agent only the minimum permissions required for the task.

```text
read-only search agent
    should not receive write/delete credentials
```

This is a standard systems principle strongly motivated by the lecture’s data-exfiltration example.

---

## Q162. INTERVIEW CLARIFICATION — Which actions should require confirmation?

High-impact or irreversible actions such as:

- sending messages externally;
- spending money;
- deleting data;
- changing access permissions;
- publishing content;
- executing code against production systems.

**Memory liner:**

> Autonomy should shrink as action consequence grows.

---

## Q163. INTERVIEW CLARIFICATION — Why validate tool output as well as input?

A tool result may be malformed, adversarial, stale, or contain instructions that should not be treated as trusted control text.

Useful controls include:

- schema validation;
- provenance tracking;
- escaping/untrusted-data boundaries;
- size limits;
- explicit interpretation rules.

The lecture does not develop prompt-injection defenses in detail, so treat this as a direct safety extension rather than a derived lecture result.

---

## Q164. SHOULD KNOW — What does Agent SafetyBench contribute at the lecture level?

The lecture mentions it as a benchmark suite covering a range of agent safety hazards, useful for testing whether tool-enabled systems behave safely.

---

## Q165. MUST REMEMBER — Why should agent loops be bounded?

Without limits, a model can repeatedly call tools, consume tokens, increase latency/cost, or wander away from the goal.

Useful bounds include:

```text
maximum steps
maximum tool calls
maximum tokens
wall-clock timeout
monetary budget
per-tool limits
```

**Memory liner:**

> Every agent loop needs an explicit stopping budget.

---

## Q166. MUST REMEMBER — Why are traces important for debugging?

The system should record:

- model decisions;
- selected tool;
- predicted arguments;
- tool response;
- updated plan;
- termination reason.

This lets developers locate the first divergence rather than only inspecting the final wrong answer.

**Memory liner:**

> Debug the trajectory, not just the endpoint.

---

## Q167. MUST REMEMBER — What practical development advice closes the lecture?

```text
1. Start with a small, simple task.
2. Use a highly capable model first to test feasibility.
3. Make correctness and observability work.
4. Only then optimise latency, cost, and model size.
```

**Memory liner:**

> Start correct, start small, then optimise.

---

## Q168. SHOULD KNOW — Why does “taste” matter for coding agents?

Generating code becomes cheap, but judging whether it is correct, maintainable, safe, and appropriate remains difficult. Human technical foundations are needed to evaluate and steer the generated work.

**Memory liner:**

> Code generation is abundant; reliable code judgment remains scarce.

---

# 16. Consolidated comparison tables

## RAG versus tool calling versus agents

| Question | RAG | Tool calling | Agent |
|---|---|---|---|
| Primary problem | Missing/current knowledge | Structured external capability | Multi-step goal completion |
| Typical unit | Text chunk | Function/API call | Iterative trajectory |
| Control loop | Usually one retrieval/generation pass | Usually one invocation plus interpretation | Repeats until done |
| Can cause side effects? | Usually no | Possibly | Often, through tools |
| Main failure | Wrong/missing evidence | Wrong tool or arguments | Compounding planning/action errors |
| Main scaling issue | Corpus search and context budget | Large tool catalogs | Steps, cost, reliability, safety |

---

## Dense retrieval, BM25, and cross-encoder reranking

| Property | Dense bi-encoder | BM25 / lexical | Cross-encoder |
|---|---|---|---|
| Matches | Meaning | Exact terms | Joint query–chunk relevance |
| Precompute chunk representation | Yes | Index statistics | No query-independent final score |
| Corpus-scale search | Yes | Yes | Usually no |
| Query–chunk token interaction | No | No neural interaction | Yes |
| Typical stage | Candidate retrieval | Candidate retrieval | Reranking |
| Strength | Paraphrases | Names/IDs/rare words | Precision |
| Weakness | Can miss exact entities | Can miss semantic matches | Expensive |

---

## Candidate retrieval versus reranking

| | Candidate retrieval | Reranking |
|---|---|---|
| Input size | Entire corpus | `K_cand` candidates |
| Main objective | High recall | Good top ordering / precision |
| Model | ANN dense retriever, BM25, hybrid | Cross-encoder or stronger scorer |
| Computation | Cheap per candidate | Expensive per candidate |
| Error consequence | Miss cannot be recovered | Candidate may be ordered poorly |

---

## Precision@K, recall@K, MRR, and NDCG@K

| Metric | Main question | Cares about all relevant items? | Cares about rank order? |
|---|---|---:|---:|
| Precision@K | How clean is top K? | No | Not within top K |
| Recall@K | How much evidence did we recover? | Yes | Not within top K |
| MRR | How early is the first relevant item? | No | Yes, first only |
| NDCG@K | How good is the complete ranked top K? | Yes | Yes |

---

## Tool SFT versus prompt-only tool use

| | Tool-use SFT | Prompt / few-shot only |
|---|---|---|
| Weight update | Yes | No |
| Setup cost | Labelled training + compute | Prompt engineering/evaluation |
| Serving prompt | Can be shorter | Often longer |
| Adapt to new tools | Slower | Fast |
| Reliability | Potentially higher for stable distribution | Depends strongly on base model and prompt |
| Best fit | Stable, frequent, strict use | Rapidly changing or low-volume tools |

---

## MCP versus A2A

| | MCP | A2A |
|---|---|---|
| Connects | Model host/client to tools/context server | Agent to agent |
| Exposes | Tools, prompts, resources | Skills/capabilities and task execution interface |
| Main goal | Reusable tool/context interoperability | Agent communication and coordination |
| Is it an agent algorithm? | No | No |

---

# 17. Equations to remember

## 17.1 Cosine similarity

$$
\boxed{
\operatorname{cos}(q,c)
=
\frac{q^\top c}{\|q\|_2\|c\|_2}
}
$$

---

## 17.2 Batched dense-retrieval scores

$$
\boxed{
S=E_qE_c^\top
}
$$

```text
E_q: [B, D_e]
E_c: [M, D_e]
S:   [B, M]
```

---

## 17.3 Unit-normalized L2/dot-product relation

If `||q||=||c||=1`:

$$
\boxed{
\|q-c\|_2^2=2-2q^\top c
}
$$

---

## 17.4 Standard contrastive retrieval loss — INTERVIEW CLARIFICATION

$$
\boxed{
\mathcal L_i
=
-\log
\frac{\exp(s(q_i,c_i^+)/\tau)}
{\sum_j\exp(s(q_i,c_j)/\tau)}
}
$$

The lecture mentions contrastive training but does not derive this formula.

---

## 17.5 Hybrid retrieval score — one possible implementation

$$
\boxed{
s_{hybrid}
=
\alpha s_{dense}
+
(1-\alpha)s_{lexical}
}
$$

The lecture teaches the hybrid idea; exact score fusion is implementation-dependent.

---

## 17.6 Precision@K

$$
\boxed{
P@K
=
\frac{|\operatorname{TopK}(q)\cap R_q|}{K}
}
$$

---

## 17.7 Recall@K

$$
\boxed{
R@K
=
\frac{|\operatorname{TopK}(q)\cap R_q|}{|R_q|}
}
$$

---

## 17.8 Reciprocal rank and MRR

$$
\boxed{
RR(q)=\frac{1}{\operatorname{rank}_{first\ relevant}(q)}
}
$$

$$
\boxed{
MRR
=
\frac1N\sum_{i=1}^{N}RR(q_i)
}
$$

---

## 17.9 DCG@K

$$
\boxed{
DCG@K
=
\sum_{i=1}^{K}
\frac{2^{rel_i}-1}{\log_2(i+1)}
}
$$

---

## 17.10 NDCG@K

$$
\boxed{
NDCG@K
=
\frac{DCG@K}{IDCG@K}
}
$$

---

## 17.11 Prompt-length approximation

$$
\boxed{
T_{prompt}
\approx
T_{system}+T_q+K T_c+T_{formatting}
}
$$

This is an accounting identity, not a model law.

---

## 17.12 Tool-selection distribution — INTERVIEW CLARIFICATION

A selector may model:

$$
p(tool_j\mid q)
=
\operatorname{softmax}(z(q))_j
$$

or use retrieval similarity over tool descriptions. The lecture does not prescribe a specific selector loss.

---

# 18. Tensor and data-shape sheet

```text
DOCUMENT INDEXING
-----------------
M chunks, maximum T_c tokens each
chunk token IDs:             [M, T_c]
chunk embeddings:            [M, D_e]

QUERY ENCODING
--------------
query token IDs:             [B, T_q]
query embeddings:            [B, D_e]

DENSE RETRIEVAL
---------------
chunk embedding transpose:   [D_e, M]
all similarity scores:       [B, M]
first-stage top IDs:         [B, K_cand]
first-stage top scores:      [B, K_cand]

CROSS-ENCODER RERANKING
-----------------------
query–chunk pair IDs:        [B, K_cand, T_pair]
flattened pair IDs:          [B*K_cand, T_pair]
encoder states:              [B*K_cand, T_pair, D]
relevance score:             [B*K_cand]
reshaped scores:             [B, K_cand]
final chunk IDs:             [B, K]

AUGMENTED PROMPT
----------------
retrieved chunk IDs/text:    B groups of K chunks
augmented input IDs:         [B, T_prompt]
LLM logits:                  [B, T_output, V]

TOOL SFT
--------
serialized conversation IDs: [B, T]
attention mask:              [B, T]
labels:                      [B, T]
assistant/tool target mask:  [B, T]
logits:                      [B, T, V]

TOOL SELECTION VIA RETRIEVAL
----------------------------
query embeddings:            [B, D_e]
tool-description embeddings: [N_tools, D_e]
tool scores:                 [B, N_tools]
selected tool IDs:           [B, K_tools]

AGENT EXECUTION
---------------
not naturally one fixed tensor;
state is a structured trajectory of up to S steps:
    goal
    messages
    plans
    tool calls
    observations
    status
```

### Two shape derivations to say aloud

$$
[B,D_e][D_e,M]=[B,M]
$$

> Every query receives one score for every indexed chunk.

$$
[B,K_{cand},T_{pair}]
\rightarrow
[BK_{cand},T_{pair}]
\rightarrow
[BK_{cand}]
\rightarrow
[B,K_{cand}]
$$

> Flatten query–candidate pairs for one reranker batch, then restore the per-query grouping.

---

# 19. Complete implementation pseudocode

## 19.1 Offline indexing

```python
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    source_id: str
    text: str
    vector: torch.Tensor  # [D_e]


def build_index(documents, parser, chunker, encoder) -> list[IndexedChunk]:
    indexed: list[IndexedChunk] = []

    for document in documents:
        parsed = parser(document)
        chunks = chunker(parsed)

        texts = [chunk.text for chunk in chunks]
        vectors = encoder(texts)                 # [num_chunks, D_e]
        vectors = F.normalize(vectors, dim=-1)

        for chunk, vector in zip(chunks, vectors, strict=True):
            indexed.append(
                IndexedChunk(
                    chunk_id=chunk.id,
                    source_id=document.id,
                    text=chunk.text,
                    vector=vector.detach().cpu(),
                )
            )

    return indexed
```

In production, vectors would be inserted into an ANN/vector index rather than kept in a Python list.

---

## 19.2 Online hybrid retrieval and reranking

```python
def retrieve_and_rerank(
    query: str,
    dense_index,
    lexical_index,
    query_encoder,
    reranker,
    *,
    dense_k: int = 100,
    lexical_k: int = 100,
    final_k: int = 8,
):
    query_vector = query_encoder([query])        # [1, D_e]
    query_vector = F.normalize(query_vector, dim=-1)

    dense_hits = dense_index.search(query_vector, k=dense_k)
    lexical_hits = lexical_index.search(query, k=lexical_k)

    # Union/deduplicate candidate IDs. The fusion strategy is application-specific.
    candidates = fuse_candidates(dense_hits, lexical_hits)

    pair_texts = [(query, candidate.text) for candidate in candidates]
    rerank_scores = reranker(pair_texts)         # [num_candidates]

    ranked = sorted(
        zip(candidates, rerank_scores, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return ranked[:final_k]
```

---

## 19.3 RAG prompt construction

```python
def build_rag_prompt(query: str, ranked_chunks) -> str:
    context_blocks = []

    for rank, (chunk, score) in enumerate(ranked_chunks, start=1):
        context_blocks.append(
            f"[Source {rank}: {chunk.source_id}]\n{chunk.text}"
        )

    context = "\n\n".join(context_blocks)

    return f"""
Use the supplied sources to answer the user's question.
If the sources do not contain the answer, say that the evidence is insufficient.

SOURCES
{context}

QUESTION
{query}
""".strip()
```

The exact grounding instruction is not specified by the lecture; this shows the retrieve–augment–generate interface.

---

## 19.4 One tool-call turn

```python
def answer_with_tools(model, messages, tool_registry):
    tool_schemas = [tool.schema for tool in tool_registry.values()]

    decision = model.generate(messages, tools=tool_schemas)

    if decision.type == "final_answer":
        return decision.text

    if decision.type != "tool_call":
        raise RuntimeError(f"Unexpected model output: {decision.type}")

    tool = tool_registry.get(decision.tool_name)
    if tool is None:
        raise ValueError(f"Unknown tool: {decision.tool_name}")

    validated_args = tool.validate_arguments(decision.arguments)
    result = tool.execute(**validated_args)
    validated_result = tool.validate_result(result)

    messages = [
        *messages,
        decision.as_assistant_message(),
        {
            "role": "tool",
            "name": decision.tool_name,
            "content": validated_result,
        },
    ]

    final = model.generate(messages, tools=tool_schemas)
    return final.text
```

---

## 19.5 Tool selector

```python
def select_tools(query, compact_tool_catalog, selector, k_tools):
    # compact_tool_catalog contains only tool IDs + short descriptions.
    scores = selector(query, compact_tool_catalog)  # [N_tools]
    selected_ids = scores.topk(k_tools).indices
    return [compact_tool_catalog[i].tool_id for i in selected_ids]


def build_tool_context(query, selected_ids, full_registry):
    schemas = [full_registry[tool_id].schema for tool_id in selected_ids]
    return {"query": query, "tools": schemas}
```

---

## 19.6 Bounded ReAct-style agent

```python
from enum import Enum


class DecisionType(str, Enum):
    TOOL_CALL = "tool_call"
    FINISH = "finish"
    ASK_USER = "ask_user"


def run_bounded_agent(
    *,
    goal,
    model,
    tool_registry,
    max_steps=8,
    max_tool_calls=6,
):
    trajectory = [{"role": "user", "content": goal}]
    tool_calls = 0

    for step in range(max_steps):
        decision = model.decide(
            messages=trajectory,
            tools=[tool.schema for tool in tool_registry.values()],
        )
        trajectory.append(decision.as_message())

        if decision.type == DecisionType.FINISH:
            return {
                "status": "completed",
                "answer": decision.answer,
                "trajectory": trajectory,
            }

        if decision.type == DecisionType.ASK_USER:
            return {
                "status": "needs_input",
                "question": decision.question,
                "trajectory": trajectory,
            }

        if decision.type != DecisionType.TOOL_CALL:
            raise RuntimeError("Invalid agent decision")

        tool_calls += 1
        if tool_calls > max_tool_calls:
            return {
                "status": "budget_exhausted",
                "trajectory": trajectory,
            }

        tool = tool_registry[decision.tool_name]
        args = tool.validate_arguments(decision.arguments)

        if tool.requires_confirmation(args):
            return {
                "status": "needs_confirmation",
                "proposed_action": decision,
                "trajectory": trajectory,
            }

        observation = tool.execute(**args)
        trajectory.append(
            {
                "role": "tool",
                "name": decision.tool_name,
                "content": observation,
            }
        )

    return {
        "status": "max_steps_reached",
        "trajectory": trajectory,
    }
```

The confirmation and budget controls are standard engineering clarifications motivated by the lecture’s reliability and safety concerns.

---

# 20. Debugging matrix

| Symptom | Likely stage | Questions to ask |
|---|---|---|
| Correct document never appears | Ingestion/retrieval | Was it indexed? Did chunking preserve it? Was the query embedded correctly? |
| Correct chunk is in candidates but not final context | Reranking | Did reranker score it poorly? Is `K` too small? |
| Retrieved chunks are semantically related but miss exact entity | Dense retrieval | Add BM25/hybrid retrieval; inspect names and IDs |
| Too many near-duplicate chunks | Chunk overlap/fusion | Deduplicate by source/overlap; diversify final context |
| Tool not selected | Tool router | Was short description clear? Was recall high enough? |
| Correct tool, wrong arguments | Argument grounding | Is schema precise? Units/types clear? Missing context? |
| Tool succeeds, final response is wrong | Observation interpretation | Was result serialized correctly? Did full history reach the model? |
| Agent loops without progress | Termination/planning | Add max steps, progress checks, explicit finish criteria |
| Agent performs dangerous action | Permissions/safety | Least privilege, action confirmation, classifier/policy gate |
| Costs grow unexpectedly | Context/loop design | Prompt caching, smaller candidate/tool sets, token/step budgets |

---

# 21. High-value oral interview questions

Practise answering these without notes.

1. Why is a large context window not a replacement for retrieval?
2. Explain RAG in one minute.
3. What changes in RAG: weights or input context?
4. Why is chunk size a bias–variance-like trade-off?
5. Why use chunk overlap, and what does it cost?
6. What is the offline versus online split in a RAG system?
7. Derive the shape of a dense similarity matrix for `B` queries and `M` chunks.
8. Why does a bi-encoder support precomputation?
9. Why is a cross-encoder usually more accurate but less scalable?
10. What is approximate nearest-neighbour search solving?
11. When can dot product, cosine, and L2 produce the same ranking?
12. What does Sentence-BERT change relative to ordinary BERT usage?
13. What is the query–document distribution mismatch?
14. Explain HyDE and its failure modes.
15. Compare semantic retrieval with BM25.
16. Give a query where BM25 is likely to beat a pure dense retriever.
17. Why use hybrid retrieval?
18. What does contextual retrieval add to a chunk?
19. Why is prompt caching useful for contextual chunk generation?
20. Compare prompt caching with autoregressive KV caching.
21. What is candidate retrieval optimising?
22. Why can a reranker not recover a first-stage miss?
23. Define precision@K and recall@K.
24. When would you prefer recall@K over precision@K?
25. Define reciprocal rank and MRR.
26. Derive the intuition behind DCG.
27. Why divide DCG by IDCG?
28. NDCG versus MRR: what behaviour does each reward?
29. How would you debug a RAG answer that is factually wrong?
30. Compare RAG and fine-tuning.
31. Compare RAG and long-context prompting.
32. Compare RAG and a search tool call.
33. What exactly does an LLM see about a tool?
34. What part of tool execution is outside the LLM?
35. Walk through one complete tool-call turn.
36. Why return tool output to the model?
37. What two supervised mappings may be required for tool use?
38. How would you serialize tool-use SFT data?
39. What does argument grounding mean?
40. How do few-shot tool examples differ from tool-use SFT?
41. Explain the offline tool-instruction optimisation loop.
42. Why does a large tool catalog create a context problem?
43. How does a tool selector solve that problem?
44. Could tool selection be implemented as retrieval?
45. What is MCP trying to standardize?
46. Define MCP server, client, tool, prompt, and resource.
47. What is an agent beyond a tool-calling LLM?
48. Explain ReAct without using paper-specific notation.
49. Walk through observe–plan–act with the thermostat example.
50. What state must an agent retain?
51. What terminates an agent loop?
52. Why do agent errors compound?
53. When is a deterministic workflow better than an agent?
54. How can multiple agents coordinate?
55. What role does A2A play at a high level?
56. Why do tool-enabled models create new safety risks?
57. Explain data exfiltration in an agent context.
58. Compare training-time and inference-time safety controls.
59. Why is least privilege essential for agent tools?
60. Which actions should require human confirmation?
61. Why should every agent have token, step, and time budgets?
62. What should be logged for agent debugging?
63. Why start with the strongest available model during prototyping?
64. Why is code judgment more important as code generation gets cheaper?

---

# 22. Thirty-five memory liners

```text
1. Parametric knowledge is a frozen snapshot.

2. More context is not always more usable context.

3. RAG retrieves only the evidence likely to matter.

4. RAG changes context, not necessarily weights.

5. Retrieval failure cannot be repaired by a perfect reranker or generator.

6. Small chunks are precise but context-poor.

7. Large chunks are coherent but semantically diluted.

8. Overlap protects boundaries but duplicates content.

9. A retriever turns relevance into geometry.

10. Bi-encoder means encode separately and compare cheaply.

11. Cross-encoder means read query and chunk together before scoring.

12. Retrieve broadly first; judge carefully second.

13. Dense retrieval matches meaning; BM25 matches terms.

14. Hybrid retrieval combines semantic and lexical evidence.

15. HyDE turns a question into document-like retrieval language.

16. Contextual retrieval makes a local chunk globally interpretable.

17. Prompt caching reuses an invariant prefix across requests.

18. Precision controls context noise; recall controls missing evidence.

19. MRR rewards the first useful hit.

20. NDCG rewards the whole ranked list.

21. The model proposes a tool call; the runtime executes it.

22. Expose a tool contract, not its private implementation.

23. Tool output must return to the conversation before the final answer.

24. Train both tool invocation and result interpretation.

25. A large tool catalog is itself a retrieval problem.

26. Route tools from short descriptions, then load full schemas.

27. MCP standardizes model-host access to tools and context.

28. A tool is an action primitive; an agent is the controller.

29. ReAct means reason, act, observe, and repeat.

30. An agent is an LLM plus state, tools, a loop, and stop rules.

31. Every extra agent step is another opportunity for failure.

32. MCP connects hosts to tools; A2A connects agents to agents.

33. Once the model can act, mistakes become external side effects.

34. Autonomy should shrink as consequence grows.

35. Start correct, start small, then optimize.
```

---

# 23. Previous-day priority checklist

## Tier 1 — Must be immediate

- RAG = retrieve → augment → generate.
- Why long context alone is insufficient.
- Chunk size and overlap trade-offs.
- Bi-encoder versus cross-encoder.
- Dense versus BM25 versus hybrid retrieval.
- Query/chunk similarity tensor shapes.
- ANN motivation.
- HyDE and contextual retrieval.
- Precision@K, recall@K, MRR, DCG, and NDCG.
- Tool-call three-stage flow.
- LLM role versus runtime role.
- Tool-use SFT’s two supervised mappings.
- Tool selector/router.
- MCP vocabulary at a high level.
- Tool versus agent.
- Observe–plan–act / ReAct loop.
- Agent termination and bounded execution.
- Data-exfiltration risk and safety layers.

## Tier 2 — Derive on paper once

- `[B,D_e] @ [D_e,M] -> [B,M]`.
- Cross-encoder flatten/restore shape flow.
- cosine similarity.
- unit-normalized L2 relation.
- precision@K and recall@K.
- RR/MRR.
- DCG and NDCG.
- augmented-prompt token accounting.

## Tier 3 — Explain as system design

- How to build an offline indexing pipeline.
- How to choose candidate and final `K`.
- How to debug retrieval versus generation.
- How to add a new tool without immediately fine-tuning.
- How to scale from ten tools to ten thousand tools.
- How to add confirmation gates and least privilege.
- When a deterministic workflow is better than an agent.
- How to log and evaluate an agent trajectory.

---

# 24. Lecture boundaries

The following topics are mentioned, hinted at, or naturally adjacent, but are **not fully developed in this lecture**:

- exact BM25 formula;
- implementation details of a particular ANN index;
- exact Sentence-BERT or reranker training losses;
- hard-negative mining;
- end-to-end RAG faithfulness and answer-quality metrics;
- exact prompt-ordering and context-compression strategies;
- full MCP wire protocol, authentication, and transport specification;
- full A2A protocol specification;
- comprehensive prompt-injection defenses;
- formal agent planning algorithms;
- long-term agent memory architectures;
- production-grade distributed agent orchestration;
- detailed agent-evaluation methodology, which the lecture defers.

Do not attribute a specific solution for those topics to this transcript without an external source.

---

# 25. Final interview monologue

> A standalone LLM has static parametric knowledge, a finite and imperfectly usable context window, and no direct ability to execute actions. RAG addresses the knowledge problem by chunking and embedding an external corpus, retrieving a high-recall candidate set, optionally reranking it with a more precise cross-encoder, and augmenting the prompt with the final top chunks. Dense retrieval captures semantic similarity, BM25 captures exact lexical matches, and hybrid retrieval combines both. Retriever quality is measured with metrics such as precision@K, recall@K, MRR, and NDCG.
>
> Tool calling addresses structured information, computation, and action. The model sees a tool contract, predicts a tool name and structured arguments, but the application validates and executes the call. The result is placed back into the conversation so the model can produce a user-facing response. Tool-use behaviour can be learned through SFT or induced through schemas, examples, and optimized instructions. When the catalog is large, a tool selector first chooses a small subset of relevant tools. MCP provides a standard interface for exposing tools, prompts, and resources.
>
> An agent is one layer above tool use: it maintains state and iterates through observation, planning, action, and new observations until the goal is reached or execution is stopped. ReAct is the canonical reason–act–observe mental map. Multi-agent systems add communication and protocols such as A2A. This increased capability also creates compounding reliability and safety risks, so practical systems need bounded loops, validation, least-privilege permissions, logging, runtime safeguards, and human confirmation for consequential actions.
