# RAGAS — Module 9: Interpret Evaluation Scores and Improve RAG

## 1. Learning Objectives

By completing this module, you will understand:

- How to interpret RAGAS metrics together.
- How to identify retrieval and generation problems.
- How to select the correct improvement based on evaluation results.
- How to perform controlled experiments.
- How to compare baseline and improved RAG.
- How to avoid optimizing one metric while damaging another.
- How to create an evaluation improvement report.

---

## 2. Understand the Four Main Metrics

| Metric | What it measures | Higher is generally better |
|---|---|---|
| Faithfulness | Answer claims supported by retrieved context | Yes |
| Answer Relevancy | Answer relevance to the question | Yes |
| Context Precision | Relevant retrieved documents ranked effectively | Yes |
| Context Recall | Relevant information successfully retrieved | Yes |

**Important:** RAGAS scores are diagnostic signals, not absolute guarantees of application quality.

A score of 0.90 does not automatically mean that 90% of users will be satisfied.

---

## 3. Diagnose RAG Problems Using Metric Combinations

This is one of the most important concepts in RAG engineering.

### Scenario A: High Context Recall + Low Context Precision

Example:

| Metric | Score |
|---|---:|
| Faithfulness | 0.85 |
| Answer Relevancy | 0.90 |
| Context Precision | 0.40 |
| Context Recall | 0.95 |

Interpretation:

- Relevant information is being retrieved.
- Many unnecessary documents are also retrieved.
- The context contains excessive noise.

Possible improvements:

- Reduce top-k.
- Apply reranking.
- Improve embedding quality.
- Add metadata filters.
- Improve chunking.

### Scenario B: Low Context Recall + High Context Precision

Example:

| Metric | Score |
|---|---:|
| Faithfulness | 0.90 |
| Answer Relevancy | 0.75 |
| Context Precision | 0.90 |
| Context Recall | 0.45 |

Interpretation:

- Retrieved documents are mostly relevant.
- However, important information is missing.
- The retriever is too restrictive or incomplete.

Possible improvements:

- Increase top-k.
- Improve chunking.
- Add hybrid retrieval.
- Improve query rewriting.
- Improve document coverage.

### Scenario C: High Retrieval Scores + Low Faithfulness

Example:

| Metric | Score |
|---|---:|
| Faithfulness | 0.35 |
| Answer Relevancy | 0.85 |
| Context Precision | 0.90 |
| Context Recall | 0.95 |

Interpretation:

- Retrieval appears strong.
- The generated answer contains unsupported claims.

Possible improvements:

- Strengthen prompt instructions.
- Reduce unnecessary generation creativity.
- Use a stronger generation model.
- Add answer verification.
- Require answers to be grounded in context.

### Scenario D: High Faithfulness + Low Answer Relevancy

Example:

| Metric | Score |
|---|---:|
| Faithfulness | 0.95 |
| Answer Relevancy | 0.40 |
| Context Precision | 0.85 |
| Context Recall | 0.90 |

Interpretation:

- The answer is grounded in retrieved documents.
- But it does not properly address the question.

Possible improvements:

- Improve system prompt.
- Improve query interpretation.
- Remove unnecessary context.
- Improve answer formatting.
- Add question-specific instructions.

---

## 4. Build a Diagnostic Decision Table

| Faithfulness | Answer Relevancy | Context Precision | Context Recall | Likely problem |
|---|---|---|---|---|
| Low | High | High | High | Hallucination |
| High | Low | High | High | Poor answer formulation |
| High | High | Low | High | Retrieval noise |
| High | High | High | Low | Missing information |
| Low | Low | Low | Low | Multiple pipeline failures |

This table is a starting point, not a deterministic diagnosis.

Always inspect failed examples before deciding what to change.

---

## 5. Create a Baseline Evaluation

Before improving the RAG application, record its current performance.

Example baseline:

| Metric | Baseline |
|---|---:|
| Faithfulness | 0.72 |
| Answer Relevancy | 0.78 |
| Context Precision | 0.61 |
| Context Recall | 0.68 |

This baseline represents the current system.

Your objective is to improve the system without unnecessarily damaging other metrics.

### Baseline configuration

```python
baseline_config = {
    "embedding_model": "text-embedding-3-small",
    "chunk_size": 500,
    "chunk_overlap": 100,
    "top_k": 3,
    "reranking": False,
    "generation_model": "gpt-4o-mini"
}
```

These are illustrative configuration values.

---

## 6. Improve RAG Systematically

Do not change everything simultaneously.

Use the following process:

```text
Baseline Evaluation
        |
        v
Identify Weak Metric
        |
        v
Inspect Failed Examples
        |
        v
Identify Root Cause
        |
        v
Change One Component
        |
        v
Run Evaluation Again
        |
        v
Compare Scores
        |
        v
Accept or Reject Change
```

### Example

Suppose Context Recall is low.

You increase top-k from 3 to 5.

New results:

| Metric | Before | After |
|---|---:|---:|
| Faithfulness | 0.72 | 0.75 |
| Answer Relevancy | 0.78 | 0.80 |
| Context Precision | 0.61 | 0.52 |
| Context Recall | 0.68 | 0.87 |

Interpretation:

- Recall improved significantly.
- Precision decreased.
- More documents are retrieved.
- Additional documents may contain noise.

Decision:

Do not immediately accept the configuration. Investigate whether the extra documents improve answer quality and whether reranking can recover precision.

---

## 7. Experiment 1: Improve Retrieval

Create a simple experiment framework.

```python
experiments = [
    {
        "name": "baseline",
        "top_k": 3
    },
    {
        "name": "top_k_5",
        "top_k": 5
    },
    {
        "name": "top_k_10",
        "top_k": 10
    }
]
```

Execute each configuration:

```python
for experiment in experiments:

    print("Running:", experiment["name"])

    top_k = experiment["top_k"]

    # Execute your RAG application
    # using this top_k configuration.

    # Collect retrieved contexts.
    # Collect generated responses.
    # Run RAGAS evaluation.
    # Store scores.
```

The important part is that every experiment must use the same evaluation questions and reference answers.

### Expected learning

- Higher top-k can increase Context Recall.
- Higher top-k can reduce Context Precision.
- The optimal top-k depends on your dataset and retrieval system.

---

## 8. Experiment 2: Improve Chunking

**Chunking:** Dividing documents into smaller pieces before embedding and retrieval.

Example document:

```text
FastAPI is a Python framework for building APIs.
It supports asynchronous programming.
It provides automatic API documentation.
It uses Python type hints.
```

### Small chunks

```text
Chunk 1:
FastAPI is a Python framework for building APIs.

Chunk 2:
It supports asynchronous programming.

Chunk 3:
It provides automatic API documentation.
```

### Large chunk

```text
FastAPI is a Python framework for building APIs.
It supports asynchronous programming.
It provides automatic API documentation.
It uses Python type hints.
```

### Trade-offs

| Small chunks | Large chunks |
|---|---|
| More precise retrieval | More surrounding context |
| Less irrelevant information | More potentially irrelevant information |
| Risk losing surrounding meaning | Risk retrieving unnecessary content |
| More document fragments | Fewer document fragments |

### Experiment

Test:

- 200-token chunks
- 500-token chunks
- 1000-token chunks

Keep overlap and other variables controlled.

Evaluate each configuration using the same dataset.

---

## 9. Experiment 3: Improve Generation

Suppose Faithfulness is low.

Your current prompt:

```python
prompt = f"""
Answer the question using the context.

Context:
{context}

Question:
{question}
"""
```

Improved prompt:

```python
prompt = f"""
You are an assistant answering questions
from a knowledge base.

Rules:
1. Use only the supplied context.
2. Do not invent facts.
3. If the context is insufficient, clearly say so.
4. Answer the question directly.
5. Do not include unsupported assumptions.

Context:
{context}

Question:
{question}

Answer:
"""
```

### Evaluate the change

Compare:

- Faithfulness
- Answer Relevancy
- Answer correctness
- Answer length
- Refusal behavior when context is insufficient

A better prompt should improve grounding without making answers unnecessarily vague.

---

## 10. Experiment 4: Improve Retrieval with Reranking

**Reranking:** Reordering retrieved documents using a second relevance model after initial retrieval.

Pipeline:

```text
User Question
     |
     v
Vector Search
     |
     v
Retrieve 20 Documents
     |
     v
Reranker
     |
     v
Select Best 5 Documents
     |
     v
LLM Generation
```

Why use reranking?

Vector search is efficient for candidate retrieval.

Reranking can improve the ordering and relevance of those candidates.

### Example

Initial retrieval:

| Rank | Document | Relevance |
|---|---|---|
| 1 | General Python information | Low |
| 2 | FastAPI API documentation | High |
| 3 | LangGraph documentation | Low |
| 4 | FastAPI async guide | High |

After reranking:

| Rank | Document | Relevance |
|---|---|---|
| 1 | FastAPI API documentation | High |
| 2 | FastAPI async guide | High |
| 3 | General Python information | Low |
| 4 | LangGraph documentation | Low |

Expected improvement:

- Context Precision
- Potentially Faithfulness
- Potentially Answer Relevancy

The actual improvement must be measured.

---

## 11. Compare Baseline and Improved RAG

Example experiment report:

| Metric | Baseline | Improved | Difference |
|---|---:|---:|---:|
| Faithfulness | 0.72 | 0.91 | +0.19 |
| Answer Relevancy | 0.78 | 0.89 | +0.11 |
| Context Precision | 0.61 | 0.82 | +0.21 |
| Context Recall | 0.68 | 0.86 | +0.18 |

### Calculate improvements using Python

```python
baseline = {
    "faithfulness": 0.72,
    "answer_relevancy": 0.78,
    "context_precision": 0.61,
    "context_recall": 0.68
}

improved = {
    "faithfulness": 0.91,
    "answer_relevancy": 0.89,
    "context_precision": 0.82,
    "context_recall": 0.86
}

for metric in baseline:

    difference = (
        improved[metric] - baseline[metric]
    )

    print(
        metric,
        "improvement:",
        round(difference, 3)
    )
```

Output:

```text
faithfulness improvement: 0.19
answer_relevancy improvement: 0.11
context_precision improvement: 0.21
context_recall improvement: 0.18
```

These are example numbers, not measured results.

---

## 12. Avoid Metric Optimization Mistakes

### Mistake 1: Optimizing only Faithfulness

A system can be highly faithful but still answer incorrectly because the retriever found the wrong information.

Faithfulness measures grounding, not complete factual correctness.

### Mistake 2: Optimizing only Context Recall

Retrieving every document may increase recall but produce excessive context noise.

### Mistake 3: Optimizing only Answer Relevancy

A concise answer can be relevant but factually incorrect.

### Mistake 4: Changing multiple components together

If you simultaneously change chunking, embeddings, top-k, and prompts, you cannot determine which change caused the improvement.

### Mistake 5: Evaluating on only a few questions

A small dataset may not represent real user behavior.

Use a representative dataset and inspect individual failures.

---

## 13. Create a RAG Improvement Report

A practical engineering report should contain:

```text
Experiment ID:
Date:
Dataset Version:
Application Version:

Baseline Configuration:
Embedding Model:
Chunk Size:
Chunk Overlap:
Top-K:
Reranking:
Generation Model:
Prompt Version:

Baseline Metrics:
Faithfulness:
Answer Relevancy:
Context Precision:
Context Recall:

Changes Introduced:

New Metrics:

Observed Improvements:

Observed Regressions:

Failure Analysis:

Decision:
ACCEPT / REJECT / INVESTIGATE

Next Experiment:
```

This report helps maintain reproducibility.

---

## 14. Interview Questions and Answers

### Q1. How do you improve a RAG application using RAGAS?

Identify weak metrics, inspect failed samples, determine the root cause, change one component, rerun evaluation, and compare against a fixed baseline.

### Q2. Can Context Recall improve while Context Precision decreases?

Yes. Retrieving more documents can capture additional relevant information while also introducing irrelevant documents.

### Q3. What is the difference between Faithfulness and Correctness?

Faithfulness measures whether an answer is supported by retrieved context.

Correctness measures whether the answer is actually accurate relative to the expected answer or real-world facts.

An answer may be faithful to an incorrect source.

### Q4. Why should we use a fixed evaluation dataset?

It allows controlled comparison between system configurations and helps identify whether improvements are genuine or caused by changing evaluation questions.

### Q5. What is reranking?

Reranking is a second-stage retrieval process that reorders candidate documents based on relevance to the query.

### Q6. How do you select the best RAG configuration?

Evaluate multiple configurations, compare quality metrics, inspect failures, and consider latency, cost, and user requirements.

### Q7. Is a higher RAGAS score always better?

Not necessarily. Scores must be interpreted with application requirements, evaluation reliability, and other metrics.

### Q8. What is regression testing in RAG?

Regression testing checks whether a new change damages previously working behavior or evaluation performance.

---

## 15. Hands-on Assignment

Use the RAG application created in Module 8.

### Tasks

1. Run baseline evaluation.
2. Save baseline metrics.
3. Identify the weakest metric.
4. Inspect at least five failed samples.
5. Change top-k.
6. Evaluate again.
7. Change chunk size.
8. Evaluate again.
9. Improve generation prompt.
10. Evaluate again.
11. Compare all configurations.
12. Select the best configuration.
13. Write a RAG improvement report.
14. Explain why the selected configuration is better.

### Deliverable

Create:

```text
rag_evaluation/
    dataset.json
    baseline.json
    experiments.json
    improvement_report.md
```

---

## 16. Module 9 Completion Checklist

- [ ] Understand all four metrics.
- [ ] Diagnose retrieval failures.
- [ ] Diagnose generation failures.
- [ ] Create a baseline.
- [ ] Run controlled experiments.
- [ ] Improve chunking.
- [ ] Improve top-k.
- [ ] Improve prompts.
- [ ] Compare baseline and improved scores.
- [ ] Create an improvement report.

---

## 17. RAGAS Learning Progress

**9 / 10 modules completed — 90%.**

| Module | Topic | Status |
|---|---|---|
| 1 | RAGAS Fundamentals | Completed |
| 2 | Faithfulness | Completed |
| 3 | Answer Relevancy | Completed |
| 4 | Context Precision | Completed |
| 5 | Context Recall | Completed |
| 6 | Evaluation Dataset | Completed |
| 7 | Python Evaluation | Completed |
| 8 | Evaluate Real RAG Application | Completed |
| 9 | Interpret Scores and Improve RAG | Completed |
| 10 | CI/CD and Production Monitoring | Next |

## Next Module: Module 10 — Integrate RAGAS into CI/CD and Production Monitoring

You will learn:

- Automated RAG evaluation.
- GitHub Actions integration.
- Evaluation regression checks.
- Quality thresholds.
- Production monitoring.
- Tracking RAG quality over time.
- Logging failed answers.
- Monitoring evaluation costs.
- Building a complete RAG evaluation workflow.

**After Module 10: 100% RAGAS roadmap completion.**