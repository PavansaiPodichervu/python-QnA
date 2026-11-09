# Module 5 — Context Recall: Evaluate Retrieval Completeness

## Learning Objectives

By the end of this module, you will understand:

1. What Context Recall is in RAG evaluation.
2. Context Precision vs Context Recall.
3. The importance of reference answers and ground truth.
4. Required information vs retrieved information.
5. Context Recall calculation.
6. High and low Context Recall examples.
7. Python implementation.
8. RAGAS Context Recall implementation.
9. Improving retrieval completeness.
10. Precision-Recall trade-offs in RAG systems.

---

# 1. What Is Context Recall?

## Definition

**Context Recall measures how much of the information required to answer a question is available in the retrieved context.**

In a RAG application, the retriever searches a knowledge base and returns documents.

However, the retrieved documents may contain only a portion of the information required to answer the question.

Context Recall evaluates whether the retriever found the necessary information.

### Example

Question:

> What are the eligibility criteria, benefits, and application deadline for the employee scholarship?

Expected answer:

- Employees must have completed two years of service.
- Employees receive a scholarship of ₹50,000.
- Applications must be submitted before December 31.

Retrieved context:

> Employees must have completed two years of service to qualify for the scholarship.

The retrieved context contains only the eligibility criteria.

It misses:

- Scholarship amount
- Application deadline

Therefore, Context Recall is low.

---

# 2. Context Recall vs Context Precision

These metrics evaluate different retrieval properties.

| Metric | Meaning |
|---|---|
| Context Precision | Are retrieved documents relevant? |
| Context Recall | Did retrieval find the required information? |

## Example A — High Precision, Low Recall

Question:

> What are the eligibility criteria and benefits of the scholarship?

Retrieved context:

> Employees must have completed two years of service.

This document is relevant, but the scholarship benefits are missing.

Result:

- High Context Precision
- Low Context Recall

## Example B — Low Precision, High Recall

Question:

> What are the eligibility criteria and benefits of the scholarship?

Retrieved contexts:

1. Employee salary policy
2. Scholarship eligibility criteria
3. Company attendance policy
4. Scholarship benefit amount
5. Employee promotion policy

The relevant information is available, but irrelevant documents are also retrieved.

Result:

- Potentially high Context Recall
- Lower Context Precision

## Example C — High Precision, High Recall

Retrieved contexts:

1. Scholarship eligibility criteria
2. Scholarship amount
3. Application deadline

All required information is retrieved, with little irrelevant information.

Result:

- High Context Precision
- High Context Recall

---

# 3. Ground Truth and Reference Answers

Context Recall typically requires a reference answer or ground truth containing the expected information.

## Definition: Ground Truth

Ground truth is the expected correct information used to evaluate an AI system.

In RAG evaluation, a reference answer can represent the information that must be retrieved and included in the generated answer.

### Example

Question:

> What is the company's work-from-home policy?

Reference answer:

> Employees can work remotely two days per week with manager approval.

This reference contains two important facts:

1. Employees can work remotely two days per week.
2. Manager approval is required.

Retrieved context:

> Employees can work remotely two days per week.

The retrieved context contains the first fact but misses the approval requirement.

Context Recall should reflect this missing information.

### Important distinction

A reference answer is not necessarily the same as a reference document.

- Reference answer: expected answer to the question.
- Reference document: original source containing the required information.
- Retrieved context: documents returned by your retrieval system.

---

# 4. Required Information vs Retrieved Information

To understand Context Recall, first identify the information required to answer the question.

## Example

Question:

> What is the refund period, refund amount, and refund processing time?

Reference answer:

> Customers can request refunds within 30 days. Eligible customers receive a full refund, processed within five business days.

Break the reference into claims:

| Claim | Required Information |
|---|---|
| Claim 1 | Refund period is 30 days |
| Claim 2 | Eligible customers receive a full refund |
| Claim 3 | Refund processing takes five business days |

Retrieved context:

> Customers can request refunds within 30 days. Refund requests are processed within five business days.

Retrieved claims:

| Claim | Available in Context |
|---|---|
| Claim 1 | Yes |
| Claim 2 | No |
| Claim 3 | Yes |

Two of the three required claims are supported by retrieved context.

Therefore, simplified Context Recall:

\[
Context\ Recall = \frac{2}{3}
\]

\[
Context\ Recall = 0.667
\]

This means approximately 66.7% of the required information was retrieved.

---

# 5. Context Recall Calculation

A common conceptual calculation is:

\[
Context\ Recall =
\frac{\text{Reference claims supported by retrieved context}}
{\text{Total reference claims}}
\]

For example:

Reference claims:

```python
reference_claims = [
    "Refund period is 30 days",
    "Eligible customers receive a full refund",
    "Refund processing takes five business days"
]
```

Retrieved context supports:

```python
supported_claims = [
    "Refund period is 30 days",
    "Refund processing takes five business days"
]
```

Calculation:

\[
Context\ Recall = \frac{2}{3}=0.667
\]

## Interpretation

| Score | Interpretation |
|---|---|
| 1.00 | All required information is retrieved |
| 0.80 | Most required information is retrieved |
| 0.50 | Approximately half the required information is retrieved |
| 0.20 | Most required information is missing |
| 0.00 | No reference information is supported |

These are illustrative interpretations, not universal quality thresholds.

---

# 6. Context Recall Example — High Score

Question:

> What are the eligibility requirements and benefits of the scholarship?

Reference:

> Employees must complete two years of service. Eligible employees receive ₹50,000.

Retrieved contexts:

1. Employees must complete two years of service.
2. Eligible employees receive a scholarship of ₹50,000.

All reference claims are supported.

\[
Context\ Recall = \frac{2}{2}=1.0
\]

<text>**Context Recall = 1.0**</text>

The retrieval system found all required information.

---

# 7. Context Recall Example — Low Score

Question:

> What are the eligibility requirements and benefits of the scholarship?

Reference:

> Employees must complete two years of service. Eligible employees receive ₹50,000.

Retrieved context:

> Employees must complete two years of service to qualify.

Only eligibility information is retrieved.

\[
Context\ Recall = \frac{1}{2}=0.5
\]

<text>**Context Recall = 0.5**</text>

The retrieval system missed the scholarship benefit.

---

# 8. Python Implementation — Manual Context Recall

We will implement a simplified Context Recall calculation using manually identified reference claims.

## Example Dataset

```python
reference_claims = [
    "Employees must complete two years of service",
    "Employees receive a scholarship of 50000",
    "Applications close on December 31"
]

retrieved_context = """
Employees must complete two years of service.
Applications close on December 31.
"""
```

## Implementation

```python
def calculate_context_recall(
    reference_claims,
    supported_claims
):
    total_claims = len(reference_claims)

    if total_claims == 0:
        return 0.0

    supported_count = len(supported_claims)

    return supported_count / total_claims


reference_claims = [
    "Eligibility requires two years of service",
    "Scholarship amount is 50000",
    "Application deadline is December 31"
]

supported_claims = [
    "Eligibility requires two years of service",
    "Application deadline is December 31"
]

score = calculate_context_recall(
    reference_claims,
    supported_claims
)

print("Context Recall:", round(score, 3))
```

Output:

```text
Context Recall: 0.667
```

## Code Explanation

### Step 1

```python
total_claims = len(reference_claims)
```

Counts the required claims.

### Step 2

```python
supported_count = len(supported_claims)
```

Counts claims supported by retrieved context.

### Step 3

```python
return supported_count / total_claims
```

Calculates the percentage of required information retrieved.

### Important limitation

This manual implementation assumes that claim support has already been identified correctly.

It does not automatically determine whether retrieved context semantically supports a claim.

In a real evaluation system, claim extraction and claim verification may require an LLM or another semantic evaluation method.

---

# 9. RAGAS Implementation — Context Recall

## Installation

```bash
pip install ragas openai
```

Set your API key:

```bash
export OPENAI_API_KEY="your_api_key"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

## Dataset Structure

Context Recall typically uses:

| Field | Purpose |
|---|---|
| user_input | Original question |
| retrieved_contexts | Retrieved documents |
| reference | Expected answer |

## Example Using the Collections API

```python
import asyncio
import os

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextRecall


async def main():

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"]
    )

    evaluator_llm = llm_factory(
        "gpt-4o-mini",
        client=client
    )

    metric = ContextRecall(
        llm=evaluator_llm
    )

    question = (
        "What are the eligibility requirements "
        "and benefits of the scholarship?"
    )

    reference = (
        "Employees must complete two years of service. "
        "Eligible employees receive a scholarship of ₹50,000."
    )

    retrieved_contexts = [
        "Employees must complete two years of service.",
        "The company provides several employee benefits."
    ]

    result = await metric.ascore(
        user_input=question,
        reference=reference,
        retrieved_contexts=retrieved_contexts
    )

    print("Context Recall:", result.value)


if __name__ == "__main__":
    asyncio.run(main())
```

### Expected Interpretation

The retrieved context contains eligibility information but does not explicitly contain the scholarship amount.

Therefore, the evaluator should identify missing reference information and assign a score below 1.0.

The exact score depends on the evaluator's claim extraction and verification judgments.

### Important

RAGAS APIs evolve across versions. Verify the installed version and official documentation if imports or parameters differ.

Official documentation:

https://docs.ragas.io/

---

# 10. Context Recall in a Real RAG Application

Consider the following RAG pipeline:

```text
User Question
     |
     v
Query Embedding
     |
     v
Vector Database
     |
     v
Retrieve Top-K Documents
     |
     v
Retrieved Contexts
     |
     v
LLM Answer Generation
     |
     v
Final Answer
```

Context Recall evaluates whether the retrieved contexts contain the information needed to answer the question.

## Example

```python
question = (
    "What is the refund period and processing time?"
)

documents = retriever.invoke(question)

retrieved_contexts = [
    document.page_content
    for document in documents
]

answer = llm.invoke(
    f"""
    Answer the question using the provided context.

    Question:
    {question}

    Context:
    {retrieved_contexts}
    """
)
```

Evaluation record:

```python
evaluation_record = {
    "user_input": question,
    "retrieved_contexts": retrieved_contexts,
    "response": answer.content,
    "reference": (
        "Customers can request refunds within 30 days. "
        "Refunds are processed within five business days."
    )
}
```

This record can be evaluated using Context Recall.

---

# 11. Improving Context Recall

Low Context Recall indicates that important information is missing from the retrieved documents.

## Improvement 1 — Increase Top-K

Example:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 10}
)
```

Increasing top-K may retrieve additional relevant documents.

However, it may also introduce irrelevant context.

Evaluate Context Precision alongside Context Recall.

## Improvement 2 — Improve Chunking

Suppose a document contains:

```text
Scholarship eligibility:
Employees must complete two years of service.

Scholarship amount:
Eligible employees receive ₹50,000.

Application deadline:
Applications close on December 31.
```

Poor chunking may separate these facts incorrectly.

Better chunking preserves meaningful sections and their relationships.

## Improvement 3 — Use Hybrid Search

Hybrid search combines:

- Vector similarity
- Keyword matching

This can improve retrieval when exact terminology matters.

Example:

```text
Question:
What is the policy for employee parental leave?
```

Keyword search may retrieve documents containing the exact phrase "parental leave".

Vector search may retrieve documents containing "new parent benefits".

Combining both can improve retrieval completeness.

## Improvement 4 — Use Query Expansion

Query expansion generates alternative search expressions.

Original question:

```text
How many vacation days do employees receive?
```

Expanded queries:

```text
Annual leave allowance
Employee vacation entitlement
Paid leave policy
```

This may help retrieve documents using different terminology.

## Improvement 5 — Use Reranking

Reranking can improve document ordering.

However, reranking primarily changes ranking and does not guarantee that missing documents will be retrieved.

If a relevant document was never retrieved, reranking cannot recover it.

## Improvement 6 — Improve Metadata Filtering

Example:

```python
filter = {
    "department": "HR",
    "document_type": "policy"
}
```

Metadata filtering can reduce search scope and help retrieve the appropriate documents.

## Improvement 7 — Retrieve Multiple Document Types

Some questions require information from multiple sources.

Example:

```text
Question:
What is the product price and warranty period?
```

Required sources:

- Product catalog
- Warranty policy

Retrieving only the product catalog may produce incomplete context.

---

# 12. Context Precision and Context Recall Together

Both metrics should be evaluated together.

| Context Precision | Context Recall | Interpretation |
|---|---|---|
| High | High | Relevant and complete retrieval |
| High | Low | Relevant documents, but missing information |
| Low | High | Required information found, but noisy retrieval |
| Low | Low | Poor retrieval quality |

## Practical Example

Question:

> What is the refund period and processing time?

Reference:

> Refunds can be requested within 30 days and are processed within five business days.

### System A

Retrieved context:

> Refunds can be requested within 30 days.

- Context Precision: potentially high
- Context Recall: low

### System B

Retrieved context:

> Refunds can be requested within 30 days. Processing takes five business days. Company revenue increased by 10%.

- Context Recall: potentially high
- Context Precision: lower because of irrelevant context

### System C

Retrieved context:

> Refunds can be requested within 30 days. Refunds are processed within five business days.

- Context Precision: high
- Context Recall: high

System C is preferable for this question.

---

# 13. Common Context Recall Mistakes

## Mistake 1 — Confusing Context Recall With Answer Recall

Context Recall evaluates retrieved context against reference information.

It does not directly evaluate whether the generated answer contains every required fact.

## Mistake 2 — Assuming High Recall Means High Precision

A retriever may retrieve all required information along with many irrelevant documents.

## Mistake 3 — Using Poor Reference Answers

Incomplete or incorrect reference answers can distort evaluation.

## Mistake 4 — Ignoring Multi-Part Questions

Questions containing multiple requirements need references that capture all relevant claims.

## Mistake 5 — Increasing Top-K Without Evaluation

Increasing top-K may improve recall but decrease precision and increase token usage.

## Mistake 6 — Assuming Reranking Fixes Missing Information

Reranking cannot recover documents that were never retrieved.

---

# 14. Interview Questions and Answers

### Q1. What is Context Recall?

Context Recall measures the proportion of reference information supported by retrieved context.

### Q2. Why is Context Recall important?

It helps identify whether the retrieval system misses important information required to answer questions.

### Q3. What is the difference between Context Precision and Context Recall?

Context Precision evaluates retrieved relevance and ranking. Context Recall evaluates retrieval completeness.

### Q4. Can Context Recall be high while Context Precision is low?

Yes. All required information may be retrieved along with many irrelevant documents.

### Q5. What is ground truth?

Ground truth is the expected correct information used to evaluate an AI system.

### Q6. Does Context Recall evaluate answer correctness?

No. It evaluates whether retrieved context supports the reference answer.

### Q7. How can Context Recall be improved?

Improve chunking, query expansion, embeddings, hybrid search, top-K selection and retrieval coverage.

### Q8. Can reranking improve Context Recall?

Reranking may improve retrieval ordering, but it cannot recover documents that were never retrieved.

### Q9. Why evaluate Context Recall and Context Precision together?

Because retrieval must be both sufficiently complete and relevant.

### Q10. What happens when Context Recall is low?

The generator may not have enough evidence to produce a complete answer.

---

# 15. Practical Exercises

## Exercise 1 — Calculate Context Recall

Reference claims:

```python
reference_claims = [
    "Refund period is 30 days",
    "Refund processing takes five business days",
    "Eligible customers receive a full refund",
    "Refund requests require an order number"
]
```

Retrieved context supports:

```python
supported_claims = [
    "Refund period is 30 days",
    "Refund processing takes five business days",
    "Refund requests require an order number"
]
```

Tasks:

1. Count reference claims.
2. Count supported claims.
3. Calculate Context Recall.
4. Identify missing information.

Expected result:

```text
Context Recall = 0.75
```

## Exercise 2 — Identify Retrieval Problems

Question:

> What are the employee eligibility criteria and scholarship amount?

Reference:

> Employees must complete two years of service and receive ₹50,000.

Retrieved context:

> Employees must complete two years of service.

Tasks:

- Identify retrieved information.
- Identify missing information.
- Calculate simplified Context Recall.
- Suggest three retrieval improvements.

## Exercise 3 — Precision vs Recall

Create three retrieval results:

1. High Precision / Low Recall
2. Low Precision / High Recall
3. High Precision / High Recall

Explain why each system produces its scores.

## Exercise 4 — Build a Dataset

Create 10 questions containing multiple facts.

For each question, store:

- Question
- Reference answer
- Retrieved contexts
- Generated answer

Evaluate Context Recall using RAGAS.

---

# 16. Module Summary

You learned:

- Context Recall measures retrieval completeness.
- Reference answers provide expected information.
- Reference claims can be compared against retrieved context.
- Missing claims reduce Context Recall.
- Context Recall differs from Context Precision.
- Increasing top-K can improve recall but introduce noise.
- Reranking improves ordering but cannot recover missing documents.
- RAGAS can automate Context Recall evaluation.
- Context Precision and Context Recall should be evaluated together.

## Completion Checklist

- [ ] I understand Context Recall.
- [ ] I understand ground truth and reference answers.
- [ ] I can identify required claims.
- [ ] I can calculate simplified Context Recall.
- [ ] I understand Context Precision vs Context Recall.
- [ ] I can implement manual Context Recall in Python.
- [ ] I understand RAGAS Context Recall.
- [ ] I can identify retrieval completeness problems.
- [ ] I understand how to improve retrieval recall.
- [ ] I can evaluate retrieval using Precision and Recall together.

---

# 17. Next Module

## Module 6 — Build a RAG Evaluation Dataset

Topics:

1. What is a RAG evaluation dataset?
2. Dataset schema and required fields
3. Questions, references and retrieved contexts
4. Synthetic dataset generation
5. Creating evaluation questions from documents
6. Ground truth generation
7. Dataset quality validation
8. Python implementation
9. RAGAS dataset preparation
10. Building a reusable evaluation dataset for production RAG applications

