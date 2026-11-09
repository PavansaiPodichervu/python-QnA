# Module 4 — Context Precision: Evaluate Retrieval Ranking

## Learning Objectives

By the end of this module, you will understand:

1. What Context Precision is in RAG evaluation.
2. Relevant vs irrelevant retrieved documents.
3. Why retrieval ranking matters.
4. Context Precision at different retrieval positions.
5. How Context Precision is calculated.
6. Context Precision with and without reference answers.
7. High and low Context Precision examples.
8. How to implement Context Precision using Python.
9. How to evaluate Context Precision using RAGAS.
10. How to improve retrieval ranking in a real RAG application.

---

# 1. What Is Context Precision?

## Definition

**Context Precision measures how effectively a retrieval system ranks relevant information above irrelevant information in the retrieved context.**

In a RAG application, the retriever searches a knowledge base and returns documents relevant to the user's query.

However, retrieval systems may return:

- Relevant documents
- Partially relevant documents
- Irrelevant documents
- Documents containing similar keywords but incorrect information

Context Precision evaluates whether the relevant documents are ranked appropriately.

### Example

User question:

> What is the company's annual vacation policy?

Retrieved documents:

| Rank | Retrieved Document | Relevance |
|---|---|---|
| 1 | Employees receive 20 vacation days annually | Relevant |
| 2 | Employees can request additional vacation | Relevant |
| 3 | Employee salary structure | Irrelevant |
| 4 | Employees receive 10 sick leave days | Irrelevant |

The first two documents are relevant, while the remaining documents are not.

The retrieval system successfully placed relevant information at the top.

This represents good Context Precision.

---

# 2. Context Precision vs Context Recall

These two metrics evaluate different aspects of retrieval.

| Metric | Meaning |
|---|---|
| Context Precision | Are retrieved documents relevant? |
| Context Recall | Did retrieval find all relevant information? |

### Example

Suppose the knowledge base contains four relevant documents.

The retriever returns only two relevant documents.

- Context Precision may be high because everything retrieved is relevant.
- Context Recall is low because two relevant documents were missed.

### Important distinction

**Context Precision evaluates retrieval quality among retrieved results.**

**Context Recall evaluates how much of the required information was retrieved.**

We will study Context Recall in Module 5.

---

# 3. Relevant vs Irrelevant Retrieved Documents

A document is considered relevant when it contributes useful information toward answering the user's question.

## Relevant Context

Question:

> What is the refund policy?

Context:

> Customers can request a refund within 30 days of purchase.

This context directly helps answer the question.

## Irrelevant Context

Question:

> What is the refund policy?

Context:

> The company has 500 employees across five offices.

This context does not help answer the refund question.

## Partially Relevant Context

Question:

> What is the refund policy?

Context:

> Customers can contact customer support for assistance with purchases.

This may be useful indirectly, but it does not contain the refund policy itself.

### Practical rule

A retrieved document should be evaluated based on its usefulness for the specific question, not merely whether it contains similar keywords.

---

# 4. Why Retrieval Ranking Matters

Most RAG applications retrieve a limited number of documents using a parameter such as `top_k`.

Example:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
```

This retrieves five documents.

But imagine the following ranking:

| Rank | Document | Relevance |
|---|---|---|
| 1 | Employee salary information | Irrelevant |
| 2 | Employee attendance policy | Irrelevant |
| 3 | Annual vacation policy | Relevant |
| 4 | Vacation approval process | Relevant |
| 5 | Vacation eligibility | Relevant |

The answer may still be generated correctly, but irrelevant documents occupy the top positions.

This can cause:

- Increased context noise
- Higher token consumption
- Confusion during answer generation
- Reduced answer quality
- Increased hallucination risk

### Better ranking

| Rank | Document | Relevance |
|---|---|---|
| 1 | Annual vacation policy | Relevant |
| 2 | Vacation approval process | Relevant |
| 3 | Vacation eligibility | Relevant |
| 4 | Employee salary information | Irrelevant |
| 5 | Employee attendance policy | Irrelevant |

The second ranking is better because useful information appears earlier.

---

# 5. Context Precision Calculation

Context Precision considers the relevance of retrieved documents and their positions in the ranking.

A common conceptual formula is:

\[
CP@K =
\frac{\sum_{k=1}^{K} Precision@k \times rel_k}
{\text{Number of relevant retrieved documents}}
\]

Where:

- \(K\) = number of retrieved documents
- \(rel_k\) = relevance of document at position \(k\)
- \(Precision@k\) = precision among the first \(k\) documents

For binary relevance:

- Relevant document = 1
- Irrelevant document = 0

Precision at position \(k\):

\[
Precision@k =
\frac{\text{Relevant documents in top }k}
{k}
\]

## Example

Question:

> What is the annual vacation allowance?

Retrieved documents:

| Rank | Document | Relevance |
|---|---|---|
| 1 | Vacation policy | 1 |
| 2 | Salary policy | 0 |
| 3 | Vacation eligibility | 1 |
| 4 | Sick leave policy | 0 |

Let's calculate.

### Step 1: Precision at each relevant position

At rank 1:

\[
Precision@1 = \frac{1}{1}=1
\]

At rank 3:

\[
Precision@3 = \frac{2}{3}=0.667
\]

### Step 2: Multiply by relevance

| Rank | Relevance | Precision@k | Contribution |
|---|---:|---:|---:|
| 1 | 1 | 1.000 | 1.000 |
| 2 | 0 | 0.500 | 0 |
| 3 | 1 | 0.667 | 0.667 |
| 4 | 0 | 0.500 | 0 |

### Step 3: Divide by relevant retrieved documents

There are two relevant documents.

\[
CP@4 = \frac{1+0.667}{2}
\]

\[
CP@4 = 0.833
\]

<text>**Conceptual Context Precision = 0.833**</text>

This indicates that relevant documents are generally ranked near the top.

> Important: This is a simplified ranking-based illustration. RAGAS implementations may use reference-based or LLM-based relevance judgments, and their exact calculation depends on the metric variant.

---

# 6. Context Precision With Reference Answers

Context Precision can be evaluated using a reference answer.

A reference answer is the expected correct answer for a question.

Example:

**Question:**

What is the company's annual vacation allowance?

**Reference Answer:**

Employees receive 20 annual vacation days.

**Retrieved Contexts:**

1. Employees receive 20 annual vacation days.
2. Vacation requests must be approved by managers.
3. Employees receive 10 sick leave days.
4. Employees can carry forward five unused vacation days.

An evaluator can compare each retrieved context with the reference answer and determine whether it contributes useful information.

### Why reference answers help

Reference answers provide a target for determining retrieval relevance.

Without a reference answer, relevance may need to be evaluated using:

- The question itself
- LLM-based relevance judgments
- Other annotations

### Practical understanding

Context Precision is not simply checking whether documents contain the same words as the reference answer.

A document can be relevant even if it uses different wording.

---

# 7. Context Precision With and Without a Reference

RAGAS supports different Context Precision evaluation approaches.

| Metric | Evaluation approach |
|---|---|
| Context Precision | Reference-based relevance evaluation |
| Context Precision without reference | Question-based relevance evaluation |
| Context Precision with reference | Context compared against expected answer |

The exact class names and supported arguments depend on the RAGAS version.

Conceptually:

### With Reference

Question:

> What is the annual vacation allowance?

Reference:

> Employees receive 20 vacation days.

Context:

> Employees are entitled to 20 paid vacation days every year.

The context supports the expected answer.

### Without Reference

Question:

> What is the annual vacation allowance?

Context:

> Employees are entitled to 20 paid vacation days every year.

The evaluator determines whether the context is relevant to the question.

---

# 8. High and Low Context Precision Examples

## Example A — High Context Precision

Question:

> What is the refund policy?

Retrieved contexts:

1. Customers can request refunds within 30 days.
2. Refunds are processed within five business days.
3. Refund requests must include the order number.

All three contexts are useful.

Result:

- High Context Precision
- Low retrieval noise
- Better input for answer generation

## Example B — Low Context Precision

Question:

> What is the refund policy?

Retrieved contexts:

1. Company revenue increased by 15%.
2. The company has 200 employees.
3. Customers can request refunds within 30 days.
4. The company operates in five countries.

Only one document is directly relevant.

Result:

- Low Context Precision
- High retrieval noise
- Relevant information is surrounded by irrelevant information

## Example C — Relevant Documents Ranked Poorly

Question:

> What is the refund policy?

Retrieved contexts:

1. Company employee benefits
2. Company revenue
3. Refund policy
4. Refund processing procedure

The relevant information appears at positions 3 and 4.

The retrieval system found relevant documents but did not rank them well.

This is an important retrieval-ranking problem.

---

# 9. Python Implementation — Manual Context Precision

Before using RAGAS, understand how relevance labels affect the calculation.

We will manually assign relevance labels.

## Example Dataset

```python
question = "What is the annual vacation allowance?"

retrieved_contexts = [
    "Employees receive 20 annual vacation days.",
    "Employees receive 10 sick leave days.",
    "Vacation requests require manager approval.",
    "The company has 500 employees."
]

relevance = [1, 0, 1, 0]
```

Here:

- `1` means relevant
- `0` means irrelevant

## Implementation

```python
def calculate_context_precision(relevance):
    relevant_count = sum(relevance)

    if relevant_count == 0:
        return 0.0

    cumulative_relevant = 0
    precision_sum = 0.0

    for rank, rel in enumerate(relevance, start=1):
        cumulative_relevant += rel

        precision_at_rank = cumulative_relevant / rank

        if rel == 1:
            precision_sum += precision_at_rank

    context_precision = precision_sum / relevant_count

    return context_precision


relevance = [1, 0, 1, 0]

score = calculate_context_precision(relevance)

print("Context Precision:", round(score, 3))
```

Output:

```text
Context Precision: 0.833
```

## Code Explanation

### Step 1

```python
relevant_count = sum(relevance)
```

Counts all relevant documents.

### Step 2

```python
cumulative_relevant += rel
```

Tracks how many relevant documents have appeared so far.

### Step 3

```python
precision_at_rank = cumulative_relevant / rank
```

Calculates precision at the current rank.

### Step 4

```python
if rel == 1:
    precision_sum += precision_at_rank
```

Only relevant document positions contribute to the final score.

### Step 5

```python
context_precision = precision_sum / relevant_count
```

Calculates the final average.

This implementation is a simplified ranking-based illustration using manually assigned relevance labels. It is not a complete reproduction of every RAGAS Context Precision implementation.

---

# 10. RAGAS Implementation — Context Precision

## Installation

```bash
pip install ragas openai
```

Set your API key:

```bash
export OPENAI_API_KEY="your_api_key"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

## Dataset Structure

RAGAS evaluation generally uses these fields:

| Field | Meaning |
|---|---|
| user_input | Original user question |
| retrieved_contexts | Documents retrieved by your RAG system |
| response | Generated answer |
| reference | Expected answer, when required by the metric |

## Example Using the Collections API

The following example uses a reference-based Context Precision metric.

```python
import asyncio
import os

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision


async def main():

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"]
    )

    evaluator_llm = llm_factory(
        "gpt-4o-mini",
        client=client
    )

    metric = ContextPrecision(
        llm=evaluator_llm
    )

    question = "What is the annual vacation allowance?"

    reference = (
        "Employees receive 20 annual vacation days."
    )

    retrieved_contexts = [
        "Employees receive 20 annual vacation days.",
        "Employees receive 10 sick leave days.",
        "Vacation requests require manager approval."
    ]

    result = await metric.ascore(
        user_input=question,
        reference=reference,
        retrieved_contexts=retrieved_contexts
    )

    print("Context Precision:", result.value)


if __name__ == "__main__":
    asyncio.run(main())
```

### Important

RAGAS APIs and metric names can vary across versions.

Before running the example, verify the installed version and the metric's expected inputs.

```bash
pip show ragas
```

If the import or arguments fail, consult the current official documentation:

https://docs.ragas.io/

Do not assume that every RAGAS metric accepts the same fields.

---

# 11. Context Precision in a Real RAG Application

Consider this architecture:

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
Context Precision Evaluation
     |
     v
LLM Answer Generation
     |
     v
Final Answer
```

Example:

```python
question = "How many annual vacation days do employees receive?"

contexts = retriever.invoke(question)

answer = llm.invoke(
    f"""
    Answer the question using the context.

    Question:
    {question}

    Context:
    {contexts}
    """
)
```

For evaluation, store the retrieved contexts separately.

```python
evaluation_record = {
    "user_input": question,
    "retrieved_contexts": [
        document.page_content
        for document in contexts
    ],
    "response": answer.content,
    "reference": (
        "Employees receive 20 annual vacation days."
    )
}
```

This record can later be passed into your RAGAS evaluation pipeline.

---

# 12. Improving Context Precision

Low Context Precision usually indicates retrieval problems.

## Improvement 1 — Improve Chunking

Poor chunking may produce documents containing unrelated information.

Instead of:

```text
Chunk 1:
Employee policy, salary, attendance, vacation,
insurance, promotion and resignation details.
```

Prefer:

```text
Chunk 1:
Employee vacation policy.

Chunk 2:
Employee salary policy.

Chunk 3:
Employee attendance policy.
```

Smaller, meaningful chunks often improve retrieval relevance.

## Improvement 2 — Improve Embeddings

Embeddings convert text into numerical representations.

Better embeddings can improve semantic retrieval.

Example:

```python
query = "How many holidays can employees take?"
```

A good embedding model should retrieve:

```text
Employees receive 20 annual vacation days.
```

Even though the wording is different.

## Improvement 3 — Use Hybrid Search

Hybrid search combines:

- Vector search
- Keyword search

This helps when queries contain exact terms such as:

- Employee IDs
- Policy numbers
- Product names
- Technical identifiers

## Improvement 4 — Use a Reranker

A reranker evaluates retrieved documents again and sorts them according to relevance.

```text
Initial Retrieval
       |
       v
Top 20 Documents
       |
       v
Reranker
       |
       v
Top 5 Relevant Documents
       |
       v
LLM
```

Reranking is especially useful when initial retrieval finds relevant documents but ranks them poorly.

## Improvement 5 — Tune Top-K

Too many retrieved documents can introduce noise.

Too few documents can miss relevant information.

Experiment with:

```python
k = 3
k = 5
k = 10
k = 20
```

Evaluate Context Precision and Context Recall together.

## Improvement 6 — Add Metadata Filtering

Example:

```python
filter = {
    "department": "HR"
}
```

If the question is about HR policies, metadata filtering can reduce unrelated retrieval results.

---

# 13. Practical Context Precision Evaluation Workflow

For every test question:

1. Retrieve documents from your vector database.
2. Store retrieved documents in ranking order.
3. Compare retrieved documents with expected relevant information.
4. Assign relevance judgments or use an evaluator.
5. Calculate Context Precision.
6. Identify irrelevant documents.
7. Investigate why they were retrieved.
8. Improve chunking, embeddings, filters or reranking.
9. Run evaluation again.
10. Compare before and after scores.

### Example Evaluation Table

| Question | Before | After | Improvement |
|---|---:|---:|---:|
| Vacation policy | 0.55 | 0.91 | +0.36 |
| Refund policy | 0.62 | 0.88 | +0.26 |
| Leave eligibility | 0.70 | 0.93 | +0.23 |

These numbers are illustrative, not actual benchmark results.

---

# 14. Common Context Precision Mistakes

### Mistake 1: Treating Context Precision as Answer Accuracy

Context Precision evaluates retrieval, not the final answer.

### Mistake 2: Assuming More Documents Means Better Retrieval

More documents may increase irrelevant context.

### Mistake 3: Evaluating Only Keywords

A document may be semantically relevant without sharing exact keywords.

### Mistake 4: Ignoring Ranking Order

Relevant documents appearing at the bottom indicate a ranking issue.

### Mistake 5: Evaluating Context Precision Alone

A system may have high Context Precision but low Context Recall.

Evaluate both.

### Mistake 6: Assuming High Context Precision Guarantees Correct Answers

The retrieved context may be relevant but incomplete, outdated, or contradictory.

The generation model may still produce an incorrect answer.

---

# 15. Interview Questions and Answers

### Q1. What is Context Precision?

Context Precision measures how well a RAG retriever ranks relevant information among retrieved documents.

### Q2. Why is Context Precision important?

It helps identify retrieval noise and improves the quality of information provided to the generation model.

### Q3. What is the difference between Context Precision and Context Recall?

Context Precision evaluates relevance among retrieved documents. Context Recall evaluates how much relevant information was retrieved.

### Q4. Can Context Precision be high while Context Recall is low?

Yes. A retriever may return only a few relevant documents while missing other important documents.

### Q5. How can Context Precision be improved?

Use better embeddings, chunking, hybrid search, metadata filtering, reranking and appropriate top-K settings.

### Q6. What is a reranker?

A reranker re-evaluates retrieved documents and sorts them based on relevance to the query.

### Q7. Does Context Precision evaluate the final answer?

No. It evaluates retrieval quality.

### Q8. Can relevant documents still produce low Context Precision?

Yes. If irrelevant documents are also retrieved or relevant documents appear lower in the ranking, Context Precision may decrease.

### Q9. Why use Context Precision with a reference answer?

A reference answer provides an expected target for determining whether retrieved documents contain information relevant to answering the question.

### Q10. Why evaluate Context Precision and Context Recall together?

Because retrieval should be both relevant and sufficiently complete.

---

# 16. Practical Exercises

## Exercise 1 — Calculate Context Precision

Question:

> What is the company's refund period?

Retrieved relevance labels:

```python
relevance = [1, 0, 1, 0, 1]
```

Tasks:

1. Calculate relevant document count.
2. Calculate Precision@1.
3. Calculate Precision@3.
4. Calculate Precision@5.
5. Calculate final Context Precision.

Expected result:

```text
Context Precision = 0.756
```

## Exercise 2 — Identify Retrieval Problems

Question:

> How many sick leave days are available?

Retrieved documents:

1. Vacation policy
2. Salary structure
3. Sick leave policy
4. Attendance policy
5. Promotion policy

Tasks:

- Identify relevant documents.
- Identify irrelevant documents.
- Explain why ranking is poor.
- Suggest three improvements.

## Exercise 3 — Improve Retrieval

Experiment with:

- Different chunk sizes
- Different embedding models
- Different top-K values
- Hybrid search
- Reranking

Record Context Precision after every change.

## Exercise 4 — Build a Small Evaluation Dataset

Create 10 questions.

For each question, store:

- Question
- Reference answer
- Retrieved contexts
- Generated answer

Evaluate Context Precision and compare results.

---

# 17. Module Summary

You learned:

- Context Precision evaluates retrieval relevance and ranking.
- Relevant documents should appear near the top.
- Irrelevant documents reduce retrieval quality.
- Context Precision differs from Context Recall.
- Reference answers can help evaluate retrieved context.
- Manual relevance labels help understand the calculation.
- RAGAS can automate Context Precision evaluation.
- Chunking, embeddings, hybrid search and reranking can improve retrieval.

## Completion Checklist

- [ ] I understand Context Precision.
- [ ] I understand relevance labels.
- [ ] I can calculate Precision@k.
- [ ] I can calculate a simplified Context Precision score.
- [ ] I understand Context Precision vs Context Recall.
- [ ] I understand reference-based evaluation.
- [ ] I can implement a manual Python calculation.
- [ ] I understand RAGAS Context Precision.
- [ ] I can identify retrieval-ranking problems.
- [ ] I understand how reranking improves retrieval.

---

# 18. Next Module

## Module 5 — Context Recall: Evaluate Retrieval Completeness

Topics:

1. What is Context Recall?
2. Context Precision vs Context Recall
3. Ground truth and reference answers
4. Required information vs retrieved information
5. Context Recall calculation
6. High and low Context Recall examples
7. Python implementation
8. RAGAS implementation
9. Improving retrieval completeness
10. Precision-Recall trade-off in RAG

