# Module 3 — Answer Relevancy: Evaluate Answers

## Learning Objectives

By completing this module, you will understand:

1. What Answer Relevancy is.
2. Answer Relevancy vs Answer Correctness.
3. Understanding questions and generated answers.
4. Reverse question generation.
5. Embedding similarity.
6. Answer Relevancy score calculation.
7. High and low relevancy examples.
8. Python implementation.
9. Improving answer relevance.

---

# 1. What Is Answer Relevancy?

## Definition

**Answer Relevancy is a RAGAS evaluation metric that measures how relevant a generated answer is to the user's original question.**

It evaluates whether the generated answer addresses the information requested by the user.

An answer can be grammatically correct and factually accurate but still irrelevant to the user's question.

## Example

### User Question

```text
What is the capital of India?
```

### Answer A

```text
The capital of India is New Delhi.
```

This answer directly addresses the question.

### Answer B

```text
India is a country in South Asia.
It has a population of more than one billion people.
```

This answer contains relevant information about India, but it does not answer the question.

### Evaluation

| Answer | Answer Relevancy |
|---|---|
| Answer A | High |
| Answer B | Low |

## Why Answer Relevancy Matters

Answer Relevancy helps identify:

- Answers that do not address the question.
- Answers containing unrelated information.
- Answers that discuss the topic without answering the request.
- Answers that provide unnecessary details.
- Answers that misunderstand user intent.

## Technical Definition

Answer Relevancy evaluates whether the generated answer contains information that addresses the user's question.

RAGAS uses an LLM-based approach involving reverse question generation and semantic similarity.

---

# 2. Answer Relevancy vs Answer Correctness

These concepts are related but different.

## 2.1 Answer Relevancy

Measures whether the answer addresses the user's question.

## 2.2 Answer Correctness

Measures whether the answer is factually correct compared with a reference answer or other factual evidence.

## Example 1: Relevant but Incorrect

Question:

```text
What is the capital of Australia?
```

Answer:

```text
The capital of Australia is Sydney.
```

Evaluation:

| Metric | Result |
|---|---|
| Answer Relevancy | High |
| Answer Correctness | Low |

The answer directly addresses the question but contains incorrect information.

## Example 2: Correct but Irrelevant

Question:

```text
What is the capital of Australia?
```

Answer:

```text
Australia is a country in the Southern Hemisphere.
```

The statement is factually correct but does not answer the question.

| Metric | Result |
|---|---|
| Answer Relevancy | Low |
| Answer Correctness | Not sufficient to establish the requested answer's correctness |

## Example 3: Relevant and Correct

Question:

```text
What is the capital of Australia?
```

Answer:

```text
Canberra is the capital of Australia.
```

| Metric | Result |
|---|---|
| Answer Relevancy | High |
| Answer Correctness | High |

## Comparison Table

| Answer Relevancy | Answer Correctness |
|---|---|
| Measures relevance to question | Measures factual accuracy |
| Focuses on user intent | Focuses on correctness |
| Can be evaluated without reference answers | Usually requires reference information |
| Identifies irrelevant responses | Identifies incorrect responses |
| Uses question-answer relationship | Uses answer-reference relationship |

## Important Understanding

**A relevant answer is not necessarily a correct answer.**

Similarly:

**A correct statement is not necessarily a relevant answer.**

Both dimensions should be evaluated separately.

---

# 3. Understanding the Question and Generated Answer

To understand Answer Relevancy, first understand the relationship between three elements.

## Components

- User Question
- Generated Answer
- Reference Answer (optional for Answer Relevancy)

## Example

User Question:

```text
What is the difference between RAM and ROM?
```

Generated Answer:

```text
RAM is temporary memory used while programs are running.
ROM stores permanent instructions and retains data without power.
```

Reference Answer:

```text
RAM is volatile memory, while ROM is non-volatile memory.
```

The generated answer explains the requested difference.

Even though the wording differs from the reference answer, the generated answer is relevant.

## Another Example

User Question:

```text
What is the difference between RAM and ROM?
```

Generated Answer:

```text
RAM is used in computers.
Computers contain CPUs, storage devices, and operating systems.
```

This answer discusses computers but does not explain the difference between RAM and ROM.

Therefore, its Answer Relevancy is low.

## Important Concept

Answer Relevancy is concerned with the relationship:

```text
User Question <----> Generated Answer
```

It is not simply a comparison between the generated answer and the reference answer.

---

# 4. Reverse Question Generation

Reverse question generation is a core concept behind RAGAS Answer Relevancy.

## Definition

**Reverse question generation is the process of generating possible questions that could have been answered by a given answer.**

Instead of directly asking:

"Does this answer address the original question?"

The evaluation approach asks:

"What question would this answer naturally answer?"

Then it compares the generated questions with the original user question.

## Why Is This Useful?

A generated answer may use completely different wording from the user's question.

Direct keyword comparison would fail.

Reverse question generation evaluates the semantic relationship instead.

## Example 1: Relevant Answer

Original Question:

```text
What is the capital of India?
```

Generated Answer:

```text
New Delhi is the capital city of India.
```

Reverse Question:

```text
What is the capital city of India?
```

The reverse question closely matches the original question.

Therefore, Answer Relevancy is high.

## Example 2: Irrelevant Answer

Original Question:

```text
What is the capital of India?
```

Generated Answer:

```text
India is located in South Asia.
```

Reverse Question:

```text
Where is India located?
```

The reverse question differs significantly from the original question.

Therefore, Answer Relevancy is low.

## Workflow

```text
Original Question
       |
       v
Generated Answer
       |
       v
LLM Generates Reverse Questions
       |
       v
Reverse Questions
       |
       v
Compare With Original Question
       |
       v
Semantic Similarity
       |
       v
Answer Relevancy Score
```

## Example With Multiple Reverse Questions

Original Question:

```text
What are the benefits of using FastAPI?
```

Generated Answer:

```text
FastAPI provides automatic API documentation,
high performance, and request validation.
```

Possible Reverse Questions:

```text
1. What features does FastAPI provide?

2. What are the benefits of FastAPI?

3. Why should developers use FastAPI?
```

These reverse questions are semantically related to the original question.

Therefore, the answer is highly relevant.

## Important Understanding

Reverse question generation is an evaluation technique.

It does not modify the original RAG answer.

It is used by the evaluation process to estimate Answer Relevancy.

---

# 5. Embedding Similarity

## Definition

**Embedding similarity measures how semantically similar two pieces of text are by comparing their vector representations.**

An embedding model converts text into numerical vectors.

Example:

```text
"What is the capital of India?"
```

Embedding:

```text
[0.12, -0.45, 0.67, ...]
```

Another text:

```text
"Which city is India's capital?"
```

Embedding:

```text
[0.10, -0.42, 0.64, ...]
```

These vectors may be close because the texts have similar meanings.

## What Is an Embedding?

An embedding is a numerical representation of text in a vector space.

Embedding models encode semantic information into vectors so that semantically similar texts tend to have similar representations.

## Cosine Similarity

A commonly used similarity measure is cosine similarity.

Formula:

\[
\text{Cosine Similarity}
=
\frac{A \cdot B}{\|A\|\|B\|}
\]

Where:

- A = First embedding vector.
- B = Second embedding vector.
- A · B = Dot product.
- ||A|| = Magnitude of vector A.
- ||B|| = Magnitude of vector B.

For normalized vectors, cosine similarity is equivalent to their dot product.

## Example

Original Question:

```text
What is the capital of India?
```

Reverse Question:

```text
Which city serves as India's capital?
```

These questions have similar meanings.

Their embeddings are expected to have high similarity.

## Important Understanding

Embedding similarity measures semantic similarity, not factual correctness.

Two texts can be semantically similar while containing different factual values.

---

# 6. Answer Relevancy Score Calculation

## Conceptual Calculation

RAGAS Answer Relevancy uses reverse question generation and semantic similarity.

Conceptually:

1. Generate reverse questions from the answer.
2. Convert the original question into an embedding.
3. Convert reverse questions into embeddings.
4. Calculate semantic similarity.
5. Average the similarities.

## Formula

\[
\text{Answer Relevancy}
=
\frac{1}{N}
\sum_{i=1}^{N}
\text{Similarity}(Q,Q_i)
\]

Where:

- Q = Original user question.
- Qᵢ = Generated reverse question.
- N = Number of reverse questions.
- Similarity = Semantic similarity between embeddings.

## Example

Original Question:

```text
What is the capital of India?
```

Generated Reverse Questions:

```text
Q1: Which city is India's capital?

Q2: What is the capital city of India?

Q3: Which city serves as the capital of India?
```

Suppose similarity scores are:

```text
Q1 = 0.95

Q2 = 0.98

Q3 = 0.94
```

Calculation:

\[
\frac{0.95+0.98+0.94}{3}
\]

\[
=\frac{2.87}{3}
\]

\[
=0.9567
\]

Answer Relevancy:

```text
0.9567
```

Interpretation:

The answer is highly relevant to the original question.

## Important RAGAS Note

This formula represents the conceptual scoring approach.

Actual RAGAS implementations can include implementation-specific handling, reverse-question generation settings, and aggregation behavior.

The score should be interpreted as a metric estimate rather than an absolute percentage of answer correctness.

---

# 7. High and Low Answer Relevancy Examples

## Example 1: High Relevancy

Question:

```text
What is LangGraph?
```

Answer:

```text
LangGraph is a framework for building stateful,
multi-step applications using LLMs and graphs.
```

Why?

The answer directly defines LangGraph.

Expected:

```text
High Answer Relevancy
```

## Example 2: Low Relevancy

Question:

```text
What is LangGraph?
```

Answer:

```text
LangChain provides abstractions for working
with language models and tools.
```

Why?

The answer discusses a related technology but does not define LangGraph.

Expected:

```text
Low Answer Relevancy
```

## Example 3: Relevant but Incomplete

Question:

```text
What are the advantages of FastAPI?
```

Answer:

```text
FastAPI supports automatic API documentation.
```

The answer addresses one advantage but misses other potential advantages.

Expected:

```text
Answer Relevancy: Potentially High
Completeness: Limited
```

Answer Relevancy does not necessarily measure completeness.

## Example 4: Relevant but Incorrect

Question:

```text
Which language is FastAPI built on?
```

Answer:

```text
FastAPI is built on Java.
```

The answer directly addresses the question but is incorrect.

Expected:

```text
Answer Relevancy: High
Answer Correctness: Low
```

## Example 5: Unnecessary Information

Question:

```text
What is the default port of FastAPI when using Uvicorn?
```

Answer:

```text
Uvicorn is an ASGI server.

FastAPI is a Python web framework.

Uvicorn commonly runs on port 8000 by default.
Python is widely used in AI development.
```

The answer contains the required information but also unnecessary details.

Answer Relevancy may remain high because the answer addresses the question.

However, answer conciseness and relevance are not identical.

---

# 8. Python Implementation

We will implement Answer Relevancy in two ways:

1. Manual embedding similarity.
2. RAGAS Answer Relevancy evaluation.

## 8.1 Manual Reverse Question Generation

This implementation demonstrates the concept.

It does not implement the complete RAGAS evaluation algorithm.

### Code

```python
original_question = "What is the capital of India?"

generated_answer = (
    "New Delhi is the capital city of India."
)

reverse_questions = [
    "Which city is India's capital?",
    "What is the capital city of India?",
    "Which city serves as the capital of India?"
]

print("Original Question:")
print(original_question)

print("\nGenerated Answer:")
print(generated_answer)

print("\nReverse Questions:")

for question in reverse_questions:
    print("-", question)
```

### Output

```text
Original Question:
What is the capital of India?

Generated Answer:
New Delhi is the capital city of India.

Reverse Questions:
- Which city is India's capital?
- What is the capital city of India?
- Which city serves as the capital of India?
```

## 8.2 Manual Embedding Similarity

Install:

```bash
pip install sentence-transformers
```

### Code

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

original_question = (
    "What is the capital of India?"
)

reverse_questions = [
    "Which city is India's capital?",
    "What is the capital city of India?",
    "Which city serves as the capital of India?"
]

# Generate embeddings
question_embedding = model.encode(
    original_question
)

reverse_embeddings = model.encode(
    reverse_questions
)

# Calculate similarities
scores = cosine_similarity(
    [question_embedding],
    reverse_embeddings
)[0]

# Calculate average similarity
answer_relevancy = scores.mean()

print("Similarity Scores:")

for question, score in zip(
    reverse_questions,
    scores
):
    print(question, ":", round(float(score), 4))

print(
    "\nAverage Answer Relevancy:",
    round(float(answer_relevancy), 4)
)
```

### Explanation

- SentenceTransformer generates embeddings.
- Cosine similarity compares the original question with reverse questions.
- The mean similarity represents a simplified Answer Relevancy score.

This is a conceptual implementation, not the official RAGAS metric.

---

## 8.3 RAGAS Answer Relevancy Implementation

### Installation

```bash
pip install -U ragas openai
```

Configure your API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

### Code

```python
import asyncio
import os

from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory

from ragas.metrics.collections import AnswerRelevancy


async def main():

    # 1. Initialize OpenAI client
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"]
    )

    # 2. Configure evaluation LLM
    evaluator_llm = llm_factory(
        "gpt-4o-mini",
        client=client
    )

    # 3. Configure embedding model
    evaluator_embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=client
    )

    # 4. Create Answer Relevancy metric
    metric = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    # 5. Define question
    question = (
        "What is the capital of India?"
    )

    # 6. Define generated answer
    answer = (
        "New Delhi is the capital city of India."
    )

    # 7. Evaluate Answer Relevancy
    result = await metric.ascore(
        user_input=question,
        response=answer
    )

    # 8. Print result
    print(
        "Answer Relevancy Score:",
        result.value
    )


if __name__ == "__main__":
    asyncio.run(main())
```

### Expected Result

The score should generally be high because the answer directly addresses the question.

```text
Answer Relevancy Score: High
```

The exact numerical result depends on the evaluation model and embedding model.

### Important

The Answer Relevancy metric does not require retrieved contexts or reference answers for this evaluation.

It primarily evaluates the relationship between:

```text
User Question + Generated Answer
```

---

# 9. Improving Answer Relevance

Low Answer Relevancy indicates that the generated answer may not address the user's actual request.

We should investigate why this happens.

## 9.1 Improve Prompt Engineering

Weak Prompt:

```text
Answer the user question.
```

Improved Prompt:

```text
Answer the user's question directly.

Use the retrieved context to provide the answer.

Avoid unrelated information.

If the context does not contain sufficient information,
state that the information is unavailable.
```

This encourages focused responses.

## 9.2 Improve Query Understanding

Example:

Question:

```text
Explain the difference between RAG and fine-tuning.
```

If the system interprets the question as:

```text
Explain RAG.
```

It may generate an incomplete answer.

Improve:

- Query rewriting.
- Intent detection.
- Question decomposition.
- Query expansion when appropriate.

## 9.3 Improve Retrieval

Irrelevant retrieval context can influence the answer.

Improve:

- Embedding models.
- Chunking strategies.
- Top-K selection.
- Reranking.
- Metadata filtering.

## 9.4 Improve Answer Structure

For comparison questions, request structured comparisons.

Example:

```text
Compare RAG and fine-tuning using a table.
```

For definition questions:

```text
Provide a concise definition and one example.
```

For procedural questions:

```text
Explain the steps in sequence.
```

This improves the probability that the answer addresses the requested format and content.

## 9.5 Reduce Unnecessary Information

Example:

Question:

```text
What is FastAPI?
```

Poor Answer:

```text
FastAPI is a Python framework.

Python is used in machine learning.

Machine learning is a branch of AI.

AI is becoming popular.
```

Improved Answer:

```text
FastAPI is a Python web framework for building APIs
with automatic documentation and request validation.
```

The improved answer is more focused.

## 9.6 Evaluate Before and After

Example:

| Version | Answer Relevancy |
|---|---:|
| Original prompt | 0.65 |
| Improved prompt | 0.85 |
| Improved retrieval + prompt | 0.93 |

These numbers are illustrative, not actual experimental results.

The important point is to measure whether changes improve answer quality.

---

# 10. Answer Relevancy Best Practices

1. Evaluate answers against the original user question.
2. Do not confuse relevance with factual correctness.
3. Use reference-based metrics for factual validation.
4. Avoid unnecessary information in generated answers.
5. Use appropriate answer formats.
6. Improve query understanding when user intent is ambiguous.
7. Improve retrieval if irrelevant context influences generation.
8. Use representative evaluation datasets.
9. Review low-scoring examples manually.
10. Compare scores before and after prompt changes.

---

# 11. Module 3 Interview Questions

## Beginner

1. What is Answer Relevancy in RAGAS?
2. Why is Answer Relevancy important?
3. What is reverse question generation?
4. What is embedding similarity?
5. What is cosine similarity?
6. Can a relevant answer be incorrect?
7. Does Answer Relevancy require reference answers?

## Intermediate

8. How does RAGAS calculate Answer Relevancy?
9. Why is reverse question generation useful?
10. How is Answer Relevancy different from Faithfulness?
11. How is Answer Relevancy different from Factual Correctness?
12. What are the limitations of embedding similarity?
13. How can prompt engineering improve Answer Relevancy?
14. How can irrelevant retrieved context affect answer relevance?

## Advanced

15. How would you evaluate Answer Relevancy for a production RAG application?
16. How would you investigate low Answer Relevancy scores?
17. How would you compare two LLMs using Answer Relevancy?
18. How would you validate an Answer Relevancy evaluation dataset?
19. Can Answer Relevancy be high when the answer is incomplete?
20. How would you improve answer relevance without increasing latency significantly?

---

# 12. Practical Exercises

## Exercise 1: Identify Answer Relevancy

Question:

```text
What is the difference between Python and Java?
```

Answer:

```text
Python is dynamically typed and generally uses indentation
to define code blocks.

Java is statically typed and uses braces to define code blocks.
```

Tasks:

1. Determine whether the answer is relevant.
2. Explain why.
3. Identify what information is being compared.

## Exercise 2: Relevant but Incorrect

Question:

```text
What is the default port of Uvicorn?
```

Answer:

```text
Uvicorn uses port 9000 by default.
```

Tasks:

1. Determine Answer Relevancy.
2. Determine Answer Correctness.
3. Explain the difference.

## Exercise 3: Reverse Question Generation

Question:

```text
What are the benefits of using vector databases?
```

Answer:

```text
Vector databases support similarity search,
efficient vector retrieval, and semantic search.
```

Tasks:

1. Generate three reverse questions.
2. Compare them with the original question.
3. Explain why Answer Relevancy should be high.

## Exercise 4: RAGAS Evaluation

Evaluate these examples:

```python
examples = [
    {
        "question": "What is Python?",
        "answer": "Python is a programming language."
    },
    {
        "question": "What is Python?",
        "answer": "Java is a programming language."
    },
    {
        "question": "What is Python?",
        "answer": "Programming languages are used to build software."
    }
]
```

Tasks:

1. Run RAGAS Answer Relevancy.
2. Compare the results.
3. Identify which answer is most relevant.
4. Explain why the second answer may be factually incorrect.

---

# 13. Module 3 Completion Checklist

- [ ] What is Answer Relevancy?
- [ ] Answer Relevancy vs Answer Correctness
- [ ] Understanding question and generated answer
- [ ] Reverse question generation
- [ ] Embedding similarity
- [ ] Answer Relevancy score calculation
- [ ] High and low relevancy examples
- [ ] Manual Python implementation
- [ ] RAGAS Python implementation
- [ ] Improving answer relevance
- [ ] Interview questions
- [ ] Practical exercises

---

# Final Takeaway

Answer Relevancy helps answer:

**"Is my RAG application actually answering what the user asked?"**

It evaluates whether generated answers address the user's question, regardless of whether the wording matches the question.

However, Answer Relevancy alone cannot guarantee:

- Factual correctness
- Context grounding
- Retrieval completeness
- Absence of hallucinations

For reliable RAG applications, evaluate Answer Relevancy together with Faithfulness, Context Precision, Context Recall, and factual correctness metrics.

---

# Official References

- RAGAS Answer Relevancy Documentation: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/
- RAGAS GitHub: https://github.com/vibrantlabsai/ragas
- RAGAS Research Paper: https://arxiv.org/abs/2309.15217

---


