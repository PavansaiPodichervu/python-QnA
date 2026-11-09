# RAGAS — Module 8: Evaluate a Real RAG Application

## 1. Learning Objectives

By completing this module, you will understand how to:

- Evaluate an actual RAG application instead of isolated examples.
- Capture retrieved documents and generated answers.
- Create evaluation datasets from real application execution.
- Measure retrieval quality and answer quality separately.
- Compare different chunk sizes, retrieval strategies, and prompts.
- Identify whether failures originate from retrieval or generation.
- Build a repeatable evaluation pipeline.

---

## 2. Real RAG Application Architecture

A real RAG application contains two separate workflows.

### Runtime workflow

```text
User Question
     |
     v
Embedding Model
     |
     v
Vector Database
     |
     v
Retrieve Top-K Documents
     |
     v
Build Prompt + Context
     |
     v
LLM Generates Answer
     |
     v
Return Answer to User
```

### Evaluation workflow

```text
Evaluation Questions
       |
       v
Execute RAG Application
       |
       v
Capture Question
Retrieved Contexts
Generated Answer
Reference Answer
       |
       v
RAGAS Evaluation
       |
       v
Faithfulness
Answer Relevancy
Context Precision
Context Recall
       |
       v
Analyze Failures
       |
       v
Improve RAG
```

**Important:** RAGAS does not automatically know what your application retrieved or generated. You must collect these values from your application.

Reference: https://docs.ragas.io/en/latest/getstarted/rag_eval/

---

## 3. Evaluation Data from a Real Application

Each evaluation record should contain:

| Field | Meaning |
|---|---|
| user_input | Original user question |
| retrieved_contexts | Documents retrieved by your application |
| response | Actual LLM-generated answer |
| reference | Expected correct answer |

Example:

```python
{
    "user_input": "What is FastAPI?",
    "retrieved_contexts": [
        "FastAPI is a Python framework for building APIs.",
        "FastAPI supports asynchronous request handling."
    ],
    "response": "FastAPI is a Python framework used to build APIs.",
    "reference": "FastAPI is a Python framework for building APIs."
}
```

This is the most important connection between your application and RAGAS.

---

## 4. Create a Small Real RAG Application

For this module, we will use:

- Python
- OpenAI embeddings
- FAISS vector database
- OpenAI chat model
- RAGAS

### Install dependencies

```bash
pip install -U openai faiss-cpu numpy ragas
```

Configure the API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

### Step 1: Create a knowledge base

Create `rag_app.py`.

```python
import numpy as np
import faiss

from openai import OpenAI

client = OpenAI()

documents = [
    "FastAPI is a Python framework for building APIs.",
    "FastAPI supports asynchronous request handling.",
    "Python is a high-level programming language.",
    "LangGraph is a framework for building stateful AI agents.",
    "RAG retrieves relevant documents before generating an answer.",
    "Vector databases store embeddings for similarity search."
]
```

These documents represent our knowledge base.

### Step 2: Generate embeddings

```python
def embed_texts(texts):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return np.array(
        [item.embedding for item in response.data],
        dtype="float32"
    )
```

**Embedding:** A numerical vector representing the semantic meaning of text.

Example:

```text
"FastAPI is a Python framework"
             |
             v
[0.12, -0.45, 0.83, ...]
```

### Step 3: Build the vector index

```python
document_embeddings = embed_texts(documents)

dimension = document_embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

# Normalize embeddings for cosine similarity
faiss.normalize_L2(document_embeddings)

index.add(document_embeddings)
```

### Step 4: Implement retrieval

```python
def retrieve(question, top_k=3):

    query_embedding = embed_texts([question])

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    retrieved_documents = [
        documents[i]
        for i in indices[0]
        if i >= 0
    ]

    return retrieved_documents
```

### Step 5: Implement answer generation

```python
def generate_answer(question, contexts):

    context_text = "\n".join(contexts)

    prompt = f"""
    Answer the question using only the provided context.

    If the context does not contain the answer,
    say that you do not know.

    Context:
    {context_text}

    Question:
    {question}

    Answer:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content
```

### Step 6: Create the complete RAG function

```python
def run_rag(question, top_k=3):

    contexts = retrieve(
        question,
        top_k=top_k
    )

    answer = generate_answer(
        question,
        contexts
    )

    return {
        "question": question,
        "retrieved_contexts": contexts,
        "response": answer
    }
```

### Step 7: Execute the application

```python
result = run_rag(
    "What is FastAPI?"
)

print(result)
```

Example output:

```python
{
    "question": "What is FastAPI?",
    "retrieved_contexts": [
        "FastAPI is a Python framework for building APIs.",
        "FastAPI supports asynchronous request handling.",
        "Python is a high-level programming language."
    ],
    "response": "FastAPI is a Python framework used to build APIs."
}
```

Your RAG application is now ready for evaluation.

---

## 5. Collect Evaluation Data Automatically

Create `collect_dataset.py`.

```python
from rag_app import run_rag

evaluation_questions = [
    {
        "question": "What is FastAPI?",
        "reference": "FastAPI is a Python framework for building APIs."
    },
    {
        "question": "What is LangGraph?",
        "reference": "LangGraph is a framework for building stateful AI agents."
    },
    {
        "question": "What is RAG?",
        "reference": "RAG retrieves relevant documents before generating an answer."
    },
    {
        "question": "What is a vector database?",
        "reference": "A vector database stores embeddings for similarity search."
    }
]

dataset = []

for item in evaluation_questions:

    output = run_rag(item["question"])

    record = {
        "user_input": item["question"],
        "retrieved_contexts": output["retrieved_contexts"],
        "response": output["response"],
        "reference": item["reference"]
    }

    dataset.append(record)

print(dataset)
```

### Explanation

For every evaluation question:

1. Execute your actual RAG application.
2. Retrieve documents from FAISS.
3. Generate an answer using the LLM.
4. Capture the actual retrieved documents.
5. Capture the generated response.
6. Add the expected reference answer.
7. Store the complete evaluation record.

This is how real RAG evaluation datasets are created.

---

## 6. Run RAGAS Evaluation

Create `evaluate_rag.py`.

The following uses the RAGAS collections API and explicit metric scoring.

```python
import asyncio

from openai import AsyncOpenAI

from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall
)

from collect_dataset import dataset


async def main():

    client = AsyncOpenAI()

    evaluator_llm = llm_factory(
        "gpt-4o-mini",
        client=client
    )

    faithfulness = Faithfulness(
        llm=evaluator_llm
    )

    answer_relevancy = AnswerRelevancy(
        llm=evaluator_llm
    )

    context_precision = LLMContextPrecisionWithReference(
        llm=evaluator_llm
    )

    context_recall = LLMContextRecall(
        llm=evaluator_llm
    )

    samples = [
        SingleTurnSample(
            user_input=item["user_input"],
            retrieved_contexts=item["retrieved_contexts"],
            response=item["response"],
            reference=item["reference"]
        )
        for item in dataset
    ]

    for sample in samples:

        scores = await asyncio.gather(
            faithfulness.single_turn_ascore(sample),
            answer_relevancy.single_turn_ascore(sample),
            context_precision.single_turn_ascore(sample),
            context_recall.single_turn_ascore(sample)
        )

        print({
            "faithfulness": scores[0].value,
            "answer_relevancy": scores[1].value,
            "context_precision": scores[2].value,
            "context_recall": scores[3].value
        })


asyncio.run(main())
```

### What each metric evaluates

| Metric | Evaluation question |
|---|---|
| Faithfulness | Is the answer supported by retrieved context? |
| Answer Relevancy | Does the answer address the question? |
| Context Precision | Are relevant documents ranked higher? |
| Context Recall | Did retrieval capture the information needed? |

Reference: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/

---

## 7. Understand the Evaluation Results

Imagine your application produces the following scores:

| Metric | Score |
|---|---:|
| Faithfulness | 0.95 |
| Answer Relevancy | 0.92 |
| Context Precision | 0.48 |
| Context Recall | 0.90 |

### Interpretation

**Faithfulness = 0.95**

Most generated claims are supported by retrieved documents.

**Answer Relevancy = 0.92**

The answers generally address the questions.

**Context Precision = 0.48**

The retriever is returning many irrelevant documents or ranking relevant documents poorly.

**Context Recall = 0.90**

Most information needed for answering is being retrieved.

### Engineering conclusion

The generator is performing reasonably well.

The retrieval system needs improvement because it includes unnecessary context.

Potential improvements:

- Improve embedding model.
- Improve document chunking.
- Tune top-k.
- Add reranking.
- Improve metadata filtering.

---

## 8. Experiment with Different Retrieval Configurations

You should never evaluate only one configuration.

Example:

| Experiment | Chunk size | Top-K |
|---|---:|---:|
| A | 200 | 3 |
| B | 500 | 3 |
| C | 1000 | 3 |
| D | 500 | 5 |
| E | 500 | 10 |

**Chunk size:** Number of characters or tokens grouped into a document chunk before embedding.

**Top-K:** Number of documents retrieved for a query.

Example:

```python
result = run_rag(
    "Explain LangGraph",
    top_k=5
)
```

For a proper chunk-size experiment, rebuild the index using the new chunking strategy before running the evaluation.

### Example experiment report

| Configuration | Faithfulness | Context Precision | Context Recall |
|---|---:|---:|---:|
| Chunk 200, K=3 | 0.80 | 0.75 | 0.60 |
| Chunk 500, K=3 | 0.91 | 0.82 | 0.84 |
| Chunk 1000, K=3 | 0.86 | 0.65 | 0.90 |
| Chunk 500, K=5 | 0.90 | 0.70 | 0.94 |

These values are illustrative, not actual measurements.

### What you may discover

- Small chunks may lose important context.
- Large chunks may contain irrelevant information.
- Increasing top-k may improve recall.
- Increasing top-k may reduce precision.
- Reranking may improve precision without sacrificing recall.

---

## 9. Failure Analysis: Retrieval vs Generation

This is one of the most important AI engineering skills.

### Case A: Retrieval failure

Question:

```text
What is LangGraph?
```

Retrieved contexts:

```text
Python is a high-level programming language.
FastAPI is a Python framework for APIs.
```

Generated answer:

```text
I don't have enough information.
```

Root cause:

The retriever did not find the correct document.

Fix:

Improve retrieval.

### Case B: Generation failure

Question:

```text
What is LangGraph?
```

Retrieved contexts:

```text
LangGraph is a framework for building stateful AI agents.
```

Generated answer:

```text
LangGraph is a database used to store embeddings.
```

Root cause:

The retrieved context is correct, but the LLM generated an unsupported answer.

Fix:

Improve prompt instructions, model behavior, or generation settings.

### Case C: Context noise

Question:

```text
What is FastAPI?
```

Retrieved contexts:

```text
FastAPI is a Python framework for building APIs.
Python is a high-level programming language.
LangGraph is a framework for stateful agents.
Vector databases store embeddings.
```

Root cause:

Too much unrelated context.

Fix:

Improve context precision, filtering, and reranking.

---

## 10. Practical Engineering Checklist

Before evaluating a real RAG application, verify:

- [ ] Evaluation questions are representative.
- [ ] Reference answers are correct.
- [ ] Retrieved contexts are captured from the real retriever.
- [ ] Generated answers are captured from the real LLM.
- [ ] The same dataset is used for every experiment.
- [ ] Each experiment has a configuration identifier.
- [ ] Evaluation model and application model are tracked separately.
- [ ] Failures are inspected individually.
- [ ] Scores are compared against a baseline.
- [ ] API costs are monitored.

---

## 11. Interview Questions and Answers

### Q1. How do you evaluate a production RAG application?

Capture actual questions, retrieved contexts, and generated answers. Prepare reference answers for evaluation and calculate appropriate RAGAS metrics.

### Q2. Why is evaluating only the generated answer insufficient?

A poor answer may originate from poor retrieval or poor generation. Evaluating both components helps isolate the root cause.

### Q3. What is the difference between application LLM and evaluator LLM?

Application LLM generates answers for users.

Evaluator LLM judges answer quality using evaluation metrics.

They can be different models.

### Q4. Why should we maintain a fixed evaluation dataset?

A fixed dataset allows meaningful comparison across prompts, embedding models, chunk sizes, retrievers, and generation models.

### Q5. How do you improve Context Recall?

- Increase top-k.
- Improve chunking.
- Improve embeddings.
- Use hybrid retrieval.
- Improve query rewriting.
- Improve document coverage.

### Q6. How do you improve Context Precision?

- Reduce irrelevant retrieved documents.
- Use metadata filters.
- Apply reranking.
- Improve embeddings.
- Tune top-k.

### Q7. What is a baseline?

A baseline is the initial measured performance of an application before improvements. Future experiments are compared against it.

---

## 12. Hands-on Assignment

Implement the following tasks:

1. Create a knowledge base with 20 documents.
2. Build a FAISS index.
3. Implement retrieval and generation.
4. Create 10 evaluation questions.
5. Capture actual retrieved contexts.
6. Generate actual responses.
7. Add reference answers.
8. Calculate Faithfulness.
9. Calculate Answer Relevancy.
10. Calculate Context Precision.
11. Calculate Context Recall.
12. Change top-k from 3 to 5.
13. Compare both experiments.
14. Identify at least three retrieval failures.
15. Identify at least two generation failures.

---

## 13. Module 8 Completion Checklist

- [ ] Understand real RAG execution.
- [ ] Implement retrieval.
- [ ] Implement generation.
- [ ] Capture evaluation records.
- [ ] Run RAGAS metrics.
- [ ] Compare retrieval configurations.
- [ ] Identify retrieval failures.
- [ ] Identify generation failures.

---

## 14. RAGAS Learning Progress

**8 / 10 modules completed — 80%**

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
| 9 | Interpret Scores and Improve RAG | Next |
| 10 | CI/CD and Production Monitoring | Pending |

---

## Next Module: Module 9 — Interpret Evaluation Scores and Improve RAG

You will learn:

- How to analyze RAGAS scores systematically.
- How to identify the weakest component.
- How to improve retrieval and generation separately.
- How to perform controlled experiments.
- How to compare baseline vs improved RAG.
- How to choose the best RAG configuration using evaluation results.
- How to avoid improving one metric while damaging another.
- How to prepare a RAG improvement report.

**After Module 9: 90% completion.**

