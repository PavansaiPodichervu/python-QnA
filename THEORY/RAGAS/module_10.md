# RAGAS — Module 10: CI/CD and Production Monitoring

## 1. Learning Objectives

After completing this module, you will understand:

- How to automate RAG evaluation.
- How to integrate RAGAS with GitHub Actions.
- How to detect RAG quality regressions.
- How to define evaluation quality thresholds.
- How to monitor RAG applications in production.
- How to track retrieval, generation, latency, and cost.
- How to store evaluation results.
- How to build an end-to-end RAG evaluation workflow.

---

# 2. What Is CI/CD?

**CI — Continuous Integration:** Automatically tests code whenever developers push changes or create pull requests.

**CD — Continuous Delivery/Deployment:** Automates the process of preparing or deploying tested changes to an environment.

For AI applications, traditional tests are not enough.

A RAG application may pass Python tests but generate incorrect answers because:

- The embedding model changed.
- The prompt changed.
- Retrieval configuration changed.
- Documents changed.
- The LLM changed.
- The vector database contains different data.

Therefore, AI applications require **quality evaluation in addition to traditional software testing**.

---

# 3. Traditional Testing vs RAG Evaluation

| Traditional testing | RAG evaluation |
|---|---|
| Test API returns HTTP 200 | Test whether answer is relevant |
| Test function output | Test answer grounding |
| Test database operations | Test retrieval quality |
| Test expected exceptions | Test unsupported-answer behavior |
| Test execution correctness | Test AI response quality |

A production-ready RAG system needs both.

```text
Python Unit Tests
        |
        v
API Integration Tests
        |
        v
RAG Quality Evaluation
        |
        v
Security Tests
        |
        v
Performance Tests
        |
        v
Deployment
```

---

# 4. RAG Evaluation CI/CD Architecture

<AsyncImage query="continuous integration machine learning pipeline diagram GitHub Actions automated testing deployment" aspectRatio="16:9"/>

```text
Developer Changes Code / Prompt
              |
              v
         GitHub Push
              |
              v
      GitHub Actions CI
              |
      +-------+-------+
      |               |
      v               v
 Unit Tests       RAG Evaluation
      |               |
      v               v
 API Tests       RAGAS Metrics
      |               |
      +-------+-------+
              |
              v
       Quality Thresholds
              |
       +------+------+
       |             |
       v             v
    PASS           FAIL
       |             |
       v             v
  Deployment     Block / Review
```

The key idea is:

**A RAG application should not be deployed blindly when its quality regresses.**

---

# 5. Create a RAG Evaluation Project

Example project structure:

```text
rag-project/
│
├── app/
│   ├── rag.py
│   ├── retriever.py
│   └── generator.py
│
├── evaluation/
│   ├── dataset.json
│   ├── evaluate.py
│   ├── thresholds.json
│   └── results/
│
├── tests/
│   ├── test_rag.py
│   └── test_api.py
│
├── requirements.txt
│
└── .github/
    └── workflows/
        └── rag-evaluation.yml
```

### Responsibilities

| File | Purpose |
|---|---|
| `rag.py` | Main RAG pipeline |
| `retriever.py` | Document retrieval |
| `generator.py` | Answer generation |
| `dataset.json` | Evaluation questions |
| `evaluate.py` | RAGAS evaluation |
| `thresholds.json` | Quality acceptance thresholds |
| `results/` | Evaluation reports |
| `rag-evaluation.yml` | GitHub Actions workflow |

---

# 6. Create an Evaluation Dataset

Create `evaluation/dataset.json`.

```json
[
  {
    "user_input": "What is FastAPI?",
    "reference": "FastAPI is a Python framework for building APIs."
  },
  {
    "user_input": "What is LangGraph?",
    "reference": "LangGraph is a framework for building stateful AI agents."
  },
  {
    "user_input": "What is RAG?",
    "reference": "RAG retrieves relevant documents before generating an answer."
  }
]
```

This dataset contains evaluation questions and reference answers.

Your evaluation script should execute the actual RAG application to collect:

- Retrieved contexts.
- Generated responses.

Do not manually invent retrieved contexts for production evaluation.

---

# 7. Define Quality Thresholds

Create `evaluation/thresholds.json`.

```json
{
  "faithfulness": 0.80,
  "answer_relevancy": 0.75,
  "context_precision": 0.70,
  "context_recall": 0.75
}
```

These thresholds are illustrative.

### What is a quality threshold?

A quality threshold is a minimum acceptable metric score.

Example:

```text
Faithfulness = 0.85

Minimum required = 0.80

Result = PASS
```

Another example:

```text
Faithfulness = 0.62

Minimum required = 0.80

Result = FAIL
```

Thresholds should be selected using:

- Baseline performance.
- Business requirements.
- Evaluation dataset quality.
- Metric reliability.
- Acceptable failure rates.

Do not blindly use the same threshold for every RAG application.

---

# 8. Build an Automated Evaluation Script

Create `evaluation/evaluate.py`.

This example assumes the RAG application exposes:

```python
run_rag(question)
```

with this output:

```python
{
    "retrieved_contexts": [...],
    "response": "..."
}
```

### Step 1: Import dependencies

```python
import asyncio
import json
import os

from pathlib import Path

from openai import AsyncOpenAI

from ragas import SingleTurnSample
from ragas.llms import llm_factory

from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall
)

from app.rag import run_rag
```

### Step 2: Load dataset

```python
BASE_DIR = Path(__file__).parent

dataset_path = BASE_DIR / "dataset.json"

with open(dataset_path) as file:
    evaluation_questions = json.load(file)
```

### Step 3: Load thresholds

```python
threshold_path = BASE_DIR / "thresholds.json"

with open(threshold_path) as file:
    thresholds = json.load(file)
```

### Step 4: Execute RAG and collect samples

```python
async def collect_samples():

    samples = []

    for item in evaluation_questions:

        output = run_rag(
            item["user_input"]
        )

        sample = SingleTurnSample(
            user_input=item["user_input"],
            retrieved_contexts=output["retrieved_contexts"],
            response=output["response"],
            reference=item["reference"]
        )

        samples.append(sample)

    return samples
```

### Step 5: Calculate metrics

```python
async def evaluate_samples(samples):

    client = AsyncOpenAI()

    evaluator_llm = llm_factory(
        "gpt-4o-mini",
        client=client
    )

    metrics = {
        "faithfulness": Faithfulness(
            llm=evaluator_llm
        ),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm
        ),
        "context_precision": LLMContextPrecisionWithReference(
            llm=evaluator_llm
        ),
        "context_recall": LLMContextRecall(
            llm=evaluator_llm
        )
    }

    results = {}

    for name, metric in metrics.items():

        scores = await asyncio.gather(
            *[
                metric.single_turn_ascore(sample)
                for sample in samples
            ]
        )

        valid_scores = [
            score.value
            for score in scores
            if score.value is not None
        ]

        if not valid_scores:
            raise RuntimeError(
                f"No valid scores for {name}"
            )

        results[name] = sum(
            valid_scores
        ) / len(valid_scores)

    return results
```

### Step 6: Check quality thresholds

```python
def check_thresholds(results):

    failures = []

    for metric, minimum in thresholds.items():

        actual = results[metric]

        if actual < minimum:

            failures.append(
                f"{metric}: {actual:.3f} "
                f"< minimum {minimum:.3f}"
            )

    return failures
```

### Step 7: Save evaluation report

```python
async def main():

    samples = await collect_samples()

    results = await evaluate_samples(
        samples
    )

    failures = check_thresholds(
        results
    )

    report = {
        "metrics": results,
        "thresholds": thresholds,
        "passed": len(failures) == 0,
        "failures": failures
    }

    output_path = (
        BASE_DIR / "results" / "evaluation.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w") as file:

        json.dump(
            report,
            file,
            indent=2
        )

    print(json.dumps(report, indent=2))

    if failures:

        raise SystemExit(
            "RAG evaluation failed"
        )


if __name__ == "__main__":

    asyncio.run(main())
```

### Important implementation notes

- This script evaluates the actual application.
- Evaluation questions are fixed.
- Thresholds are configurable.
- Evaluation results are saved.
- CI fails if the quality threshold is not met.
- A metric failure should trigger investigation, not automatic assumptions about the root cause.

The exact RAGAS APIs can change between versions. Pin your tested RAGAS version and validate imports against its documentation.

---

# 9. Integrate RAGAS with GitHub Actions

Create:

`.github/workflows/rag-evaluation.yml`

```yaml
name: RAG Quality Evaluation

on:
  pull_request:
  push:
    branches:
      - main

jobs:
  rag-evaluation:

    runs-on: ubuntu-latest

    steps:

      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run unit tests
        run: |
          pytest tests/

      - name: Run RAG evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python -m evaluation.evaluate

      - name: Upload evaluation report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: rag-evaluation-report
          path: evaluation/results/evaluation.json
          if-no-files-found: ignore
```

### Explanation

| Step | Purpose |
|---|---|
| Checkout | Download repository code |
| Setup Python | Configure runtime |
| Install dependencies | Install packages |
| Unit tests | Validate software behavior |
| RAG evaluation | Calculate quality metrics |
| Upload report | Preserve evaluation results |

### Configure API key

In GitHub:

1. Open repository.
2. Navigate to Settings.
3. Open Secrets and variables.
4. Select Actions.
5. Create repository secret.
6. Name it `OPENAI_API_KEY`.
7. Store the API key securely.

Never commit API keys into source code.

---

# 10. Production Monitoring Architecture

CI/CD evaluation tests changes before deployment.

Production monitoring evaluates actual application behavior after deployment.

<AsyncImage query="AI application observability monitoring architecture dashboard metrics logs tracing" aspectRatio="16:9"/>

```text
Production User
      |
      v
   RAG API
      |
      +----------------+
      |                |
      v                v
 Retrieval          Generation
      |                |
      v                v
 Retrieved Docs    Final Answer
      |                |
      +--------+-------+
               |
               v
       Observability Layer
               |
       +-------+--------+
       |       |        |
       v       v        v
     Logs   Metrics   Traces
       |       |        |
       +-------+--------+
               |
               v
        Monitoring DB
               |
               v
       Evaluation Pipeline
               |
               v
        Quality Dashboard
```

---

# 11. What Should You Monitor in Production?

## A. Retrieval metrics

Track:

- Retrieved document count.
- Retrieval latency.
- Retrieval score distribution.
- Empty retrieval rate.
- Document source distribution.
- Retrieval failures.

## B. Generation metrics

Track:

- Generation latency.
- Token usage.
- Model name.
- Prompt version.
- Response length.
- Error rate.
- Refusal rate.

## C. RAG quality metrics

Track:

- Faithfulness.
- Answer Relevancy.
- Context Precision.
- Context Recall.
- User feedback.
- Answer correctness from reviewed samples.

## D. Business metrics

Track:

- User satisfaction.
- Successful task completion.
- Escalation rate.
- Unanswered questions.
- Repeat user queries.

---

# 12. Implement Basic Production Logging

Example:

```python
import logging
import time
import uuid

logger = logging.getLogger("rag_app")


def monitored_rag(question):

    request_id = str(uuid.uuid4())

    start_time = time.perf_counter()

    output = run_rag(question)

    latency = (
        time.perf_counter() - start_time
    )

    logger.info(
        "rag_request_completed",
        extra={
            "request_id": request_id,
            "latency_seconds": latency,
            "retrieved_document_count": len(
                output["retrieved_contexts"]
            )
        }
    )

    return output
```

### Why logging matters

If a user reports a bad answer, you should be able to investigate:

- Which question was asked?
- Which documents were retrieved?
- Which model generated the answer?
- Which prompt version was used?
- How long did retrieval take?
- How long did generation take?

Production logging enables troubleshooting and evaluation.

**Privacy note:** Avoid logging sensitive user data or confidential document contents unnecessarily. Apply access controls, retention limits, and redaction.

---

# 13. Production RAG Evaluation Strategies

There are three practical approaches.

## Strategy 1: Offline Evaluation

Run a fixed evaluation dataset.

Advantages:

- Repeatable.
- Easy to compare.
- Suitable for CI/CD.
- Useful for regression testing.

Disadvantages:

- May not represent actual production questions.
- Requires maintenance of evaluation data.

## Strategy 2: Sampled Production Evaluation

Select a sample of production interactions for evaluation.

Advantages:

- Reflects real user behavior.
- Helps detect new failure patterns.
- Identifies real-world retrieval problems.

Disadvantages:

- Evaluation cost.
- Privacy requirements.
- Potential evaluation latency.

## Strategy 3: Human Review

Human reviewers inspect selected answers.

Advantages:

- Can evaluate nuanced correctness.
- Can identify evaluator mistakes.
- Useful for sensitive or high-impact applications.

Disadvantages:

- Expensive.
- Time-consuming.
- Requires review guidelines.

### Recommended combination

```text
Offline Evaluation
        +
Production Sampling
        +
Human Review
        |
        v
Continuous RAG Improvement
```

---

# 14. Track Quality Over Time

Suppose your weekly evaluation results are:

| Week | Faithfulness | Answer Relevancy | Context Precision |
|---|---:|---:|---:|
| Week 1 | 0.82 | 0.80 | 0.72 |
| Week 2 | 0.85 | 0.84 | 0.76 |
| Week 3 | 0.79 | 0.82 | 0.68 |
| Week 4 | 0.91 | 0.89 | 0.83 |

You can identify:

- Quality regressions.
- Improvements after deployments.
- Retrieval problems after data updates.
- Prompt changes that improve or reduce grounding.
- Model changes that affect answer quality.

Example:

Week 3 Faithfulness decreased.

Possible causes:

- Prompt change.
- Model update.
- Knowledge base update.
- Retrieval configuration change.

You should investigate before assuming the cause.

---

# 15. Evaluation Cost Management

RAG evaluation may invoke LLMs repeatedly.

For example:

```text
100 evaluation questions
        |
        v
4 metrics per question
        |
        v
Potentially hundreds of evaluator operations
```

Actual token usage depends on metric implementation, retries, and batching.

### Cost reduction strategies

1. Use a smaller evaluation dataset for frequent CI runs.
2. Run a larger evaluation dataset nightly.
3. Use production sampling.
4. Avoid reevaluating unchanged data unnecessarily.
5. Track evaluator model usage.
6. Cache deterministic preprocessing results.
7. Separate inexpensive tests from expensive LLM evaluation.

### Suggested evaluation schedule

| Frequency | Evaluation |
|---|---|
| Every pull request | Small regression dataset |
| Every deployment | Full critical dataset |
| Nightly | Larger evaluation dataset |
| Weekly | Production sample review |
| Monthly | Human quality review |

These frequencies are examples; choose based on cost and risk.

---

# 16. CI/CD Quality Gates

A quality gate determines whether a change can proceed.

Example:

```python
quality_gate = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
    "context_recall": 0.75
}
```

Example evaluation:

```python
evaluation_results = {
    "faithfulness": 0.91,
    "answer_relevancy": 0.88,
    "context_precision": 0.81,
    "context_recall": 0.86
}
```

Result:

```text
Faithfulness       PASS
Answer Relevancy   PASS
Context Precision  PASS
Context Recall     PASS

Overall Result     PASS
```

### Regression-based gate

A more advanced approach compares against the baseline.

Example:

```text
Baseline Faithfulness = 0.90

New Faithfulness = 0.82

Regression = -0.08
```

You may decide to block deployment if the regression exceeds an acceptable limit.

This is useful when absolute thresholds alone do not detect meaningful degradation.

---

# 17. Final RAG Evaluation Architecture

```text
                    RAG APPLICATION
                          |
             +------------+------------+
             |                         |
             v                         v
        RAG Runtime              Evaluation Dataset
             |                         |
             v                         v
       User Requests              RAG Execution
             |                         |
             v                         v
       Generated Answers        Retrieved Contexts
             |                         |
             +------------+------------+
                          |
                          v
                     RAGAS
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
     Faithfulness    Relevancy       Retrieval
          |               |               |
          +---------------+---------------+
                          |
                          v
                 Quality Evaluation
                          |
             +------------+------------+
             |                         |
             v                         v
         CI/CD Gate             Monitoring
             |                         |
             v                         v
        Deployment              Improvement
```

---

# 18. Interview Questions and Answers

### Q1. Why is RAG evaluation required in CI/CD?

RAG behavior can change due to prompts, models, embeddings, and retrieval configuration. Automated evaluation detects quality regressions before deployment.

### Q2. What is a RAG quality gate?

A quality gate is a set of conditions that must be satisfied before a RAG application can proceed to deployment.

### Q3. How do you monitor a production RAG application?

Monitor retrieval quality, generation quality, latency, errors, token usage, user feedback, and sampled evaluation results.

### Q4. What is RAG regression testing?

Regression testing checks whether a new change reduces RAG quality compared with a previously validated version.

### Q5. Why do we need production evaluation if CI evaluation already exists?

CI uses controlled datasets. Production evaluation captures real user behavior and new failure patterns that may not exist in the evaluation dataset.

### Q6. What is the difference between offline and online evaluation?

Offline evaluation uses a prepared dataset.

Online evaluation analyzes real application interactions or continuously collected production samples.

### Q7. Can RAGAS run automatically after every code change?

Yes. It can be integrated into CI workflows, provided dependencies, credentials, evaluation data, and quality thresholds are configured.

### Q8. How do you control RAG evaluation costs?

Use smaller CI datasets, larger scheduled evaluations, production sampling, and evaluator usage monitoring.

### Q9. Should every evaluation failure block deployment?

Not necessarily. Critical quality regressions may block deployment, while smaller deviations can trigger warnings or manual review depending on application requirements.

### Q10. What is the complete RAG quality lifecycle?

```text
Build
  |
  v
Evaluate
  |
  v
Analyze
  |
  v
Improve
  |
  v
Regression Test
  |
  v
Deploy
  |
  v
Monitor
  |
  v
Collect Feedback
  |
  v
Evaluate Again
```

---

# 19. Hands-on Assignment

Build a complete evaluation pipeline.

### Tasks

1. Create a RAG application.
2. Create a fixed evaluation dataset.
3. Implement RAGAS metrics.
4. Create evaluation thresholds.
5. Create an evaluation JSON report.
6. Integrate evaluation with GitHub Actions.
7. Configure API keys securely.
8. Make a prompt change.
9. Run CI evaluation.
10. Intentionally introduce a retrieval regression.
11. Observe whether the evaluation detects it.
12. Restore the previous configuration.
13. Add basic production logging.
14. Create a weekly evaluation report.
15. Document baseline and improved scores.

### Final deliverables

```text
rag-project/
    app/
    evaluation/
        dataset.json
        evaluate.py
        thresholds.json
        results/
    tests/
    .github/
        workflows/
            rag-evaluation.yml
    requirements.txt
    README.md
```

---

# 20. Final RAGAS Roadmap Completion

<badge color="success">10 / 10 Modules Completed — 100%</badge>

| Module | Topic | Status |
|---|---|---|
| 1 | RAGAS Fundamentals and Architecture | Completed |
| 2 | Faithfulness | Completed |
| 3 | Answer Relevancy | Completed |
| 4 | Context Precision | Completed |
| 5 | Context Recall | Completed |
| 6 | Build a RAG Evaluation Dataset | Completed |
| 7 | Install RAGAS and Implement Python Evaluation | Completed |
| 8 | Evaluate a Real RAG Application | Completed |
| 9 | Interpret Scores and Improve RAG | Completed |
| 10 | CI/CD and Production Monitoring | Completed |

---

# 21. What You Should Know After This Roadmap

You should now understand how to:

- Evaluate RAG applications.
- Measure hallucinations using Faithfulness.
- Evaluate answer relevance.
- Measure retrieval ranking and completeness.
- Create evaluation datasets.
- Execute RAGAS metrics using Python.
- Evaluate actual RAG pipelines.
- Improve retrieval and generation.
- Compare baseline and improved configurations.
- Automate evaluation using CI/CD.
- Monitor RAG quality in production.

**The next practical step is to build one complete RAG evaluation project rather than studying additional metrics independently.**

Suggested project:

```text
RAG Evaluation Platform

1. Document ingestion
2. Chunking
3. Embeddings
4. Vector database
5. Retrieval
6. LLM generation
7. RAGAS evaluation
8. Evaluation dashboard
9. CI/CD quality gate
10. Production monitoring
```

Reference documentation:

- https://docs.ragas.io/en/latest/
- https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/
- https://docs.ragas.io/en/latest/getstarted/rag_eval/
- https://docs.github.com/en/actions
