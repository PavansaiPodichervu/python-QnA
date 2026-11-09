# Module 2 — Faithfulness: Detect Hallucinations

## Learning Objectives

By completing this module, you will understand:

1. What Faithfulness is.
2. Faithfulness vs Factual Correctness.
3. How hallucinations occur in RAG.
4. Claim extraction.
5. Claim verification against retrieved context.
6. Faithfulness score calculation.
7. High and low Faithfulness examples.
8. Python implementation.
9. Practical hallucination evaluation using RAGAS.

---

# 1. What Is Faithfulness?

## Definition

**Faithfulness is an evaluation metric that measures whether the claims made by an LLM-generated answer are supported by the retrieved context.**

In RAG applications, the LLM is expected to generate answers based on retrieved documents.

Faithfulness evaluates whether the generated answer remains grounded in those documents.

A faithful answer does not introduce unsupported claims.

## Example

### Retrieved Context

```text
Employees receive 20 annual vacation days.
Employees receive 10 sick leave days.
```

### User Question

```text
How many annual vacation days do employees receive?
```

### Faithful Answer

```text
Employees receive 20 annual vacation days.
```

### Unfaithful Answer

```text
Employees receive 30 annual vacation days.
```

The second answer contradicts the retrieved context.

## Technical Definition

Let:

- C = Retrieved Context
- A = Generated Answer
- S = Claims extracted from A
- Supported(S, C) = Claims supported by C

Faithfulness measures the proportion of generated claims supported by the retrieved context.

## Why Faithfulness Matters

Faithfulness helps identify:

- Unsupported information
- Context contradictions
- Hallucinations
- Incorrect use of retrieved documents
- Answers that introduce invented facts

## Key Understanding

**Faithfulness evaluates the relationship between the generated answer and retrieved context, not merely whether the answer sounds correct.**

---

# 2. Faithfulness vs Factual Correctness

These two concepts are related but different.

## 2.1 Faithfulness

Faithfulness checks whether the answer is supported by retrieved context.

## 2.2 Factual Correctness

Factual Correctness evaluates whether the generated answer agrees with factual reference information or a reference answer.

## Example

### Retrieved Context

```text
The company provides 20 annual vacation days.
```

### Actual Company Policy

```text
The company provides 25 annual vacation days.
```

### Generated Answer

```text
The company provides 20 annual vacation days.
```

Evaluation:

| Evaluation | Result |
|---|---|
| Faithfulness | High |
| Factual Correctness | Low |

The answer follows the retrieved context but the retrieved context itself is outdated.

## Another Example

### Retrieved Context

```text
The company provides 20 annual vacation days.
```

### Generated Answer

```text
The company provides 20 annual vacation days and 10 sick leave days.
```

Suppose the actual company policy confirms both statements.

| Evaluation | Result |
|---|---|
| Faithfulness | Depends on whether both claims are supported by retrieved context |
| Factual Correctness | High, assuming the reference confirms both claims |

The sick leave statement may be factually correct but unsupported by the retrieved context.

## Comparison Table

| Faithfulness | Factual Correctness |
|---|---|
| Checks answer against retrieved context | Checks answer against reference information |
| Detects unsupported claims | Detects factual errors relative to reference |
| Does not require reference answer | Typically requires reference information |
| Measures grounding | Measures correctness |
| Helps identify context misuse | Helps identify incorrect information |

## Important Interview Point

A high Faithfulness score does not guarantee a factually correct answer.

An answer can be faithful to incorrect retrieved documents.

---

# 3. How Hallucinations Occur in RAG

## Definition

**Hallucination is generated information that is false, fabricated, or unsupported by available evidence.**

In RAG, hallucinations can occur even when retrieval is implemented correctly.

## RAG Architecture

```text
User Question
     |
     v
Retriever
     |
     v
Retrieved Documents
     |
     v
Prompt Construction
     |
     v
LLM
     |
     v
Generated Answer
```

## Common Causes

### 3.1 Incorrect Retrieval

The retriever selects irrelevant or incorrect documents.

Example:

Question:

```text
What is the annual leave policy?
```

Retrieved Context:

```text
Employees receive 10 sick leave days.
```

Generated Answer:

```text
Employees receive 10 annual vacation days.
```

The model incorrectly uses sick leave information.

### 3.2 Missing Context

The retriever does not provide the required information.

Retrieved Context:

```text
Employees can apply for leave through the HR portal.
```

Generated Answer:

```text
Employees receive 25 annual vacation days.
```

The number was not present in context.

### 3.3 Context Misinterpretation

The LLM incorrectly interprets retrieved information.

Retrieved Context:

```text
Employees receive 20 annual vacation days and 10 sick leave days.
```

Generated Answer:

```text
Employees receive 30 annual vacation days.
```

The model combines two different leave categories.

### 3.4 Contradictory Documents

Retrieved Context:

```text
Document A: Employees receive 20 annual vacation days.

Document B: Employees receive 25 annual vacation days.
```

The LLM may select the wrong value or combine information incorrectly.

### 3.5 Model Knowledge Leakage

The LLM uses its internal learned knowledge instead of relying on retrieved documents.

This can be especially problematic when:

- Company policies change.
- Documents contain domain-specific information.
- The model's prior knowledge is outdated.
- The question requires private knowledge unavailable during training.

## Hallucination Prevention Techniques

- Improve document retrieval.
- Improve chunking.
- Add reranking.
- Use clear system prompts.
- Explicitly instruct the LLM to abstain when context is insufficient.
- Use source citations.
- Evaluate Faithfulness.
- Validate generated answers.

**Faithfulness evaluation detects grounding failures; it does not itself prevent hallucinations.**

---

# 4. Claim Extraction

## Definition

**Claim extraction is the process of breaking a generated answer into individual factual statements that can be independently evaluated.**

A claim is a statement that expresses information that can be verified against evidence.

RAGAS uses an LLM to identify claims from generated answers as part of Faithfulness evaluation.

## Why Claim Extraction Is Required

Consider this answer:

```text
Employees receive 20 annual vacation days,
10 sick leave days, and unused vacation
days cannot be carried forward.
```

This answer contains multiple factual claims.

If we evaluate the answer as a single sentence, identifying individual unsupported statements becomes difficult.

Claim extraction allows each statement to be checked independently.

## Example 1: Simple Claim Extraction

Generated Answer:

```text
The company provides 20 annual vacation days.
```

Extracted Claim:

```text
Claim 1: The company provides 20 annual vacation days.
```

Total Claims = 1

## Example 2: Multiple Claims

Generated Answer:

```text
The company provides 20 annual vacation days.
Employees also receive 10 sick leave days.
Unused annual vacation days cannot be carried forward.
```

Extracted Claims:

```text
Claim 1: The company provides 20 annual vacation days.

Claim 2: Employees receive 10 sick leave days.

Claim 3: Unused annual vacation days cannot be carried forward.
```

Total Claims = 3

## Example 3: Complex Sentence

Generated Answer:

```text
The company provides 20 annual vacation days,
and employees receive 10 sick leave days.
```

Extracted Claims:

```text
Claim 1: The company provides 20 annual vacation days.

Claim 2: Employees receive 10 sick leave days.
```

Total Claims = 2

## Important Concept

Claim extraction converts natural-language answers into smaller factual statements.

These statements are then evaluated against retrieved context.

---

# 5. Claim Verification Against Retrieved Context

## Definition

**Claim verification determines whether an extracted claim can be supported or inferred from retrieved context.**

Each extracted claim is checked independently.

## Example

### Retrieved Context

```text
Employees receive 20 annual vacation days.

Employees receive 10 sick leave days.

Unused annual vacation days cannot be carried forward.
```

### Generated Answer

```text
Employees receive 20 annual vacation days.

Employees receive 15 sick leave days.

Unused annual vacation days cannot be carried forward.
```

### Claim Extraction

```text
Claim 1:
Employees receive 20 annual vacation days.

Claim 2:
Employees receive 15 sick leave days.

Claim 3:
Unused annual vacation days cannot be carried forward.
```

### Verification

| Claim | Context Evidence | Result |
|---|---|---|
| 20 annual vacation days | Explicitly stated | Supported |
| 15 sick leave days | Context says 10 days | Unsupported |
| Vacation cannot be carried forward | Explicitly stated | Supported |

Supported Claims = 2

Total Claims = 3

## Verification Categories

For learning purposes, we can classify claims as:

- Supported
- Unsupported
- Contradicted

RAGAS Faithfulness fundamentally measures whether claims are supported by context.

A contradicted claim is not supported and therefore negatively affects Faithfulness.

## Important Distinction

Unsupported does not always mean false.

Example:

Context:

```text
Employees receive 20 annual vacation days.
```

Answer:

```text
Employees receive 10 sick leave days.
```

The answer may be correct according to company policy, but the retrieved context does not establish it.

Therefore, the claim is unsupported by the provided context.

---

# 6. Faithfulness Score Calculation

## Formula

The Faithfulness score is calculated as:

Faithfulness = Supported Claims / Total Claims

The score ranges from 0 to 1.

- 1.0 = All evaluated claims are supported.
- 0.0 = No evaluated claims are supported.
- Values between 0 and 1 = Some claims are supported.

## Example 1: Perfect Faithfulness

Retrieved Context:

```text
Employees receive 20 annual vacation days.
```

Generated Answer:

```text
Employees receive 20 annual vacation days.
```

Claims:

```text
Claim 1: Employees receive 20 annual vacation days.
```

Supported Claims = 1

Total Claims = 1

Faithfulness = 1 / 1 = 1.0

## Example 2: Partial Faithfulness

Retrieved Context:

```text
Employees receive 20 annual vacation days.

Employees receive 10 sick leave days.
```

Generated Answer:

```text
Employees receive 20 annual vacation days.

Employees receive 15 sick leave days.
```

Claims:

```text
Claim 1: Employees receive 20 annual vacation days.

Claim 2: Employees receive 15 sick leave days.
```

Supported Claims = 1

Total Claims = 2

Faithfulness = 1 / 2 = 0.50

## Example 3: Zero Faithfulness

Retrieved Context:

```text
Employees receive 20 annual vacation days.
```

Generated Answer:

```text
Employees receive 30 annual vacation days.
```

Supported Claims = 0

Total Claims = 1

Faithfulness = 0 / 1 = 0.0

## Example 4: Multiple Claims

Retrieved Context:

```text
Employees receive 20 annual vacation days.

Employees receive 10 sick leave days.

Unused annual vacation days cannot be carried forward.
```

Generated Answer:

```text
Employees receive 20 annual vacation days.

Employees receive 10 sick leave days.

Unused annual vacation days can be carried forward.

Employees receive 5 personal leave days.
```

| Claim | Supported? |
|---|---|
| 20 annual vacation days | Yes |
| 10 sick leave days | Yes |
| Vacation can be carried forward | No |
| 5 personal leave days | No |

Supported Claims = 2

Total Claims = 4

Faithfulness = 2 / 4 = 0.50

## Important Note

The formula is conceptually simple, but RAGAS uses an LLM-based evaluation process to extract and verify claims.

The actual score depends on the judge's claim decomposition and verification decisions.

Source: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/

---

# 7. High and Low Faithfulness Examples

## Example 1: High Faithfulness

Context:

```text
The first Super Bowl was played on January 15, 1967.
```

Answer:

```text
The first Super Bowl was held on January 15, 1967.
```

Score:

```text
Faithfulness = 1.0
```

Reason:

The answer is directly supported by context.

## Example 2: Low Faithfulness

Context:

```text
The first Super Bowl was played on January 15, 1967.
```

Answer:

```text
The first Super Bowl was held on January 20, 1967.
```

Score:

```text
Faithfulness = 0.0
```

Reason:

The date contradicts the retrieved context.

## Example 3: Partial Faithfulness

Context:

```text
The first Super Bowl was played on January 15, 1967.
It took place at the Los Angeles Memorial Coliseum.
```

Answer:

```text
The first Super Bowl was played on January 15, 1967,
at the Los Angeles Memorial Coliseum,
in front of 100,000 spectators.
```

Claims:

```text
Claim 1: The first Super Bowl was played on January 15, 1967.

Claim 2: It took place at the Los Angeles Memorial Coliseum.

Claim 3: It was attended by 100,000 spectators.
```

The first two claims are supported.

The third claim is unsupported.

Faithfulness:

```text
2 / 3 = 0.667
```

## Example 4: Correct Answer but Low Faithfulness

Context:

```text
The company was founded in 2010.
```

Answer:

```text
The company was founded in 2010 and is headquartered in Bengaluru.
```

Suppose the company is actually headquartered in Bengaluru.

The answer may be factually correct, but the retrieved context only supports the founding year.

Therefore:

```text
Faithfulness = 0.5
```

This demonstrates:

**Faithfulness is context-grounded evaluation, not unrestricted fact checking.**

---

# 8. Python Implementation

We will implement Faithfulness in two ways:

1. Manual Faithfulness calculation.
2. RAGAS Faithfulness evaluation.

## 8.1 Manual Python Implementation

This implementation demonstrates the calculation using manually verified claims.

It is not an automatic hallucination detector.

### Code

```python
# Retrieved context
context = """
Employees receive 20 annual vacation days.
Employees receive 10 sick leave days.
Unused annual vacation days cannot be carried forward.
"""

# Generated answer
answer = """
Employees receive 20 annual vacation days.
Employees receive 15 sick leave days.
Unused annual vacation days cannot be carried forward.
"""

# Claims extracted manually
claims = [
    {
        "claim": "Employees receive 20 annual vacation days.",
        "supported": True
    },
    {
        "claim": "Employees receive 15 sick leave days.",
        "supported": False
    },
    {
        "claim": "Unused annual vacation days cannot be carried forward.",
        "supported": True
    }
]

# Calculate faithfulness
total_claims = len(claims)

supported_claims = sum(
    claim["supported"]
    for claim in claims
)

faithfulness = (
    supported_claims / total_claims
    if total_claims > 0
    else None
)

print("Total Claims:", total_claims)
print("Supported Claims:", supported_claims)
print("Faithfulness:", faithfulness)
```

### Expected Output

```text
Total Claims: 3
Supported Claims: 2
Faithfulness: 0.6666666666666666
```

### Explanation

- We manually extracted three claims.
- We manually verified two claims.
- We calculated the supported claim ratio.
- The score is approximately 0.667.

This implementation teaches the scoring logic.

---

## 8.2 Install RAGAS

Use a virtual environment.

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -U ragas openai
```

Configure the OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

For Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-api-key"
```

## 8.3 RAGAS Faithfulness Implementation

This example uses the current collections-based Faithfulness API documented by RAGAS.

### Code

```python
import asyncio
import os

from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness


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

    # 3. Create Faithfulness metric
    faithfulness_metric = Faithfulness(
        llm=evaluator_llm
    )

    # 4. Define user question
    question = (
        "How many annual vacation days "
        "do employees receive?"
    )

    # 5. Define retrieved context
    retrieved_contexts = [
        "Employees receive 20 annual vacation days.",
        "Employees receive 10 sick leave days."
    ]

    # 6. Define generated answer
    answer = (
        "Employees receive 20 annual vacation days "
        "and 15 sick leave days."
    )

    # 7. Evaluate Faithfulness
    result = await faithfulness_metric.ascore(
        user_input=question,
        response=answer,
        retrieved_contexts=retrieved_contexts
    )

    # 8. Print result
    print("Faithfulness Score:", result.value)


if __name__ == "__main__":
    asyncio.run(main())
```

### Expected Interpretation

The answer contains:

```text
Claim 1: Employees receive 20 annual vacation days.

Claim 2: Employees receive 15 sick leave days.
```

The context supports the first claim but contradicts the second.

Expected score is approximately:

```text
Faithfulness Score: 0.5
```

The exact score may vary depending on how the evaluation model extracts and verifies claims.

### Important

The evaluation LLM is separate from the LLM used by your RAG application.

For example:

```text
RAG LLM:
Claude / GPT / Other model

Evaluation LLM:
GPT-4o-mini / Other supported evaluator
```

The evaluation model judges the answer generated by your application.

---

# 9. Practical Hallucination Evaluation

Now let's evaluate multiple answers systematically.

## 9.1 Test Dataset

```python
test_cases = [
    {
        "question": "How many annual vacation days do employees receive?",
        "context": [
            "Employees receive 20 annual vacation days."
        ],
        "answer": "Employees receive 20 annual vacation days.",
        "expected_faithfulness": 1.0
    },
    {
        "question": "How many annual vacation days do employees receive?",
        "context": [
            "Employees receive 20 annual vacation days."
        ],
        "answer": "Employees receive 30 annual vacation days.",
        "expected_faithfulness": 0.0
    },
    {
        "question": "What leave benefits are provided?",
        "context": [
            "Employees receive 20 annual vacation days.",
            "Employees receive 10 sick leave days."
        ],
        "answer": (
            "Employees receive 20 annual vacation days "
            "and 15 sick leave days."
        ),
        "expected_faithfulness": 0.5
    }
]
```

## 9.2 Evaluation Strategy

For each test case:

1. Read the question.
2. Retrieve context.
3. Generate the answer.
4. Extract answer claims.
5. Verify claims against context.
6. Calculate Faithfulness.
7. Record the score.
8. Analyze unsupported claims.

## 9.3 Evaluation Report

Example report:

| Test Case | Faithfulness | Result |
|---|---:|---|
| Correct answer | 1.00 | Pass |
| Incorrect annual leave | 0.00 | Fail |
| Partially unsupported answer | 0.50 | Fail |

## 9.4 Finding the Root Cause

Suppose the score is low.

Investigate:

```text
Question
   |
   v
Retrieved Context
   |
   v
Does Context Contain Required Information?
   |
   +-- No --> Retrieval Failure
   |
   +-- Yes
         |
         v
Does Answer Follow Context?
         |
         +-- No --> Generation / Prompt Failure
         |
         +-- Yes --> Faithful Answer
```

This helps identify whether the issue is retrieval or generation.

## 9.5 Improving Faithfulness

### Retrieval Improvements

- Improve chunking.
- Improve embeddings.
- Adjust top-K.
- Add reranking.
- Remove duplicate chunks.
- Improve document quality.

### Generation Improvements

- Improve system prompts.
- Require answers to use retrieved context.
- Tell the model not to invent missing information.
- Require uncertainty when evidence is insufficient.
- Use structured outputs where appropriate.
- Provide source citations.

### Evaluation Improvements

- Use representative evaluation datasets.
- Review low-scoring examples.
- Validate LLM judge outputs.
- Compare evaluation scores before and after changes.
- Include human review for high-impact domains.

---

# 10. Faithfulness Best Practices

1. Evaluate generated answers against the exact retrieved context used by the application.
2. Do not assume a high Faithfulness score means the answer is globally correct.
3. Keep evaluation datasets representative of real user questions.
4. Review low-scoring examples manually.
5. Evaluate both retrieval and generation independently.
6. Do not use Faithfulness as the only RAG quality metric.
7. Track scores across model and prompt changes.
8. Use reference-based metrics when factual correctness matters.
9. Treat LLM judge outputs as estimates, not perfect ground truth.
10. Use human validation for critical decisions.

---

# 11. Module 2 Interview Questions

## Beginner

1. What is Faithfulness in RAGAS?
2. Why is Faithfulness important?
3. What is a claim?
4. What is claim extraction?
5. How is Faithfulness calculated?
6. What is a hallucination?
7. Can a correct answer have low Faithfulness?

## Intermediate

8. How does RAGAS evaluate Faithfulness?
9. What is the role of the evaluation LLM?
10. Why is Faithfulness different from Factual Correctness?
11. How can retrieval failures cause hallucinations?
12. How can prompt engineering improve Faithfulness?
13. Can Faithfulness be evaluated without reference answers?
14. What are the limitations of LLM-based Faithfulness evaluation?

## Advanced

15. How would you design a hallucination evaluation pipeline?
16. How would you investigate low Faithfulness scores in production?
17. How would you compare Faithfulness across different LLMs?
18. How would you validate the reliability of an LLM judge?
19. How would you improve Faithfulness without increasing retrieval latency significantly?
20. Why should Faithfulness and Context Recall be evaluated together?

---

# 12. Practical Exercises

## Exercise 1: Calculate Faithfulness

Context:

```text
The product costs ₹10,000.

The product has a 2-year warranty.

The product is available in black and white.
```

Answer:

```text
The product costs ₹10,000.

It has a 3-year warranty.

It is available in black and white.

It is available in blue.
```

Tasks:

1. Extract claims.
2. Identify supported claims.
3. Identify unsupported claims.
4. Calculate Faithfulness.

## Exercise 2: Identify Hallucination Type

Context:

```text
The company was founded in 2015.
```

Answer:

```text
The company was founded in 2015 in Hyderabad.
```

Tasks:

1. Extract claims.
2. Identify unsupported information.
3. Calculate Faithfulness.
4. Explain why the answer might still be factually correct.

## Exercise 3: RAGAS Implementation

Modify the Python implementation to evaluate:

```text
Question:
Who discovered penicillin?

Context:
Penicillin was discovered by Alexander Fleming in 1928.

Answer:
Penicillin was discovered by Alexander Fleming in 1928.
```

Tasks:

1. Run RAGAS Faithfulness.
2. Record the score.
3. Change the answer to an incorrect discovery year.
4. Run evaluation again.
5. Compare the scores.

---

# 13. Module 2 Completion Checklist

- [ ] What is Faithfulness?
- [ ] Faithfulness vs Factual Correctness
- [ ] How hallucinations occur in RAG
- [ ] Claim extraction
- [ ] Claim verification against retrieved context
- [ ] Faithfulness score calculation
- [ ] High and low Faithfulness examples
- [ ] Manual Python implementation
- [ ] RAGAS Python implementation
- [ ] Practical hallucination evaluation
- [ ] Evaluation result interpretation
- [ ] Root cause analysis
- [ ] Faithfulness improvement techniques
- [ ] Interview questions
- [ ] Practical exercises

## Final Takeaway

Faithfulness is one of the most important metrics for evaluating grounding in RAG applications.

It helps answer:

**"Is my LLM generating answers based on retrieved documents, or is it inventing information?"**

However, Faithfulness alone cannot guarantee factual correctness, retrieval completeness, or overall answer quality.

Use it together with Answer Relevancy, Context Precision, Context Recall, and reference-based evaluation metrics.

---

## Official References

- RAGAS Faithfulness Documentation: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/
- RAGAS GitHub: https://github.com/vibrantlabsai/ragas
- RAGAS Research Paper: https://arxiv.org/abs/2309.15217
