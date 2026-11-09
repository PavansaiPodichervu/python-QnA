What is RAGAS?
RAGAS stands for Retrieval Augmented Generation Assessment (commonly referred to as RAG Assessment).

It is an evaluation framework used to measure the quality of Retrieval-Augmented Generation (RAG) applications.

WHy we need the RGAS ?
=========================
Suppose you build a RAG chatbot using:
- FastAPI
- LangChain
- FAISS / pgvector
- OpenAI / Claude
- Your company documents

Your chatbot answers:
What is the company's leave policy?

But the answer may be incorrect, even when your application technically works.

We need to evaluate:
1. Did retrieval fetch the correct documents?
2. Did the LLM generate an answer supported by those documents?
3. Is the answer actually relevant to the user's question?
4. Did the LLM hallucinate?
5. Is the answer complete and correct?

=> RAGAS helps evaluate these questions systematically.

RAGAS evaluation metrics
==========================
RAGAS evaluates different dimensions of RAG quality using metrics. 

Metric	                      What it measures
========================================================================
Faithfulness	          Is the answer supported by retrieved context?
Answer Relevancy          Does the answer address the user's question?
Context Precision	      Are the retrieved documents relevant and ranked properly?
Context Recall	          Did retrieval capture all information required for the correct answer?
Factual Correctness	      Is the generated answer factually correct compared with a reference answer?

Example
User question:
What is the maximum leave allowed per year?

Retrieved context:
Employees receive 24 annual leaves.

Generated answer:
Employees receive 24 annual leaves and 10 sick leaves.

Assume the retrieved context mentions only 24 annual leaves.
- Faithfulness: Low, because sick leaves were invented.
- Answer relevancy: Potentially high, because the answer addresses annual leave.
- Context precision: Depends on how relevant the retrieved document is.
- Context recall: Depends on whether the retrieved context covers the reference answer.
This demonstrates why one metric is not enough to evaluate RAG.