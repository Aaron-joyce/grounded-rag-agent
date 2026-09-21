# RAG Engine Evaluation Results
## Benchmark Summary
| Metric | Score | Target |
| :--- | :--- | :--- |
| **Retrieval Recall @ 5** | **100.0%** (4/4) | ≥ 85.0% |
| **Refusal Accuracy** | **100.0%** (2/2) | 100.0% |
| **Avg Query Latency** | **12302.90 ms** | N/A |
---
## Detailed Test Case Results
| ID | Type | Question | Expected Source / Action | Retrieved Sources | Latency (ms) | Status |
|---|---|---|---|---|---|---|
| 1 | Standard | How do I install this project? | `api.md` | `api.md, providence-website-design-document.pdf, providence.pdf` | 2958.43 ms | **PASS** |
| 2 | Standard | What Python version and prerequisites are required? | `api.md` | `api.md, providence-website-design-document.pdf, providence.pdf` | 2472.65 ms | **PASS** |
| 3 | Standard | what is providence in 5 lines? | `providence.pdf` | `providence-website-design-document.pdf, providence.pdf` | 3574.85 ms | **PASS** |
| 4 | Standard | What is the market provdence is trying to capture in 5 lines? | `providence.pdf` | `providence-website-design-document.pdf, providence.pdf` | 4999.52 ms | **PASS** |
| 5 | Trick | What was Apple's total revenue in 2024? | `REFUSAL` | `providence-website-design-document.pdf, providence.pdf` | 4266.52 ms | **PASS** |
| 6 | Trick | How do I configure multi-region Kubernetes clusters on AWS? | `REFUSAL` | `providence-website-design-document.pdf, providence.pdf` | 55545.46 ms | **PASS** |
