## Speculative RAG

This repo implements **Speculative RAG**, a two-stage RAG framework from [this paper (ICLR 2025)](https://arxiv.org/pdf/2407.08223). It combines:

- **Drafting**: A small LM generates answer drafts from different document subsets.
- **Verification**: A larger LM selects the best draft using confidence and rationale.

Speculative RAG improves accuracy and reduces latency without additional tuning.

## Paper

Wang *et al.* (2024). *Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting*. arXiv:2407.08223   
[Read the full paper →](https://arxiv.org/abs/2407.08223)