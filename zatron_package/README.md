# ZATRON

**Zero-Access Transformed Retrieval Over Noise** — privacy-preserving semantic search via multi-channel modular arithmetic. Search sensitive documents by meaning without exposing content to the database, the server, or even the key holder.

## Install

```bash
pip install zatron
```

## Quick start

```python
from zatron import ModularBarcodeSystem

system = ModularBarcodeSystem(key="your-secret-key", n_channels=200)
system.fit(corpus_embeddings)                      # fit PCA + quantization

barcodes  = system.encode(corpus_embeddings, doc_ids)
query_bc  = system.encode_query(query_embedding)
distance  = system.compare(query_bc, barcodes[0])  # search in modular space
```

A neural attacker trained on 80,000 labeled pairs recovers similarity from
unprotected signals almost perfectly (rho = 0.90, AUC = 0.999) and gets
nothing from ZATRON barcodes (rho = 0.00, AUC = 0.50).

- Demo: https://huggingface.co/spaces/zahraarman/ZATRON
- Code & benchmarks: https://github.com/zahraarmantech/ZATRON

MIT licensed. Method covered by a pending US provisional patent.
