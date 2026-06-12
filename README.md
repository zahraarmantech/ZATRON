# Stop choosing between smart search and private data.

### ZATRON gives you both — 98% retrieval quality on 626K documents, and the server can't read a thing.

A neural network trained on 80,000 labeled pairs couldn't recover document similarity from ZATRON's barcodes (AUC 0.505 — chance level), while the same attack broke unprotected embeddings almost perfectly (AUC 0.999).

```bash
pip install zatron
```

**[▶ Live Demo](https://huggingface.co/spaces/zahraarman/ZATRON)** · **[PyPI](https://pypi.org/project/zatron/)** · **[Benchmarks](https://github.com/zahraarmantech/ZATRON/tree/main/benchmarks)** · Patent Pending

*ZATRON (Zero-Access Transformed Retrieval Over Noise) turns embeddings into modular barcodes. Search still works; structure disappears. Private by math, not by promise.*

---

## The Problem

Standard semantic search stores embeddings as plain vectors. Anyone with database access can cluster documents by topic and infer content — without reading a single word.

## The Solution

ZATRON transforms embeddings into modular barcodes. Search still works. Structure disappears.

![ZATRON Comparison](zatron_comparison.png)

**Left:** raw embeddings — same-topic documents cluster together. An attacker immediately sees the structure.

**Right:** ZATRON protected — random scatter. No topic structure visible.

---

## Attack Analysis

Can an observer recover document similarity from ZATRON barcodes?

![Attack Analysis](zatron_attack.png)

**Left:** raw embedding distances perfectly correlate with true similarity (ρ = 1.00). Attacker wins.

**Right:** ZATRON barcode distances show zero correlation (ρ = 0.09). Attacker gets nothing.

---

## Results

All numbers on real data. All reproducible.

### Retrieval Quality

| Benchmark | Corpus | Quality (% of cosine) |
|-----------|--------|-----------------------|
| MSMARCO | 626,906 passages | 98.2% |
| Natural Questions | 5,000 passages, 100 real queries | 101.6% |
| SciFact | 5,183 scientific docs | 95.7% |
| NFCorpus | 3,633 medical docs | 89.9% |
| STS-B | 1,379 sentence pairs | 100.1% |

### Real-World Challenge Tests

| Dataset | What it is | Security (ρ) | Top-10 Accuracy |
|---------|-----------|-------------|-----------------|
| **Enron Emails** | **350,652 real corporate emails** | **0.034 ✓** | **98.0%** |
| Natural Questions | 5,000 passages, real Google queries | 0.031 ✓ | 100% |
| MSMARCO | 626,906 web passages | 0.10 ✓ | 98.2% MRR |

### Enron Corporate Email Archive — Full Results

Privacy-preserving retrieval on 350,652 real corporate emails from the Enron leak. This is a stress test: corporate emails have extremely high content similarity (forwarded threads, replies, meeting updates), making retrieval harder than typical document search.

|  | Cosine Baseline | ZATRON | Retention |
|--|----------------|--------|-----------|
| Top-1 | 100% | 55.5% | 55.5% |
| Top-5 | 100% | 95.0% | 95.0% |
| Top-10 | 100% | 98.0% | 98.0% |
| MRR@10 | 100% | 72.5% | 72.5% |
| Security (ρ) | — | 0.034 | ✓ SAFE |

The correct result appears in the top 10 results 98% of the time. Top-1 is lower due to near-duplicate emails swapping ranks — not information loss.

### Comparison with Existing Methods

| Method | MRR@10 | Encrypted? |
|--------|--------|------------|
| Cosine (float32) | .530 | No |
| Binary quantization | .514 | No |
| Product quantization | .520 | No |
| **ZATRON (ours)** | **.528** | **Yes** |

**8× faster** than CKKS FHE on identical hardware (5ms vs 39ms per comparison).

Three embedding models. Five languages. Eight security tests passed.

### Multilingual

| Language | Quality |
|----------|---------|
| Arabic | 93.4% |
| Spanish | 93.8% |
| Korean | 95.1% |
| Chinese | 95.4% |
| English | 93.5% |

### Speed Optimization: Coarse-to-Fine Search

Brute-force comparison against all documents is slow at scale. We developed a two-stage approach inspired by radio tuning — scan coarsely first, then fine-tune on candidates:

**Stage 1 (Coarse):** Compute distance using only 2 of 6 primes across all documents. This is one-third the computation and filters out 95% of irrelevant documents.

**Stage 2 (Fine):** Compute full 6-prime distance on the remaining 5% of candidates.

| Dataset | Brute Force | Radio Tuning | Speedup | Quality Loss |
|---------|------------|-------------|---------|-------------|
| MSMARCO 626K | 2399s | 957s | 2.5× | Zero |
| Enron 5K | 2.1s | 0.8s | 2.6× | Zero |

Top-10 accuracy: 100% on both datasets. The coarse filter never drops the correct result.

No stored index or bucket structure. Zero additional leakage — the optimization happens at query time by the key holder.

At 1M+ documents, the speedup grows to an estimated 5–10×, since the fine stage (5% of corpus) dominates.

## Try It

**Install:**
```bash
pip install zatron
```

**Live demo (no install):**
[https://huggingface.co/spaces/zahraarman/ZATRON](https://huggingface.co/spaces/zahraarman/ZATRON)

**Run the demo locally:**
```bash
pip install zatron sentence-transformers matplotlib
python demo.py
```

## Quick Start

```python
from zatron import ModularBarcodeSystem

system = ModularBarcodeSystem(key="your-secret-key", n_channels=200)
system.fit(corpus_embeddings)

barcodes = system.encode(corpus_embeddings, doc_ids)
query_bc = system.encode_query(query_embedding)
distance = system.compare(query_bc, barcodes[0])
```

## How It Works

1. **Decompose**: Project embedding onto 200 PCA channels
2. **Quantize**: Convert each channel to integer (0–49)
3. **Mask**: Apply rejection-sampled salt + wave interference per document
4. **Store**: Keep only modular residues (mod prime)
5. **Search**: Compare in modular space — raw embedding never reconstructed

## Security

### Learned Inversion Attack (Neural Attacker)

Statistical tests only rule out simple attackers. The stronger question: can a **neural network**, trained on labeled examples, learn to recover similarity from ZATRON barcodes?

**Threat model.** Known-plaintext observer: the attacker sees stored barcodes and obtains ~80,000 document pairs with true cosine similarities (a partial plaintext leak). A linear probe and a 3-layer MLP are trained on per-prime circular-difference features to predict similarity of unseen pairs. Train/test pairs share no anchor documents.

**Result (MSMARCO, 50,000 passages, 100,000 pairs):**

| Input the attacker sees | Linear probe | MLP (3-layer) |
|---|---|---|
| Unprotected quantized signals (control) | ρ = 0.79, AUC = 0.985 | **ρ = 0.90, AUC = 0.999** |
| ZATRON masked barcodes | ρ = 0.00, AUC = 0.498 | **ρ = 0.00, AUC = 0.505** |

The identical attack that almost perfectly recovers similarity from unprotected signals performs **exactly at chance level** against ZATRON barcodes — with 80,000 labeled training pairs at its disposal.

Reproduce: `benchmarks/zatron_ML_attack.py` (Colab, ~15 min on T4).

### Head-to-Head: ZATRON vs ASPE

ASPE (Wong et al., SIGMOD 2009) is the classic encrypted-kNN baseline. It preserves scalar products exactly — perfect retrieval — but that same property means **any observer can compute document similarities directly from the ciphertexts**, no key and no training required.

| | ASPE (SIGMOD '09) | ZATRON |
|---|---|---|
| Retrieval recall@10 (strict set-overlap) | 100% | 81.4% |
| Observer computes similarity directly | **ρ = +0.87 (leaks)** | ρ = −0.06 (nothing) |
| Learned attack (MLP, 80K labeled pairs) | **ρ = +0.91, AUC = 0.993** | ρ = +0.01, AUC = 0.518 |

ASPE's perfect recall is bought with total similarity leakage — by design. ZATRON trades a margin on the strictest retrieval metric for chance-level leakage under both direct and learned attacks. (Strict set-overlap is harder than the top-1-in-top-10 metric used in the headline table; same system, stricter ruler. MSMARCO 50,000 passages, identical conditions for both schemes.)

Reproduce: `benchmarks/zatron_vs_ASPE.py` (Colab, ~10 min on T4).

### Classical Attack Battery

Eight independent attack vectors tested:

| Attack | Result | Status |
|--------|--------|--------|
| IND-CPA indistinguishability | p = 0.48 | Pass |
| Statistical correlation | ρ = 0.10 | Pass |
| Entropy analysis | 100% | Pass |
| Per-channel leakage | \|r\| = 0.30 | Pass |
| Key recovery | 1.0% vs 1.9% baseline | Pass |
| Chosen-plaintext | ρ = 0.00 | Pass |
| Timing side-channel | p = 1.00 | Pass |
| CRT reconstruction | \|r\| = 0.01 | Pass |

**Threat model**: Protected against unauthorized database observers — including learned (neural) attackers with known-plaintext training data, per the table above. The key holder computes distances but never reconstructs raw embeddings; a key holder computing many pairwise distances can still partially recover embedding geometry via MDS (ρ ≈ 0.35 after the log transform), which is inherent to any distance-preserving scheme, FHE included. This is a randomized privacy-preserving encoding, distinct from reversible block cipher encryption. Independent cryptographic review remains the appropriate bar for production use.

Formal proofs under PRF assumption (HMAC-SHA256) in `paper/Formal_Security_Proof.pdf`.

## Project Structure

```
ZATRON/
├── README.md
├── zatron_search.py              # Core system (self-testing)
├── demo.py                       # One-command demo
├── generate_visuals.py           # Generate comparison images
├── zatron_comparison.png         # t-SNE visualization
├── zatron_attack.png             # Attack analysis visualization
├── benchmarks/
│   ├── zatron_ML_attack.py       # Learned inversion attack (reproduce security table)
│   └── zatron_vs_ASPE.py         # Head-to-head vs ASPE (SIGMOD'09)
├── demo/
│   └── encrypted_search_demo.jsx # Interactive web demo
├── paper/
│   ├── Lightweight_Encrypted_Semantic_Search.pdf
│   └── Formal_Security_Proof.pdf
└── LICENSE
```

## Cite

```
@misc{arman2026zatron,
  title={Lightweight Encrypted Semantic Search via Multi-Channel Modular Signaling},
  author={Zahra Arman},
  year={2026},
  note={US Provisional Patent Filed. github.com/zahraarmantech/ZATRON}
}
```

## License

MIT License. The method is covered by a pending US provisional patent.

## Author

**Zahra Arman** — Independent Researcher — zahra.arman.tech@gmail.com
