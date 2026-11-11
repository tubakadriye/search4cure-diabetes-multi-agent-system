# Embedding model candidates

Official model cards, configurations and documentation reviewed on November 10,
2026. This is a research shortlist, not an executed model benchmark or a selected
production configuration. No candidate weights were downloaded or models run
during that research. A subsequent [MiniLM/BGE operational comparison](EMBEDDING_COMPARISON.md)
now checks local inference, token coverage and output integrity; retrieval
relevance has not been scored and no replacement has been selected.

The original models and their code paths remain documented in
[MODEL_REVIEW.md](MODEL_REVIEW.md). Embedding models produce retrieval vectors;
the answer-generating model is a separate choice.

## Contents

- [Text candidates](#text-candidates)
- [Image and document-page candidates](#image-and-document-page-candidates)
- [Local and AWS constraints](#local-and-aws-constraints)
- [Proposed first comparison](#proposed-first-comparison)
- [Storage and reproducibility](#storage-and-reproducibility)

## Text candidates

Values below describe published configurations, not locally measured outputs.

| Candidate | Output dimension | Published input configuration | Reason to test | Declared license |
|---|---:|---|---|---|
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | 384 | 512-token sequence; English | Lightweight first challenger to MiniLM for English passage retrieval | MIT |
| [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | Up to 1,024; configurable 32–1,024 | 32K context; 100+ languages; 0.6B parameters | Multilingual questions and longer input; a larger runtime than small encoder models | Apache-2.0 |
| [ncbi/MedCPT-Article-Encoder](https://huggingface.co/ncbi/MedCPT-Article-Encoder) plus [Query-Encoder](https://huggingface.co/ncbi/MedCPT-Query-Encoder) | 768 | Examples use 512 tokens for title/abstract pairs and 64 for queries | Biomedical retrieval specialist; requires a matched pair of encoders | Public domain, as declared on both model cards |

BGE recommends a retrieval instruction on short queries, without adding it to
passages. Its card cautions against treating absolute similarity values as a
universal relevance threshold. Record the query instruction and normalization
policy used in a comparison. [BGE usage and guidance](https://huggingface.co/BAAI/bge-small-en-v1.5#usage).

Qwen uses a task instruction on the query side. A long supported context does
not establish that embedding whole papers gives better retrieval than passages.
Its documented Transformers requirement starts at 4.51.0; a compatible current
runtime must still be tested on this project's Python environment.
[Qwen model usage](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B#usage).

MedCPT was designed for biomedical queries and articles such as PubMed titles
and abstracts. Applying its article encoder to full-text page chunks is an
adaptation requiring evaluation. Its example's 64-token query setting is not a
64-position architectural limit: the [query configuration](https://huggingface.co/ncbi/MedCPT-Query-Encoder/blob/main/config.json)
has 512 positions. The official retrieval example uses inner-product scoring;
do not silently substitute a normalized cosine pipeline.
[NCBI retrieval implementation](https://github.com/ncbi/MedCPT#searching-pubmed-with-medcpt).

The recommendation to test BGE first is an engineering judgment about incremental
integration and resource use, not a claim of measured superiority on diabetes
publications. MiniLM remains a reference baseline; no replacement is chosen yet.

## Image and document-page candidates

| Candidate | Representation | Reason to test | Main limitation |
|---|---|---|---|
| [Microsoft BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | One 512-dimensional vector per image/text input; 256-token text configuration; 224-pixel image encoder | Trained on biomedical figure/caption pairs; relevant to extracted research figures | Figure/caption training does not establish complete-page retrieval quality; English-only; different OpenCLIP runtime |
| [vidore/colqwen2-v1.0](https://huggingface.co/vidore/colqwen2-v1.0) | Multiple 128-dimensional vectors per page/query, with variable vector counts | Designed for visual document retrieval using page images and text queries | Larger 2B-class backbone and different storage/scoring requirements; not a drop-in single-vector replacement |

BiomedCLIP's published model card declares MIT licensing. Its
[configuration](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/main/open_clip_config.json)
confirms the dimensions and input sizes. The setup example uses Python 3.10 and
older package versions, so compatibility with this Python 3.13 project remains
untested. A domain-specific name is not proof of improved results on full pages.

ColQwen2 follows the visual document retrieval approach described in the
[ColPali paper](https://arxiv.org/abs/2407.01449). It combines query/page vectors
through late interaction: match query vectors to page vectors, then aggregate
those matches. Averaging everything into one vector would change the method.
Its card lists Apache-2.0 for the backbone and MIT for adapters, and notes that
the query prefix depends on the chosen library/version. Pin both model artifacts
and processor behavior before comparing results.
[ColQwen2 usage and license](https://huggingface.co/vidore/colqwen2-v1.0).

A smaller page-retrieval candidate, [vidore/colSmol-500M](https://huggingface.co/vidore/colSmol-500M),
was also inspected. Its card describes a SmolVLM backbone but its license
paragraph names Qwen2-VL. That documentation inconsistency needs artifact/base
license verification before adoption; it is not the first proposed image trial.

Regardless of the retriever, retain the complete original page PNG and page
text. A later vision-capable answer model receives the selected actual images
and text; retrieval vectors are not image evidence.

## Local and AWS constraints

The inspected development host is ARM64 with 16 GiB of physical memory, read
from the local OS. No candidate's RAM use, latency or Apple GPU compatibility
was measured. Start with one small text model at a time; verify a larger model
with a small batch before expanding. Do not assume CUDA-specific examples run
unchanged on this machine.

The local candidates have published downloadable-weight usage paths. They do
not require Google Cloud, Gemini, ADK or Google service credentials. Access to
model artifacts and a tested local dependency set are still prerequisites.

For a managed AWS alternative, [Amazon Titan Text Embeddings V2](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-embedding-models.html)
accepts up to 8,192 tokens or 50,000 characters and provides 1,024 dimensions by
default, with 512/256 options. This would introduce a Bedrock API dependency and
request charges; it does not run locally as downloaded weights. No AWS access,
availability, permissions or paid inference was tested in this research.

## Proposed first comparison

1. Establish the first functioning local MiniLM baseline, including actual token
   counts and rejection/handling of oversize input.
2. Compare BGE-small on the same frozen corpus and relevance set. Preserve all
   failures/coverage differences rather than silently excluding difficult cases.
3. Add Qwen or MedCPT if multilingual behavior or biomedical matching justifies
   the extra runtime/adapter work. Select this on development questions only.
4. Evaluate a page-specific visual retriever separately after text retrieval is
   working. Use BiomedCLIP for a figure experiment, not as an assumed solution
   to complete-page layout retrieval.

| Check | What to record |
|---|---|
| Page-level Hit@5 | Count questions with a confirmed relevant document-version/page among the first five unique pages |
| Ranking | Reciprocal rank of the first confirmed relevant page; report the case count |
| Coverage | Oversize/rejected input, empty results and unsupported cases |
| Operation | Load/encode/query duration, peak memory where measurable, calls and failures |
| Model configuration | Model revision, tokenizer, instruction/template, pooling, normalization, precision and similarity method |

Keep development questions separate from held-out questions. Suggested relevance
labels remain provisional until checked. A small corpus needs explicit limits:
document-level Hit@5 can be trivial with five or fewer documents. Page-level
ranking avoids that particular issue but does not make a small sample conclusive.
Abstention and support for answer claims need separate evaluation after the
answer-generation step exists; embedding similarity is not accuracy.

## Storage and reproducibility

Different models must use separate vector spaces and configuration identities.
MiniLM and BGE both produce 384 values, but their vectors cannot be mixed.
The CLIP family also needs matching text/image encoders from the same model
configuration. New candidates require new embeddings; original PDF, page and
chunk records can remain unchanged.

Single-vector text models fit a per-chunk vector design. ColQwen2 requires
multiple vectors linked to one page plus late-interaction scoring; the future
schema must account for that explicitly. Candidate model revisions, runtime
versions and processing settings must be pinned when implementing a trial.

This research added no embedding code, tests, model installation, benchmarks,
database writes or cloud resources. Notebook section 14 still documents the
original-model inspection; an executable candidate comparison belongs to the
next implementation step.
