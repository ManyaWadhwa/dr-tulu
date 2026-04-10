
> The original upstream README is preserved at [`README_original.md`](README_original.md).

---

## Setup for New Users

### 1. Create your `.env` file

**This is required before running anything.** Copy the example and fill in your keys:

```bash
cp .env.example .env   # or copy from the template below
```

Edit `.env` with your actual credentials:

```dotenv
# Required
SERPER_API_KEY=your_key_here   # https://serper.dev/          (free $50 credit)
S2_API_KEY=your_key_here       # https://api.semanticscholar.org/  (free)
JINA_API_KEY=your_key_here     # https://jina.ai/reader/      (1M tokens free/month)

# Required for downloading the model
HF_TOKEN=your_hf_token_here    # https://huggingface.co/settings/tokens

# Optional: only if using OpenAI models instead of local DR-Tulu-8B
OPENAI_API_KEY=
```

> `.env` is listed in `.gitignore` and will never be committed.

---

### 2. Set up the Python environment

Run once on the login node (no GPU needed):

```bash
bash setup_env.sh
```

This creates a `uv` virtual environment at `.venv/` and installs the `agent/` package with vLLM support.

---

### 3. Run a test inference (HPC / SLURM)

The test script runs end-to-end: launches services, runs a query, saves documents + timing, and pushes results to HuggingFace.

**Submit as a batch job (recommended):**
```bash
sbatch run_test_inference.slurm
```

**Or run interactively on a GPU node:**
```bash
srun --gres=gpu:1 --mem=40G --cpus-per-task=8 --pty bash
source .venv/bin/activate
export HF_HOME=/scratch/$USER/.cache/huggingface
python test_inference.py --verbose
```

**Custom query:**
```bash
python test_inference.py \
    --query "What are the latest findings on Alzheimer's treatment?" \
    --hf-dataset "deepresearchediting/dr-tulu-runs" \
    --verbose
```

**`test_inference.py` captures:**
- Final model response
- All retrieved documents (title, URL, snippet, text, score, source tool)
- Per-step timing: each tool call, search phase total, overall total
- Full agent trace
- Pushes everything as a row to the HuggingFace dataset `deepresearchediting/dr-tulu-runs`

### Input modes

`test_inference.py` accepts input in three mutually exclusive ways:

**1. Single query (string)**
```bash
python test_inference.py --query "What are the latest findings on Alzheimer's treatment?"
```

**2. Batch from a JSONL file**

Each line is a JSON object with at least a `query` field. Any extra columns are preserved and passed through to the output row.

```bash
python test_inference.py --input-file sample_queries.jsonl
```

Example `sample_queries.jsonl`:
```jsonl
{"query": "What are the main approaches to neural network pruning?"}
{"query": "How does retrieval-augmented generation improve factual accuracy?", "topic": "rag"}
```

**3. Batch from a HuggingFace dataset**

The dataset must have a `query` column. By default loads the `train` split — override with `--input-split`.

```bash
python test_inference.py --input-dataset org/my-queries
python test_inference.py --input-dataset org/my-queries --input-split validation
```

**Batch behavior:**
- Each query runs sequentially through the full pipeline (MCP server and vLLM stay up between queries).
- Results are appended to the output file as JSONL (one row per query).
- Each result is pushed to HuggingFace immediately after the query completes — you can watch results appear live on the dataset page as the batch progresses.

---

**Options:**
```
--query TEXT           Single query string (ignored if --input-file or --input-dataset is set)
--input-file PATH      Path to a JSONL file with a "query" column
--input-dataset NAME   HuggingFace dataset with a "query" column
--input-split TEXT     Dataset split when using --input-dataset (default: train)
--vllm-port INT        vLLM server port (default: 30001)
--mcp-port INT         MCP server port (default: 8000)
--model TEXT           HuggingFace model ID (default: rl-research/DR-Tulu-8B)
--max-model-len INT    Max token length for vLLM (default: 40960)
--dataset-name TEXT    Prompt hint: healthbench, simpleqa, deep_research_bench, etc.
--output PATH          Output path — JSON for single query, JSONL for batch
--hf-dataset TEXT      HuggingFace dataset to push results to
--no-hf-push           Skip HuggingFace push
--skip-launch          Skip launching services (if already running)
--verbose              Show full agent reasoning trace
```

---

### Architecture (inference only)

```
Your query
    │
    ▼
SearchAgent (DR-Tulu-8B via vLLM)
    ├─► snippet_search  ──► Serper / Semantic Scholar
    ├─► google_search   ──► Serper
    └─► browse_webpage  ──► Jina / Crawl4AI
    │         (up to 10 tool calls, with retrieved Documents)
    ▼
AnswerAgent (same model) ──► synthesises final answer
    │
    ▼
Result: final_response + all documents + timing
```

All agent–tool communication goes through an **MCP server** (port 8000). The model is served locally via **vLLM** (port 30001). No data leaves your machine except the external search API calls (Serper, Jina, S2).

---

### Repository structure

```
dr-tulu/
├── agent/               # Core inference library (install this)
│   ├── dr_agent/        # Agents, tools, MCP backend, prompts
│   └── workflows/       # AutoReasonSearchWorkflow + YAML configs
├── rl/                  # GRPO RL training (multi-GPU, not needed for inference)
├── sft/                 # SFT training (LLaMA-Factory, not needed for inference)
├── test_inference.py    # Non-UI end-to-end inference script
├── run_test_inference.slurm  # SLURM batch job
├── setup_env.sh         # One-time environment setup
├── .env                 # YOUR API KEYS — create this, never commit it
├── .env.example         # Template (safe to commit)
└── README_original.md   # Original upstream README
```

---

For the original paper, training details, and evaluation benchmarks, see [`README_original.md`](README_original.md) and the subdirectory READMEs.
