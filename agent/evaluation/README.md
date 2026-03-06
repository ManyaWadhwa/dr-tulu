# Evaluation Scripts

This directory contains evaluation scripts for benchmarking DR agents across various tasks, from short-form QA to long-form deep research.

---

## Available Benchmarks

| Benchmark | Type | Description | Benchmark Name |
|-----------|------|-------------|----------------|
| **SQA-CS-V2** | Long-form | Scientific question answering with structured citations | `sqa_cs_v2` |
| **Deep Research Bench** | Long-form | Deep research reports (RACE & FACT metrics) | `deep_research_bench` |
| **ResearchQA** | Long-form | Research question answering with coverage metrics | `research_qa` |
| **HealthBench** | Long-form | Medical QA with physician-level rubrics | `healthbench` |
| **Genetic Diseases** | Domain-specific | Clinical genetics questions | `genetic_diseases` |
| **SimpleQA** | Short-form | Factuality in short-form answers | `simpleqa` |
| **Short Form QA** | Short-form | Multi-dataset QA framework (14+ datasets) | See supported tasks below |

---

## Running Evaluations

### Example: Evaluate Across All Benchmarks

**Prerequisites**: Before running the evaluation script, launch the required servers **on the same node**:

```bash
# Launch VLLM servers (requires 2 GPUs)
CUDA_VISIBLE_DEVICES=0 vllm serve rl-research/DR-Tulu-8B --dtype auto --port 30001 --max-model-len 40960
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-8B --dtype auto --port 30002 --max-model-len 40960

# Launch MCP server
python -m dr_agent.mcp_backend.main --port 8000
```

Then run the evaluation script:

```bash
#!/bin/bash
# Example script to run DR Tulu on multiple benchmarks

SAVE_FOLDER=eval_output/
MODEL=auto_search_sft
YAML_CONFIG=workflows/auto_search_sft.yaml
MAX_CONCURRENT=20

mkdir -p $SAVE_FOLDER

# Run evaluations on all benchmarks
for task in healthbench deep_research_bench research_qa genetic_diseases simpleqa 2wiki webwalker; do 
    echo "Running $MODEL on $task"
    python workflows/$MODEL.py \
        generate-dataset $task \
        --num-examples final_run \
        --max-concurrent $MAX_CONCURRENT \
        --batch-size $MAX_CONCURRENT \
        --use-cache \
        --config $YAML_CONFIG \
        --config-overrides "use_browse_agent=true,search_agent_max_tool_calls=10,browse_tool_name=jina" \
        --output $SAVE_FOLDER/$MODEL/$task.jsonl
    
    python scripts/evaluate.py $task $SAVE_FOLDER/$MODEL/$task.jsonl
done
```

For SQA-CS-V2 and Deep Research Bench evaluations, see the dedicated sections below.

---

## Deep Research Bench (DRB) Evaluation — Self-Contained

DRB evaluation measures deep research report quality using two metrics:
- **RACE**: Reference-based and Adaptive Criteria-driven Evaluation (article quality)
- **FACT**: Factual Abundance and Citation Trustworthiness (citation verification)

### Prerequisites

```bash
pip install google-genai tqdm huggingface_hub
export GEMINI_API_KEY="your_gemini_api_key_here"
```

> **Note**: DRB reference data (query, criteria, reference articles) is hosted at [`rl-research/dr-tulu-eval-data`](https://huggingface.co/datasets/rl-research/dr-tulu-eval-data) and will be auto-downloaded on first run.

### Quick Start (Self-Contained — No External Repo Needed)

```bash
# Full pipeline: format conversion + RACE + FACT evaluation
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model

# RACE only (skip FACT):
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model --skip_fact

# FACT only (skip RACE):
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model --skip_race

# Test with limit:
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model --limit 2

# English only:
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model --only_en
```

### Or via the unified evaluate.py script

```bash
python scripts/evaluate.py deep_research_bench eval_output/auto_search_sft/deep_research_bench.jsonl
```

### Output Structure

```
<output_dir>/
├── raw_data/<task_name>.jsonl          # Formatted articles
├── cleaned_data/<task_name>.jsonl      # Cleaned articles (citations removed)
├── race/<task_name>/
│   ├── raw_results.jsonl               # Per-item RACE scores
│   └── race_result.txt                 # Aggregated RACE metrics
└── fact/<task_name>/
    ├── scraped.jsonl                   # Articles with scraped citation content
    ├── validated.jsonl                 # Validated citations
    └── fact_result.txt                 # Aggregated FACT metrics
```

---

## SQA-CS-V2 Evaluation — Self-Contained

SQA-CS-V2 evaluates scientific question answering with structured citations.

### Prerequisites

```bash
pip install astabench==0.3.1 inspect_ai datasets
export GOOGLE_API_KEY="your_google_api_key_here"
export HF_TOKEN="your_hf_token_here"   # Needs access to allenai/asta-bench (gated dataset)
```

### Quick Start (Self-Contained — No External Repo Needed)

```bash
# Full pipeline: convert + evaluate
python evaluation/sqa_eval/run_eval.py run \
    --input_file eval_output/auto_search_sft/sqa_cs_v2.jsonl

# Step-by-step:
# 1. Convert DR Tulu output to ASTA format
python evaluation/sqa_eval/run_eval.py convert \
    --input_file eval_output/auto_search_sft/sqa_cs_v2.jsonl

# 2. Run evaluation
python evaluation/sqa_eval/run_eval.py eval \
    --input_file eval_output/auto_search_sft/sqa_cs_v2_asta_format.jsonl

# With custom scorer model:
python evaluation/sqa_eval/run_eval.py run \
    --input_file eval_output/auto_search_sft/sqa_cs_v2.jsonl \
    --scorer_model "google/gemini-2.5-flash" \
    --max_connections 16
```

### Or via the unified evaluate.py script

```bash
python scripts/evaluate.py sqa_cs_v2 eval_output/auto_search_sft/sqa_cs_v2.jsonl
```

### SQA Response Format

SQA-CS-V2 requires responses in a specific JSON format with structured sections and citations:

```json
{
  "sections": [
    {
      "text": "text of section 1",
      "citations": [
        {
          "id": "[cite_id]",
          "title": "paper title",
          "snippets": ["evidence 1", "evidence 2"]
        }
      ]
    }
  ]
}
```

### Legacy Method (External Repo)

If you prefer using the original agent-baselines repository:

1. **Convert DR Tulu outputs to SQA format**:
   ```bash
   python evaluation/sqa_eval/convert_to_asta_format.py --folder <folder_name> --file <file_name>
   ```

2. **Clone the evaluation repository**:
   ```bash
   git clone https://github.com/allenai/agent-baselines
   cd agent-baselines
   ```

3. **Run evaluation**:
   ```bash
   uv run --extra sqa inspect eval astabench/sqa --display plain \
     --solver agent_baselines/solvers/sqa/debug/cached_solver.py \
     -S path=<outputfile_from_step1> \
     -T split=test \
     -T with_search_tools=False \
     -T simplified_eval=true \
     -T assess_jointly=true \
     --max-connections 16 \
     -T sentence_wise_cit_eval=false \
     -T all_at_once=true \
     -T scorer_model="google/gemini-2.5-flash"
   ```

**Note**: Export `GOOGLE_API_KEY` and `HF_TOKEN` before running.

