"""
DR Tulu - Non-UI Test Inference Script
=======================================
Runs a single query end-to-end through the AutoReasonSearchWorkflow.
Designed to run inside a SLURM GPU allocation (no interactive prompts).

Captures:
  (a) All retrieved documents (title, url, snippet, text, score, source tool, runtime)
  (b) Final model response
  (c) Per-step timing (search phase, answer phase, each tool call)
  (d) Pushes results to HuggingFace Hub under deepresearchediting/

Usage:
    python test_inference.py --query "What are the latest advances in protein folding?"
    python test_inference.py --query "..." --hf-dataset deepresearchediting/dr-tulu-runs
    python test_inference.py --skip-launch --query "..."   # if servers already running
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent / "agent"))
sys.path.insert(0, str(Path(__file__).parent / "agent" / "workflows"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def check_api_keys():
    missing = []
    for key in ["SERPER_API_KEY", "S2_API_KEY", "JINA_API_KEY"]:
        val = os.environ.get(key, "")
        if not val or val == "xxx":
            missing.append(key)
    if missing:
        log.error(f"Missing or placeholder API keys in .env: {missing}")
        sys.exit(1)
    log.info("API keys: OK")


# ---------------------------------------------------------------------------
# Service management
# ---------------------------------------------------------------------------

def wait_for_port(port: int, timeout: int = 300, label: str = "") -> bool:
    import socket
    label = label or f"port {port}"
    log.info(f"Waiting for {label} on port {port} ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                log.info(f"  -> {label} ready ({int(time.time()-start)}s elapsed)")
                return True
        except OSError:
            time.sleep(2)
    log.error(f"  -> {label} NOT ready after {timeout}s")
    return False


def start_services(vllm_port: int, mcp_port: int, model_name: str, max_model_len: int):
    import socket
    import subprocess

    def port_open(p):
        try:
            with socket.create_connection(("localhost", p), timeout=1):
                return True
        except OSError:
            return False

    procs = []

    # --- MCP server ---
    if port_open(mcp_port):
        log.info(f"MCP server already running on port {mcp_port}")
    else:
        log.info(f"Starting MCP server on port {mcp_port} ...")
        env = os.environ.copy()
        env["MCP_CACHE_DIR"] = f".cache-{os.uname().nodename}"
        mcp_log = open(f"/scratch/mw4141/code/dr-tulu/logs/mcp_server_{mcp_port}.log", "w")
        proc = subprocess.Popen(
            [sys.executable, "-m", "dr_agent.mcp_backend.main", "--port", str(mcp_port)],
            stdout=mcp_log, stderr=subprocess.STDOUT, env=env,
        )
        procs.append(proc)
        if not wait_for_port(mcp_port, timeout=120, label="MCP server"):
            log.error(f"MCP server failed. Check /scratch/mw4141/code/dr-tulu/logs/mcp_server_{mcp_port}.log")
            sys.exit(1)

    # --- vLLM server ---
    if port_open(vllm_port):
        log.info(f"vLLM server already running on port {vllm_port}")
    else:
        log.info(f"Starting vLLM for {model_name} on port {vllm_port} ...")
        log.info("  First run will download ~16 GB. Set HF_HOME to scratch to avoid quota issues.")
        env = os.environ.copy()
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env["HUGGING_FACE_HUB_TOKEN"] = hf_token
        hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        env["HF_HOME"] = hf_home
        env["TRANSFORMERS_CACHE"] = hf_home
        vllm_log = open(f"/tmp/vllm_server_{vllm_port}.log", "w")
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--port", str(vllm_port),
            "--dtype", "auto",
            "--max-model-len", str(max_model_len),
            "--gpu-memory-utilization", "0.90",
        ]
        proc = subprocess.Popen(cmd, stdout=vllm_log, stderr=subprocess.STDOUT, env=env)
        procs.append(proc)
        log.info("Waiting for vLLM to load model (up to 10 min) ...")
        if not wait_for_port(vllm_port, timeout=600, label="vLLM server"):
            log.error(f"vLLM failed. Check /tmp/vllm_server_{vllm_port}.log")
            sys.exit(1)

    return procs


# ---------------------------------------------------------------------------
# Document extraction from tool_calls
# ---------------------------------------------------------------------------

def extract_documents(tool_calls) -> list[dict]:
    """
    Walk every tool call and extract all retrieved documents with metadata.
    Returns a flat list of dicts — one per document.
    """
    from dr_agent.tool_interface.data_types import DocumentToolOutput

    docs = []
    for tc in tool_calls:
        base = {
            "tool_name": tc.tool_name,
            "call_id": getattr(tc, "call_id", None),
            "tool_runtime_s": round(getattr(tc, "runtime", 0.0), 3),
            "tool_error": tc.error or None,
            "query": getattr(tc, "query", None),
        }
        if isinstance(tc, DocumentToolOutput) and tc.documents:
            for doc in tc.documents:
                docs.append({
                    **base,
                    "doc_id": doc.id,
                    "title": doc.title,
                    "url": doc.url,
                    "snippet": doc.snippet,
                    "text": doc.text,
                    "summary": doc.summary,
                    "score": doc.score,
                    "doc_error": doc.error,
                })
        else:
            # Non-document tool call (e.g. browse that returned raw text)
            docs.append({
                **base,
                "doc_id": None,
                "title": None,
                "url": None,
                "snippet": tc.output if tc.output else None,
                "text": None,
                "summary": None,
                "score": None,
                "doc_error": None,
            })
    return docs


# ---------------------------------------------------------------------------
# Timing callback
# ---------------------------------------------------------------------------

class StepTimer:
    """Attached as on_step_callback; records per-step wall-clock times."""

    def __init__(self):
        self.steps: list[dict] = []
        self._last_tool_call_count = 0
        self._step_start = time.perf_counter()

    def __call__(self, text: str, tool_calls: list):
        now = time.perf_counter()
        n_new_calls = len(tool_calls) - self._last_tool_call_count
        if n_new_calls > 0:
            for tc in tool_calls[self._last_tool_call_count:]:
                self.steps.append({
                    "event": "tool_call",
                    "tool_name": getattr(tc, "tool_name", "unknown"),
                    "call_id": getattr(tc, "call_id", None),
                    "tool_runtime_s": round(getattr(tc, "runtime", 0.0), 3),
                    "wall_clock_s": round(now - self._step_start, 3),
                })
            self._last_tool_call_count = len(tool_calls)
            self._step_start = now

    def summary(self) -> dict:
        if not self.steps:
            return {"steps": [], "total_tool_wall_clock_s": 0.0}
        total = sum(s["wall_clock_s"] for s in self.steps)
        return {
            "steps": self.steps,
            "total_tool_wall_clock_s": round(total, 3),
        }


# ---------------------------------------------------------------------------
# Core async runner
# ---------------------------------------------------------------------------

async def run_query(
    query: str,
    vllm_port: int,
    mcp_port: int,
    model_name: str,
    dataset_name: str | None,
    verbose: bool,
) -> dict:
    from auto_search_sft import AutoReasonSearchWorkflow

    config_path = Path(__file__).parent / "agent" / "workflows" / "auto_search_sft.yaml"

    workflow = AutoReasonSearchWorkflow(
        configuration=str(config_path),
        search_agent_base_url=f"http://localhost:{vllm_port}/v1",
        search_agent_model_name=model_name,
        mcp_port=mcp_port,
    )
    # Services already running — skip the interactive check
    workflow.before_launch_check = lambda: None

    log.info("Setting up workflow components ...")
    workflow.setup_components(mcp_port=mcp_port)

    timer = StepTimer()

    # --- Search phase ---
    log.info(f"Query: {query!r}")
    t_total_start = time.perf_counter()
    t_search_start = t_total_start

    raw = await workflow(
        problem=query,
        dataset_name=dataset_name,
        verbose=verbose,
        step_callback=timer,
    )

    t_search_end = time.perf_counter()

    # Extract full tool call list from whichever traces object was returned
    full_traces = raw.get("full_traces")
    tool_calls_list = []
    if full_traces is not None:
        # full_traces is GenerateWithToolsOutput; it may have been serialised to dict already
        if hasattr(full_traces, "tool_calls"):
            tool_calls_list = full_traces.tool_calls
        elif isinstance(full_traces, dict) and "tool_calls" in full_traces:
            # already model_dump'd — we can't re-extract Document objects, store as-is
            tool_calls_list = []

    documents = extract_documents(tool_calls_list)

    timing = {
        "search_phase_s": round(t_search_end - t_search_start, 3),
        "total_s": round(t_search_end - t_total_start, 3),
        "step_detail": timer.summary(),
        # per-tool-call runtimes already inside step_detail; summarise by tool
        "by_tool": {},
    }
    for step in timer.steps:
        tn = step["tool_name"]
        if tn not in timing["by_tool"]:
            timing["by_tool"][tn] = {"calls": 0, "total_tool_runtime_s": 0.0}
        timing["by_tool"][tn]["calls"] += 1
        timing["by_tool"][tn]["total_tool_runtime_s"] += step["tool_runtime_s"]

    # Serialise full_traces (GenerateWithToolsOutput pydantic model)
    traces_serialised = None
    if full_traces is not None:
        try:
            traces_serialised = full_traces.model_dump()
        except Exception:
            traces_serialised = str(full_traces)

    return {
        # --- identity ---
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "query": query,
        "model": model_name,
        "dataset_name": dataset_name,
        # --- core outputs ---
        "final_response": raw.get("final_response", ""),
        # --- retrieved documents ---
        "documents": documents,
        "searched_links": raw.get("searched_links", []),
        "browsed_links": raw.get("browsed_links", []),
        "total_tool_calls": raw.get("total_tool_calls", len(tool_calls_list)),
        "total_failed_tool_calls": raw.get("total_failed_tool_calls", 0),
        "failed_tool_call_errors": raw.get("failed_tool_call_errors", []),
        # --- timing ---
        "timing": timing,
        # --- full trace ---
        "full_traces": traces_serialised,
    }


# ---------------------------------------------------------------------------
# HuggingFace push
# ---------------------------------------------------------------------------

def push_to_hf(result: dict, hf_dataset: str, hf_token: str | None):
    """Push one inference result as a new row to a HF Hub dataset."""
    try:
        from datasets import Dataset, load_dataset
    except ImportError:
        log.warning("datasets library not installed; skipping HF push. pip install datasets")
        return

    log.info(f"Pushing result to HuggingFace: {hf_dataset} ...")

    # Make a JSON-safe copy (convert nested objects → strings where needed)
    row = {}
    for k, v in result.items():
        if k == "documents":
            row[k] = json.dumps(v, default=str)
        elif k == "full_traces":
            row[k] = json.dumps(v, default=str)
        elif k == "timing":
            row[k] = json.dumps(v, default=str)
        elif k == "failed_tool_call_errors":
            row[k] = json.dumps(v, default=str)
        elif isinstance(v, list):
            row[k] = json.dumps(v, default=str)
        else:
            row[k] = str(v) if v is not None else ""

    new_ds = Dataset.from_list([row])

    # Try to append to existing dataset, else create new
    try:
        existing = load_dataset(hf_dataset, split="train", token=hf_token)
        merged = existing.select(range(len(existing)))
        from datasets import concatenate_datasets
        merged = concatenate_datasets([merged, new_ds])
        merged.push_to_hub(hf_dataset, token=hf_token)
        log.info(f"Appended to existing dataset ({len(existing)} -> {len(merged)} rows)")
    except Exception:
        # Dataset doesn't exist yet — create it
        new_ds.push_to_hub(hf_dataset, token=hf_token)
        log.info(f"Created new dataset at https://huggingface.co/datasets/{hf_dataset}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DR Tulu non-UI inference test")
    parser.add_argument(
        "--query",
        default="What are the most recent breakthroughs in large language model alignment research?",
    )
    parser.add_argument("--vllm-port", type=int, default=30001)
    parser.add_argument("--mcp-port", type=int, default=8000)
    parser.add_argument("--model", default="rl-research/DR-Tulu-8B")
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Prompt hint: 'healthbench', 'simpleqa', 'deep_research_bench', etc. "
             "Leave unset for generic open-ended answer.",
    )
    parser.add_argument("--output", default="test_inference_result.json")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--skip-launch",
        action="store_true",
        help="Skip launching MCP+vLLM servers (assume they are already running)",
    )
    parser.add_argument(
        "--hf-dataset",
        default="deepresearchediting/dr-tulu-runs",
        help="HuggingFace dataset repo to push results to (org/name format)",
    )
    parser.add_argument(
        "--no-hf-push",
        action="store_true",
        help="Skip pushing to HuggingFace Hub",
    )
    args = parser.parse_args()

    log.info("=== DR Tulu Test Inference ===")
    check_api_keys()

    procs = []
    if not args.skip_launch:
        procs = start_services(
            vllm_port=args.vllm_port,
            mcp_port=args.mcp_port,
            model_name=args.model,
            max_model_len=args.max_model_len,
        )
    else:
        log.info("Skipping service launch (--skip-launch)")

    try:
        result = asyncio.run(run_query(
            query=args.query,
            vllm_port=args.vllm_port,
            mcp_port=args.mcp_port,
            model_name=args.model,
            dataset_name=args.dataset_name,
            verbose=args.verbose,
        ))

        # ---- Print summary ----
        timing = result["timing"]
        docs = result["documents"]
        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)
        print(result["final_response"])
        print("=" * 70)
        print(f"\nTiming:")
        print(f"  Total                : {timing['total_s']}s")
        print(f"  Search phase         : {timing['search_phase_s']}s")
        print(f"  Tool calls breakdown :")
        for tool, stats in timing["by_tool"].items():
            print(f"    {tool:30s}  {stats['calls']} calls  {stats['total_tool_runtime_s']:.1f}s total runtime")
        print(f"\nRetrieved documents  : {len(docs)}")
        print(f"Unique searched URLs : {len(result['searched_links'])}")
        print(f"Unique browsed URLs  : {len(result['browsed_links'])}")
        print(f"Tool calls           : {result['total_tool_calls']}")
        print(f"Failed tool calls    : {result['total_failed_tool_calls']}")

        # ---- Save locally ----
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log.info(f"Result saved to {out_path}")

        # ---- Push to HF ----
        if not args.no_hf_push:
            hf_token = os.environ.get("HF_TOKEN") or None
            push_to_hf(result, args.hf_dataset, hf_token)

    finally:
        for proc in procs:
            log.info(f"Terminating subprocess PID {proc.pid}")
            proc.terminate()


if __name__ == "__main__":
    main()
