"""
Build TuRTLe benchmarks into CVDP JSONL datasets.

Supported benchmarks (flags):
  --rtllm          -> writes benchmark_datasets/TuRTLe/rtllm_complete.jsonl
  --verilogeval    -> writes benchmark_datasets/TuRTLe/verilogeval_complete.jsonl
Debug helper:
  --test           -> only write the first task for each selected benchmark

Defaults to all supported benchmarks if no flags are provided.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_REPOS_ROOT = Path(__file__).resolve().parents[1] / "dataset_repos"
TURTLE_ROOT = DATASET_REPOS_ROOT / "TuRTLe"
RTLLM_ROOT = TURTLE_ROOT / "turtle/tasks/RTLLM"
VERILOG_EVAL_ROOT = TURTLE_ROOT / "turtle/tasks/verilog-eval"
RTLLM_MODULE_PATH = TURTLE_ROOT / "turtle/tasks/rtllm.py"
VERILOG_EVAL_MODULE_PATH = TURTLE_ROOT / "turtle/tasks/verilog_eval.py"
ADAPTERS_ROOT = REPO_ROOT / "src" / "datasets" / "builders" / "turtle_cvdp" / "adapters"
OUT_DIR = REPO_ROOT / "benchmark_datasets/TuRTLe"

def _install_import_stubs():
    """Provide lightweight stubs so we can import the task builders without heavy deps."""
    if "datasets" not in sys.modules:
        ds = types.ModuleType("datasets")
        ds.load_dataset = lambda *a, **k: {}
        ds.load_from_disk = lambda *a, **k: {}
        sys.modules["datasets"] = ds

    # metrics package and children
    if "metrics" not in sys.modules:
        sys.modules["metrics"] = types.ModuleType("metrics")
    for child in ["code_eval", "eval_verilog", "openlane_unified", "ppa_score"]:
        name = f"metrics.{child}"
        if name not in sys.modules:
            mod = types.ModuleType(name)
            if child == "code_eval":
                mod.estimate_pass_at_k = lambda *a, **k: 0
                mod.compute_perplexities = lambda *a, **k: 0
                mod.compute_min_k = lambda *a, **k: 0
                mod.compute_min_k_veriContaminated = lambda *a, **k: 0
            elif child == "eval_verilog":
                mod.eval_rtllm = lambda *a, **k: {}
                mod.eval_verilog_eval = lambda *a, **k: {}
            elif child == "openlane_unified":
                mod.create_problem_structure = lambda *a, **k: {}
                mod.run_openlane_for_generation = lambda *a, **k: {"status": "Skipped", "metrics": {}}
            elif child == "ppa_score":
                mod.compute_ppa_score = lambda *a, **k: 0
            sys.modules[name] = mod

    # transformers AutoTokenizer stub
    if "transformers" not in sys.modules:
        tr = types.ModuleType("transformers")

        class _DummyTokenizer:
            chat_template = None

            @classmethod
            def from_pretrained(cls, *a, **k):
                return cls()

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, thinking=False):
                # Return concatenated user content for simple passthrough
                return "\n".join([m["content"] for m in messages])

            def encode(self, text, add_special_tokens=True):
                return [ord(c) for c in text]

            def decode(self, ids, skip_special_tokens=False):
                return "".join(chr(i) for i in ids)

        class _DummyConfig:
            def __init__(self, max_position_embeddings=32768):
                self.max_position_embeddings = max_position_embeddings

            @classmethod
            def from_pretrained(cls, *a, **k):
                return cls()

        tr.AutoTokenizer = _DummyTokenizer
        tr.AutoConfig = _DummyConfig
        sys.modules["transformers"] = tr

    # src.turtle_eval.base.TaskExtension stub
    if "src" not in sys.modules:
        sys.modules["src"] = types.ModuleType("src")
    if "src.turtle_eval" not in sys.modules:
        sys.modules["src.turtle_eval"] = types.ModuleType("src.turtle_eval")
    if "src.turtle_eval.base" not in sys.modules:
        base = types.ModuleType("src.turtle_eval.base")

        class _TaskExtension:
            def __init__(self, *a, **k):
                pass

        base.TaskExtension = _TaskExtension
        sys.modules["src.turtle_eval.base"] = base


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_install_import_stubs()

rtllm = _load_module("rtllm_module", RTLLM_MODULE_PATH)
verilog_eval = _load_module("verilog_eval_module", VERILOG_EVAL_MODULE_PATH)


def _new_rtllm_instance():
    obj = rtllm.RTLLM.__new__(rtllm.RTLLM)
    obj.path_dataset = str(RTLLM_ROOT)
    obj.tokenizer = None
    obj.model = None
    obj.prompt = ""
    obj.debug = False
    return obj


def _new_verilog_eval_instance():
    obj = verilog_eval.VerilogEvalCodeComplete.__new__(verilog_eval.VerilogEvalCodeComplete)
    obj.path_dataset = str(VERILOG_EVAL_ROOT / "dataset_code-complete-iccad2023")
    obj.path_examples = str(VERILOG_EVAL_ROOT / "scripts")
    obj.examples = 0  # no few-shot prefix
    obj.tokenizer = None
    obj.model = None
    obj.prompt = ""
    obj.debug = False
    obj.task_name = "code-complete-iccad2023"
    obj.simulator = "icarus"
    return obj


def _rtllm_docs():
    design_files = sorted(RTLLM_ROOT.rglob("design_description.txt"))
    for design_path in design_files:
        folder_rel = design_path.parent.relative_to(RTLLM_ROOT).as_posix()
        yield {"folder_path": folder_rel}, design_path


def _verilog_eval_docs():
    dataset_dir = VERILOG_EVAL_ROOT / "dataset_code-complete-iccad2023"
    prompt_files = sorted(dataset_dir.glob("*_prompt.txt"))
    for prompt_path in prompt_files:
        m = re.match(r"Prob\d+_(.+?)_prompt\.txt", prompt_path.name)
        if not m:
            continue
        task_id = m.group(1)
        yield {"task_id": task_id}, prompt_path


def build_rtllm_entry(doc, design_path, client, golden_mode: bool = False):
    prompt = client.get_prompt(doc).strip()
    prompt = prompt.rstrip() + "\nWrite your solution inside rtl/."
    ref_path = Path(client.get_reference_path(doc))
    ref_content = client.get_reference(doc)

    test_path = design_path.parent / "testbench.v"
    makefile_path = design_path.parent / "makefile"
    adapter_root = ADAPTERS_ROOT / "rtllm"
    harness_path = adapter_root / "rtllm_cvdp_harness.py"
    compose_path = adapter_root / "docker-compose.yml"
    env_path = adapter_root / ".env"

    harness = {}
    if ref_path.exists():
        harness[f"src/{ref_path.name}"] = ref_content
    if test_path.exists():
        harness[f"src/{test_path.name}"] = test_path.read_text()
    if makefile_path.exists():
        harness[f"src/{makefile_path.name}"] = makefile_path.read_text()
    if harness_path.exists():
        harness[f"src/{harness_path.name}"] = harness_path.read_text()
    if compose_path.exists():
        harness["docker-compose.yml"] = compose_path.read_text()
    if env_path.exists():
        harness["src/.env"] = env_path.read_text()

    context = {}
    if golden_mode and ref_path.exists():
        # Place the golden reference into rtl/ for harness testing
        context["rtl/" + ref_path.name] = ref_content

    id_suffix = design_path.parent.relative_to(RTLLM_ROOT).as_posix().replace("/", "_")
    entry = {
        "id": f"cvdp_agentic_RTLLM_{id_suffix}_0001",
        "categories": ["cid003", "easy"],
        "system_message": "",
        "prompt": prompt,
        "context": context,
        "patch": {},
        "harness": harness,
    }
    return entry


def build_verilog_eval_entry(doc, client, golden_mode: bool = False):
    prompt = client.get_prompt(doc).strip()
    # Strip the VerilogEval completion system message
    system_msg = """
You only complete chats with syntax correct Verilog code. End the Verilog module code completion with 'endmodule'. Do not include module, input and output definitions.
"""
    prompt = prompt.replace(system_msg.strip(), "").strip()
    prompt = prompt.replace("\nAnswer:\n", "").replace("Answer:\n", "").strip()
    # Add explicit instruction about where to write code
    prompt = prompt.rstrip() + "\nWrite all code inside rtl/. Your code MUST be a single module called TopModule in a file called TopModule.sv."
    ref_path = Path(client.get_path(doc["task_id"], suffix="ref"))
    test_path = Path(client.get_path(doc["task_id"], suffix="test"))
    ifc_path = None
    adapter_root = ADAPTERS_ROOT / "verilogeval"
    harness_path = adapter_root / "verilogeval_cvdp_harness.py"
    compose_path = adapter_root / "docker-compose.yml"
    env_path = adapter_root / ".env"

    harness = {}
    if ref_path.exists():
        harness[f"src/{ref_path.name}"] = ref_path.read_text()
    if test_path.exists():
        harness[f"src/{test_path.name}"] = test_path.read_text()
    if ifc_path and ifc_path.exists():
        harness[f"src/{ifc_path.name}"] = ifc_path.read_text()
    if harness_path.exists():
        harness[f"src/{harness_path.name}"] = harness_path.read_text()
    if compose_path.exists():
        harness["docker-compose.yml"] = compose_path.read_text()
    if env_path.exists():
        harness["src/.env"] = env_path.read_text()

    context = {}
    if golden_mode and ref_path.exists():
        ref_text = ref_path.read_text()
        # Ensure module name is TopModule and file is TopModule.sv
        ref_text = re.sub(r"\bmodule\s+\w+", "module TopModule", ref_text, count=1)
        context["rtl/TopModule.sv"] = ref_text

    id_prefix = "cvdp_agentic_VerilogEval_Completion"

    entry = {
        "id": f"{id_prefix}_{doc['task_id']}_0001",
        "categories": ["cid002", "easy"],
        "system_message": "",
        "prompt": prompt,
        "context": context,
        "patch": {},
        "harness": harness,
    }
    return entry


def main():
    parser = argparse.ArgumentParser(description="Convert TuRTLe benchmarks to CVDP JSONL.")
    parser.add_argument("--rtllm", action="store_true", help="Build RTLLM dataset")
    parser.add_argument("--verilogeval", action="store_true", help="Build VerilogEval dataset")
    parser.add_argument("--golden", action="store_true", help="(RTLLM/VerilogEval) place golden solution in rtl/ for testing harness")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Only include the first task for each selected benchmark (for quick debugging)",
    )
    args = parser.parse_args()

    selected = []
    if args.rtllm or args.verilogeval:
        if args.rtllm:
            selected.append("rtllm")
        if args.verilogeval:
            selected.append("verilogeval")
    else:
        selected = ["rtllm", "verilogeval"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if "rtllm" in selected:
        client = _new_rtllm_instance()
        entries = []
        for doc, design_path in _rtllm_docs():
            entries.append(build_rtllm_entry(doc, design_path, client, golden_mode=args.golden))
            if args.test:
                break
        out_path = OUT_DIR / "rtllm_complete.jsonl"
        with out_path.open("w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {len(entries)} RTLLM tasks to {out_path}")

    if "verilogeval" in selected:
        client = _new_verilog_eval_instance()
        entries: list[dict] = []
        # Completion set
        for doc, _prompt_path in _verilog_eval_docs():
            entries.append(build_verilog_eval_entry(doc, client, golden_mode=args.golden))
            if args.test:
                break
        out_path = OUT_DIR / "verilogeval_complete.jsonl"
        with out_path.open("w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {len(entries)} VerilogEval tasks to {out_path}")


if __name__ == "__main__":
    main()
