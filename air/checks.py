"""Program-side (non-LLM) checks applied to every LLM-written candidate before
it is allowed to use the GPU.  Every rejection reason is recorded and fed back
to the LLM in the next prompt."""
import ast
import hashlib
import re
from pathlib import Path

REQUIRED_DEFS = ["get_model", "get_training_config", "get_augmentation", "BarlowTwinsLoss"]

# Patterns that would break comparability or leak evaluation data.  The
# candidate only defines model/loss/augmentation/config; it must not touch
# data loading, the test split, result files, the network, or the shell.
FORBIDDEN = [
    (r"\bCIFAR100\s*\(|\bCIFAR10\s*\(|torchvision\.datasets", "loads a dataset (data loading belongs to the official script)"),
    (r"hidden_test|test_targets|test_data|--split|results_tmp|val_info|test_info", "references evaluation splits or result files"),
    (r"\bsubprocess\b|\bos\.system\b|\bos\.popen\b|\beval\s*\(|\bexec\s*\(|__import__", "runs shell / dynamic code"),
    (r"\brequests\b|\burllib\b|\bsocket\b|torch\.hub|load_state_dict_from_url|http[s]?://", "uses the network"),
    (r"weights\s*=\s*(?!None\b)['\"A-Za-z]|pretrained\s*=\s*True", "uses pretrained weights"),
    (r"\bopen\s*\(|\bshutil\b|\bos\.remove\b|\bos\.unlink\b", "touches the filesystem"),
    (r"train_eval_baseline|eval_wrapper|split_config", "imports or edits the evaluation harness"),
    (r"sys\.modules|setattr\s*\(\s*sys|builtins", "monkeypatches the interpreter"),
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def static_check(code: str) -> list[str]:
    """Return a list of problems (empty == pass)."""
    problems = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    defined |= {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    for name in REQUIRED_DEFS:
        if name not in defined:
            problems.append(f"missing top-level definition `{name}`")
    for pattern, why in FORBIDDEN:
        m = re.search(pattern, code)
        if m:
            problems.append(f"forbidden pattern `{m.group(0)}`: {why}")
    if len(code) > 60_000:
        problems.append("algorithm.py longer than 60k characters")
    return problems
