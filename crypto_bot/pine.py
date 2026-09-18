"""Safe Pine Script inspection for the local research workspace.

This module deliberately parses Pine as text. It never evaluates, imports, or
executes user-provided code.
"""

import re
from typing import Any, Dict, List


_VERSION = re.compile(r"^\s*//@version\s*=\s*(\d+)", re.IGNORECASE | re.MULTILINE)
_DECLARATION = re.compile(
    r"^\s*(indicator|strategy|library)\s*\(\s*(.*?)\)", re.IGNORECASE | re.MULTILINE
)
_INPUT = re.compile(
    r"\binput(?:\.(?:bool|float|int|string|source|symbol|timeframe|session|color|"
    r"enum))?\s*\((.*?)\)",
    re.IGNORECASE | re.DOTALL,
)
_CALL = re.compile(
    r"\b(alertcondition|alert|plot|plotshape|plotchar|hline|"
    r"strategy\.(?:entry|order|exit|close))\s*\(",
    re.IGNORECASE,
)


def _first_string(value: str) -> str:
    match = re.search(r"""["']([^"']+)["']""", value)
    return match.group(1).strip() if match else ""


def inspect_script(source: str, name: str = "untitled.pine") -> Dict[str, Any]:
    """Return bounded, human-readable metadata without running Pine code."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Pine Script source is required")
    if len(source.encode("utf-8")) > 512_000:
        raise ValueError("Pine Script is limited to 512 KB")

    version_match = _VERSION.search(source)
    declaration_match = _DECLARATION.search(source)
    declaration = declaration_match.group(1).lower() if declaration_match else "unknown"
    declaration_args = declaration_match.group(2) if declaration_match else ""
    title = _first_string(declaration_args) or name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

    inputs: List[str] = []
    for match in _INPUT.finditer(source):
        label = _first_string(match.group(1))
        if label and label not in inputs:
            inputs.append(label)
    calls = sorted({match.group(1).lower() for match in _CALL.finditer(source)})
    plots = [call for call in calls if call.startswith(("plot", "hline"))]
    alerts = [call for call in calls if call.startswith("alert")]
    strategy_actions = [call for call in calls if call.startswith("strategy.")]
    functions = sorted(set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", source)))
    unsupported = []
    if "request.security" in source or "request.security_lower_tf" in source:
        unsupported.append("external timeframe or symbol requests need explicit data mapping")
    if "strategy." in source:
        unsupported.append("strategy order calls are descriptive only; INTRA will not execute them")
    if "import " in source:
        unsupported.append("imported libraries are not resolved locally")

    return {
        "name": name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] or "untitled.pine",
        "title": title[:160],
        "version": int(version_match.group(1)) if version_match else None,
        "declaration": declaration,
        "inputs": inputs[:100],
        "plots": plots[:50],
        "alerts": alerts[:20],
        "strategy_actions": strategy_actions[:50],
        "function_count": len(functions),
        "lines": source.count("\n") + 1,
        "warnings": unsupported,
        "read_only": True,
    }
