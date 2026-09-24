"""Code-aware chunking for Python: one module overview plus one chunk per top-level function/class (or method)."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from app.ingestion.chunking import DEFAULT_MAX_CHARS, chunk_paragraphs, split_paragraphs


@dataclass
class CodeChunk:
    content: str
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_python(source: str, module_name: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[CodeChunk]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        chunks = chunk_paragraphs(split_paragraphs(source), max_chars) or [source.strip() or "Empty file."]
        return [CodeChunk(content=chunk, metadata={"syntax_error": str(exc)}) for chunk in chunks]

    lines = source.splitlines()
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    imports = [ast.unparse(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]

    overview = [f"Python module {module_name}"]
    module_doc = ast.get_docstring(tree)
    if module_doc:
        overview.append(f"Docstring: {module_doc}")
    overview.append(f"Imports: {'; '.join(imports) if imports else 'none'}")
    overview.append("Functions:" if functions else "Functions: none")
    overview.extend(f"  {_signature(node)}" for node in functions)
    overview.append("Classes:" if classes else "Classes: none")
    for node in classes:
        methods = [child.name for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]
        overview.append(f"  {_class_signature(node)}" + (f" — methods: {', '.join(methods)}" if methods else ""))

    chunks = [
        CodeChunk(
            content="\n".join(overview),
            section="module overview",
            metadata={
                "symbol_kind": "module",
                "functions": [node.name for node in functions],
                "classes": [node.name for node in classes],
                "imports": imports,
                "start_line": 1,
                "end_line": len(lines),
            },
        )
    ]

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(_symbol_chunk(node, lines, "function", node.name))
        elif isinstance(node, ast.ClassDef):
            segment = _segment(node, lines)
            if len(segment) <= max_chars:
                chunks.append(_symbol_chunk(node, lines, "class", node.name))
                continue
            header = [_class_signature(node) + ":"]
            doc = ast.get_docstring(node)
            if doc:
                header.append(f'    """{doc}"""')
            header.extend(
                f"    {_signature(child)}: ..."
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            )
            chunks.append(
                CodeChunk(
                    content="\n".join(header),
                    section=f"class: {node.name}",
                    metadata={"symbol_kind": "class", "symbol": node.name, **_line_range(node)},
                )
            )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append(_symbol_chunk(child, lines, "method", f"{node.name}.{child.name}"))
    return chunks


def _symbol_chunk(node: ast.AST, lines: list[str], kind: str, name: str) -> CodeChunk:
    return CodeChunk(
        content=_segment(node, lines),
        section=f"{kind}: {name}",
        metadata={"symbol_kind": kind, "symbol": name, **_line_range(node)},
    )


def _line_range(node: Any) -> dict[str, int]:
    start = min([node.lineno] + [decorator.lineno for decorator in getattr(node, "decorator_list", [])])
    return {"start_line": start, "end_line": node.end_lineno or node.lineno}


def _segment(node: Any, lines: list[str]) -> str:
    span = _line_range(node)
    return "\n".join(lines[span["start_line"] - 1 : span["end_line"]])


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){returns}"


def _class_signature(node: ast.ClassDef) -> str:
    bases = ", ".join(ast.unparse(base) for base in node.bases)
    return f"class {node.name}({bases})" if bases else f"class {node.name}"
