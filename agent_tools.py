"""Executable tools for the Agentic RAG system."""

import ast
import hashlib
import json
import math
import operator


def search_materials(query, course, file_type="All Files", top_k=4):
    """Search one course and return passages with stable citation IDs."""
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 2000:
        raise ValueError(
            "Query must contain between 1 and 2000 characters."
        )

    if type(top_k) is not int or not 1 <= top_k <= 8:
        raise ValueError(
            "top_k must be an integer between 1 and 8."
        )

    if file_type not in {"All Files", "PDF", "DOCX", "CSV", "TXT"}:
        raise ValueError("Unsupported file type.")

    from rag_core import COURSES, retrieve_context

    if course not in COURSES.values():
        raise ValueError("Unknown course.")

    chunks = retrieve_context(
        query.strip(), course, file_type, top_k
    )

    results = []

    for chunk in chunks:
        identity = json.dumps(
            [
                chunk["course"],
                chunk["source"],
                chunk["page"],
                chunk["text"],
            ],
            ensure_ascii=False,
        )

        citation_id = (
            "S"
            + hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()[:12]
        )

        results.append({
            "citation_id": citation_id,
            **chunk,
        })

    return {
        "query": query.strip(),
        "count": len(results),
        "chunks": results,
    }


def calculate(expression):
    """Calculate using +, -, *, / and parentheses without eval."""
    if (
        not isinstance(expression, str)
        or not 1 <= len(expression.strip()) <= 200
    ):
        raise ValueError(
            "Expression must contain between 1 and 200 characters."
        )

    expression = expression.strip()

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            "Invalid arithmetic expression."
        ) from exc

    if len(list(ast.walk(tree))) > 40:
        raise ValueError("Expression is too complex.")

    operations = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    def evaluate(node):
        if (
            isinstance(node, ast.Constant)
            and type(node.value) in (int, float)
        ):
            value = node.value

        elif (
            isinstance(node, ast.BinOp)
            and type(node.op) in operations
        ):
            value = operations[type(node.op)](
                evaluate(node.left),
                evaluate(node.right),
            )

        elif (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
        ):
            value = evaluate(node.operand)

            if isinstance(node.op, ast.USub):
                value = -value

        else:
            raise ValueError(
                "Only numbers, +, -, *, / "
                "and parentheses are allowed."
            )

        if abs(value) > 1e12 or not math.isfinite(value):
            raise ValueError(
                "Numbers and intermediate results must be "
                "finite and <= 1e12 in magnitude."
            )

        return value

    try:
        result = evaluate(tree.body)
    except ZeroDivisionError as exc:
        raise ValueError(
            "Division by zero is not allowed."
        ) from exc

    return {
        "expression": expression,
        "result": result,
    }