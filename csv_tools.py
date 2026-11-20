"""CSV inventory and exact numeric analysis for the Agentic RAG project."""

import csv
import hashlib
import io
import json
import re
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000
OPERATIONS = {"describe", "count_rows", "sum", "mean", "min", "max"}


def _course_folder(course):
    from rag_core import COURSES

    if course not in COURSES.values():
        raise ValueError("Unknown course.")
    root = (BASE_DIR / "data" / "uploads").resolve()
    folder = (root / course).resolve()
    if not root.is_relative_to(BASE_DIR.resolve()) or folder.parent != root:
        raise ValueError("CSV folder must stay inside this project's upload directory.")
    return folder


def list_csv_files(course):
    """List up to 100 CSV files uploaded to the selected course."""
    folder = _course_folder(course)
    files = []
    if folder.is_dir():
        for path in sorted(folder.iterdir()):
            if (path.is_file() and path.suffix.lower() == ".csv"
                    and path.resolve().parent == folder):
                files.append({"source": path.name, "size_bytes": path.stat().st_size})
    return {
        "course": course, "count": len(files), "files": files[:100],
        "truncated": len(files) > 100,
    }


def _read_csv(source, course):
    if (not isinstance(source, str) or not source
            or "/" in source or "\\" in source or ":" in source
            or Path(source).suffix.lower() != ".csv"):
        raise ValueError("source must be a CSV filename from list_csv_files.")

    folder = _course_folder(course)
    path = (folder / source).resolve()
    if path.parent != folder:
        raise ValueError("CSV file must stay inside the selected course folder.")
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found in selected course: {source}")

    with path.open("rb") as handle:
        raw = handle.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("CSV files must be no larger than 10 MiB.")

    try:
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
        if not reader.fieldnames:
            raise ValueError("The CSV file needs a header row.")
        columns = [name.strip() for name in reader.fieldnames]
        if any(not name for name in columns) or len(set(columns)) != len(columns):
            raise ValueError("CSV column names must be non-empty and unique.")
        reader.fieldnames = columns
        rows = []
        for number, row in enumerate(reader, 1):
            if number > MAX_ROWS:
                raise ValueError("CSV files must contain at most 100000 data records.")
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Data record {number} does not match the header.")
            rows.append((number, row))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("Provide a valid, comma-separated UTF-8 CSV file.") from exc

    return hashlib.sha256(raw).hexdigest(), columns, rows


def analyze_csv(source, course, operation="describe", column="",
                filter_column="", filter_value=""):
    """Describe or aggregate an entire CSV, optionally using an exact text filter.

    count_rows counts matching data records. Numeric operations skip blank cells
    and report their count; invalid nonblank numeric cells cause an error.
    Numeric results are strings to retain decimal precision in JSON output.
    """
    if operation not in OPERATIONS:
        raise ValueError("Unsupported CSV operation.")
    if not all(isinstance(value, str) for value in (column, filter_column, filter_value)):
        raise ValueError("Column names and filter values must be strings.")
    digest, columns, rows = _read_csv(source, course)
    if filter_column and filter_column not in columns:
        raise ValueError(f"Unknown filter column. Available columns: {columns}")
    if not filter_column and filter_value:
        raise ValueError("filter_value requires filter_column.")

    matched = [
        (number, row) for number, row in rows
        if not filter_column or row[filter_column].strip() == filter_value.strip()
    ]
    numeric_operations = {"sum", "mean", "min", "max"}
    summary = {
        "source": source, "course": course, "operation": operation,
        "total_rows": len(rows), "matched_rows": len(matched),
        "columns": columns, "column": column,
        "filter_column": filter_column, "filter_value": filter_value,
    }

    if operation == "describe":
        if column:
            raise ValueError("describe does not use a numeric column.")
        summary["result"] = {"rows": len(matched), "columns": columns}
    elif operation == "count_rows":
        if column:
            raise ValueError("count_rows counts records; leave column empty.")
        summary["result"] = len(matched)
    elif operation in numeric_operations:
        if column not in columns:
            raise ValueError(f"Choose a numeric column from: {columns}")
        values, missing = [], 0
        for number, row in matched:
            value = row[column].strip()
            if not value:
                missing += 1
                continue
            if len(value) > 80 or not re.fullmatch(
                r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value
            ):
                raise ValueError(f"Non-numeric value in {column}, data record {number}.")
            try:
                numeric = Decimal(value)
            except InvalidOperation as exc:
                raise ValueError(f"Invalid number in data record {number}.") from exc
            if (not numeric.is_finite() or abs(numeric) > Decimal("1e12")
                    or not -12 <= numeric.as_tuple().exponent <= 12):
                raise ValueError(f"Number outside supported range in data record {number}.")
            values.append(numeric)
        if not values:
            raise ValueError("No nonblank numeric values match this request.")

        with localcontext() as context:
            context.prec = 50
            if operation == "sum":
                value = sum(values, Decimal(0))
            elif operation == "mean":
                value = sum(values, Decimal(0)) / len(values)
            elif operation == "min":
                value = min(values)
            else:
                value = max(values)
            result = format(value, "f")
        if "." in result:
            result = result.rstrip("0").rstrip(".")
        summary.update(
            result=result, numeric_rows=len(values), missing_numeric_rows=missing
        )

    identity = json.dumps([digest, summary], sort_keys=True, ensure_ascii=False)
    citation_id = "S" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    text = "CSV analysis result:\n" + json.dumps(summary, ensure_ascii=False, sort_keys=True)
    return {
        **summary, "citation_id": citation_id, "type": "CSV_ANALYSIS",
        "page": -1, "text": text,
    }
