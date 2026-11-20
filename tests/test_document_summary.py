import re

from document_tools import summarize_document


SOURCE = "c24d04d419f5_summary_demo.txt"
COURSE = "database_systems"


def main():
    result = summarize_document(
        source=SOURCE,
        course=COURSE,
        max_points=8,
    )

    print("\nSOURCE:", result["source"])

    print(
        "COVERED CHUNKS:",
        f"{result['covered_chunks']}/{result['total_chunks']}",
    )

    print("COVERAGE:", result["coverage"])
    print("MODEL CALLS:", result["model_calls"])

    print("\nSUMMARY")
    print(result["summary"])

    print("\nSOURCES")
    for source in result["sources"]:
        print(source["citation_id"], source["source"])
        print("Quote:", source["supporting_quotes"][0])

    assert result["covered_chunks"] == result["total_chunks"] == 1
    assert result["sources"]

    statements = " ".join(
        point["statement"]
        for point in result["points"]
    ).lower()

    assert "bookstore" in statements, (
        "The summary should mention the project topic."
    )

    assert re.search(r"\b(?:three|3)\b", statements), (
        "The summary should preserve the team size."
    )

    for number in ("40", "35", "25"):
        assert re.search(
            rf"\b{number}\b", statements
        ), f"Missing grading value: {number}"

    print("\nDOCUMENT SUMMARY TEST PASSED")


if __name__ == "__main__":
    main()