from csv_tools import list_csv_files, analyze_csv


COURSE = "database_systems"
SOURCE = "agentic_scores.csv"


def main():
    inventory = list_csv_files(COURSE)

    assert any(
        item["source"] == SOURCE
        for item in inventory["files"]
    ), "The test CSV file was not found."

    print("CSV FILE FOUND:", SOURCE)

    tests = [
        ("count_rows", "", "4"),
        ("sum", "score", "320"),
        ("mean", "score", "80"),
        ("min", "score", "60"),
        ("max", "score", "100"),
    ]

    for operation, column, expected in tests:
        result = analyze_csv(
            source=SOURCE,
            course=COURSE,
            operation=operation,
            column=column,
        )

        print(f"{operation}: {result['result']}")

        assert str(result["result"]) == expected, (
            f"{operation}: expected {expected}, "
            f"got {result['result']}"
        )

    filtered = analyze_csv(
        source=SOURCE,
        course=COURSE,
        operation="mean",
        column="score",
        filter_column="group",
        filter_value="A",
    )

    print("Filtered mean:", filtered["result"])

    assert filtered["result"] == "90"
    assert filtered["matched_rows"] == 2

    print("Source:", filtered["source"])
    print("Citation:", filtered["citation_id"])
    print("\nCSV TOOL TESTS PASSED")


if __name__ == "__main__":
    main()