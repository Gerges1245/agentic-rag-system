import json

from agentic_rag import agentic_answer


def main():
    question = (
        "First discover the available uploaded CSV files. "
        "Inspect the columns of agentic_scores.csv, then report "
        "the mean score for rows whose group is A."
    )

    result = agentic_answer(
        question=question,
        course="database_systems",
        max_steps=6,
    )

    print("\nPLAN")
    for number, step in enumerate(result["plan"], 1):
        print(f"{number}. {step}")

    print("\nACTIONS")
    for action in result["actions"]:
        print(
            f"{action['step']}. {action['tool']} "
            f"| {action['status']}"
        )
        print("Arguments:", json.dumps(action["arguments"]))
        print("Result:", action["summary"])

    print("\nSTATUS:", result["status"])
    print("TOOL STEPS:", result["tool_steps"])
    print("STOP REASON:", result["stop_reason"])
    print("\nANSWER\n", result["answer"])

    print("\nSOURCES")
    for source in result["sources"]:
        print(source["citation_id"], source["source"])
        print("Quote:", source["supporting_quote"])

    assert result["status"] == "answered", result["status"]
    assert result["tool_steps"] <= 6

    successful_tools = {
        action["tool"]
        for action in result["actions"]
        if action["status"] == "success"
    }

    assert {
        "list_csv_files",
        "analyze_csv",
    }.issubset(successful_tools), (
        "Expected CSV discovery and analysis."
    )

    assert any(
        source["type"] == "CSV_ANALYSIS"
        and source["source"] == "agentic_scores.csv"
        and source["operation"] == "mean"
        and source["column"] == "score"
        and source["filter_column"] == "group"
        and source["filter_value"] == "A"
        and source["matched_rows"] == 2
        and source["result"] == "90"
        for source in result["sources"]
    ), "Expected a cited mean of 90 for group A."

    print("\nAGENTIC CSV TEST PASSED")


if __name__ == "__main__":
    main()