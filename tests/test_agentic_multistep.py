from decimal import Decimal

from agentic_rag import agentic_answer


def main():
    question = (
        "Find the uploaded file agentic_scores.csv. "
        "Calculate the overall mean score and the mean score "
        "for group A. Then calculate the difference between "
        "the group A mean and the overall mean. "
        "Show the results with sources."
    )

    result = agentic_answer(
        question=question,
        course="database_systems",
        max_steps=6,
    )

    print("\nPLAN")
    for step in result["plan"]:
        print("-", step)

    print("\nACTIONS")
    for action in result["actions"]:
        print(
            action["step"],
            action["tool"],
            action["status"],
        )
        print("Arguments:", action["arguments"])
        print("Result:", action["summary"])

    print("\nSTATUS:", result["status"])
    print("TOOL STEPS:", result["tool_steps"])
    print("STOP REASON:", result["stop_reason"])

    print("\nANSWER")
    print(result["answer"])

    print("\nSOURCES")
    for source in result["sources"]:
        print(source["citation_id"], source["source"])
        print("Quote:", source["supporting_quote"])

    assert result["status"] == "answered"
    assert result["tool_steps"] <= 6

    successful_tools = {
        action["tool"]
        for action in result["actions"]
        if action["status"] == "success"
    }

    assert "analyze_csv" in successful_tools, (
        "The agent did not analyze the CSV."
    )

    assert "calculate" in successful_tools, (
        "The agent did not use the calculator."
    )

    csv_sources = [
        source
        for source in result["sources"]
        if source["type"] == "CSV_ANALYSIS"
        and source["source"] == "agentic_scores.csv"
        and source["operation"] == "mean"
        and source["column"] == "score"
    ]

    assert any(
        source["filter_column"] == ""
        and Decimal(source["result"]) == Decimal("80")
        for source in csv_sources
    ), "The overall mean of 80 was not cited."

    assert any(
        source["filter_column"] == "group"
        and source["filter_value"] == "A"
        and Decimal(source["result"]) == Decimal("90")
        for source in csv_sources
    ), "The group A mean of 90 was not cited."

    assert any(
        source["type"] == "CALCULATION"
        and Decimal(
            source["text"].rsplit(" = ", 1)[1]
        ) == Decimal("10")
        for source in result["sources"]
    ), "The calculated difference of 10 was not cited."

    print("\nMULTISTEP TEST PASSED")


if __name__ == "__main__":
    main()