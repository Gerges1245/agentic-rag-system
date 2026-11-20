from agentic_rag import agentic_answer


SOURCE = "c24d04d419f5_summary_demo.txt"


def main():
    result = agentic_answer(
        question=(
            f"Summarize the indexed document {SOURCE}. "
            "Include the project requirements and grading."
        ),
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
            action["summary"],
        )

    print("\nSTATUS:", result["status"])
    print("\nANSWER")
    print(result["answer"])

    print("\nSOURCES")
    for source in result["sources"]:
        print(source["citation_id"], source["source"])
        print("Quote:", source["supporting_quote"])

    assert result["status"] == "answered"
    assert result["tool_steps"] <= 6

    assert any(
        action["tool"] == "summarize_document"
        and action["status"] == "success"
        for action in result["actions"]
    ), "The agent did not execute the summary tool."

    assert any(
        source["source"] == SOURCE
        for source in result["sources"]
    ), "The expected document was not cited."

    print("\nAGENTIC SUMMARY TEST PASSED")


if __name__ == "__main__":
    main()