import json
import sys

from agentic_rag import agentic_answer


TESTS = {
    "calc": (
        "Calculate (120 + 80) / 4.",
        "calculate",
    ),
    "search": (
        "What is a primary key?",
        "search_materials",
    ),
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "calc"

    if mode not in TESTS:
        raise ValueError("Choose calc or search.")

    question, expected_tool = TESTS[mode]

    result = agentic_answer(
        question=question,
        course="database_systems",
        max_steps=6,
    )

    print("\nPLAN")
    for number, step in enumerate(result["plan"], 1):
        print(f"{number}. {step}")

    print("\nACTIONS")
    print(json.dumps(result["actions"], indent=2))

    print("\nSTATUS:", result["status"])
    print("TOOL STEPS:", result["tool_steps"])
    print("STOP REASON:", result["stop_reason"])

    print("\nANSWER\n", result["answer"])

    print("\nSOURCES")
    for source in result["sources"]:
        print(
            source["citation_id"],
            source["source"],
            source["page"],
        )
        print("Quote:", source["supporting_quote"])

    assert result["status"] == "answered", result["status"]
    assert result["tool_steps"] <= 6

    assert any(
        action["tool"] == expected_tool
        and action["status"] == "success"
        for action in result["actions"]
    ), "The expected tool was not executed successfully."

    if mode == "calc":
        assert any(
            source["type"] == "CALCULATION"
            and float(
                source["text"].rsplit(" = ", 1)[1]
            ) == 50
            for source in result["sources"]
        ), "Expected a verified calculation result of 50."

    else:
        assert any(
            source["type"] != "CALCULATION"
            for source in result["sources"]
        ), "Expected document evidence."

    print("\nAGENTIC TEST PASSED:", mode)


if __name__ == "__main__":
    main()