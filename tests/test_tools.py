from agent_tools import calculate, search_materials


def main():
    assert calculate("(120 + 80) / 4")["result"] == 50
    assert calculate("-3 * (2 + 4)")["result"] == -18

    for expression in ["1 / 0", "print(123)", "2 ** 100"]:
        try:
            calculate(expression)
        except ValueError:
            print(f"Correctly rejected: {expression}")
        else:
            raise AssertionError(
                f"Unsafe expression accepted: {expression}"
            )

    print("\nCALCULATOR TESTS PASSED")

    result = search_materials(
        query="What is a primary key?",
        course="database_systems",
        top_k=2,
    )

    print(f"\nRetrieved chunks: {result['count']}")

    for chunk in result["chunks"]:
        print("\nCitation:", chunk["citation_id"])
        print("File:", chunk["source"])
        print("Page:", chunk["page"])
        print("Text:", chunk["text"][:300])

    if result["count"] == 0:
        raise AssertionError(
            "No passages retrieved. Check the copied database."
        )

    print("\nRETRIEVAL TEST PASSED")


if __name__ == "__main__":
    main()