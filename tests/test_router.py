from app.agents.router import RouterAgent


def test_router_routes_document_queries_to_retrieval():
    decision = RouterAgent().route_query("What does the contract say about termination?")

    assert decision.route == "retrieval"
    assert "retrieval" in decision.selected_agents


def test_router_routes_data_queries_and_mult_agent():
    decision = RouterAgent().route_query("Compare the PDF revenue with the Excel Q4 totals and summarize the trend.")

    assert decision.route == "multi_agent"
    assert "retrieval" in decision.selected_agents
    assert "data" in decision.selected_agents
    assert decision.needs_parallel is True


def test_router_routes_vision_queries():
    decision = RouterAgent().route_query("What does this architecture diagram show?")

    assert decision.route == "vision"
    assert "vision" in decision.selected_agents


def test_router_does_not_over_trigger_multi_agent_for_generic_and_queries():
    decision = RouterAgent().route_query("Explain the policy and the process for the team.")

    assert decision.route == "retrieval"
    assert decision.selected_agents == ["retrieval"]


def test_vector_store_falls_back_to_lexical_search_without_embeddings():
    from app.retrieval.vector_store import LocalVectorStore
    from app.ingestion.models import Evidence

    store = LocalVectorStore()
    store.add_documents([
        Evidence(document_id="1", file_name="report.txt", file_type="txt", source_type="text", content="Revenue increased by 12% in Q4."),
        Evidence(document_id="2", file_name="plan.txt", file_type="txt", source_type="text", content="The team plans a new launch next quarter."),
    ])

    results = store.search("Q4 revenue", top_k=1)

    assert results
    assert results[0].file_name == "report.txt"
