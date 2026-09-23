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
