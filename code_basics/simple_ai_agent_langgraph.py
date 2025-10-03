from typing import TypedDict
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END


class PortfolioState(TypedDict):
    amount_usd: float
    total_usd: float
    total_inr: float
def calc_total(state: PortfolioState) -> PortfolioState:
    state['total_usd'] = state['amount_usd'] * 1.08
    return state

def convert_to_inr(state: PortfolioState) -> PortfolioState:
    state['total_inr'] = state['total_usd'] * 85
    return state

builder = StateGraph(PortfolioState)

builder.add_node("calc_total_node", calc_total)
builder.add_node("convert_to_inr_node", convert_to_inr)

builder.add_edge(START, "calc_total_node")
builder.add_edge("calc_total_node", "convert_to_inr_node")
builder.add_edge("convert_to_inr_node", END)

graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

graph.invoke({"amount_usd": 100000})
{'amount_usd': 100000, 'total_usd': 108000.0, 'total_inr': 9180000.0}