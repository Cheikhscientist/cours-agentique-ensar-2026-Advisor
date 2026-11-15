"""Tests rapides, sans GPU, du contrat et des fonctions de l'agent Advisor."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.agent_advisor.agent import CustomAgentPlugin
from src.agents.agent_advisor.tools import (
    calculer_total_ects,
    format_domain_response,
    process_domain_logic,
)
from src.core.state import AgentState


def _state(messages: list[HumanMessage]) -> AgentState:
    """Construire un état complet conforme au TypedDict du professeur."""
    return {
        "messages": messages,
        "next_agent": None,
        "current_agent": None,
        "context_data": {},
    }


def _doc(code: str, ects: object = 5) -> Document:
    return Document(
        page_content=f"L'UE {code} porte sur SQL et la gestion des données.",
        metadata={
            "code_ue": code,
            "source": f"{code}_Syllabus.md",
            "semestre": "S9",
            "filiere": "SD",
            "ects": ects,
        },
    )


def test_contrat_plugin() -> None:
    agent = CustomAgentPlugin()
    assert agent.name == "advisor"
    assert len(agent.description) >= 15
    assert callable(agent.create_node())


def test_calcul_ects() -> None:
    assert calculer_total_ects.invoke({"credits": [5, 4]}) == 9
    assert calculer_total_ects.invoke({"credits": []}) == 0
    with pytest.raises(ValueError):
        calculer_total_ects.invoke({"credits": [-1]})


def test_dedoublonnage() -> None:
    result = process_domain_logic("Combien d'ECTS ?", [_doc("SD_211"), _doc("SD_211")])
    assert result["status"] == "success"
    assert result["nombre_ue"] == 1
    assert result["total_ects"] == 5


def test_ue_manquante_et_ects_invalides() -> None:
    missing = process_domain_logic(
        "SD_211 et SD_999", [_doc("SD_211")], ["SD_211", "SD_999"]
    )
    assert missing["status"] == "incomplete"
    assert missing["total_ects"] is None
    assert missing["codes_manquants"] == ["SD_999"]
    invalid = process_domain_logic("ECTS", [_doc("SD_211", "inconnu")])
    assert invalid["total_ects"] is None


def test_aucun_document() -> None:
    result = process_domain_logic("Question", [])
    assert result["status"] == "no_documents"
    assert "Aucune UE" in format_domain_response(result)


@patch("src.agents.agent_advisor.agent.ChatOllama")
@patch("src.agents.agent_advisor.agent.get_vector_store")
def test_noeud_mocke(
    mock_store: MagicMock, mock_llm: MagicMock
) -> None:
    mock_store.return_value.as_retriever.return_value.invoke.return_value = [
        _doc("SD_211")
    ]
    mock_llm.return_value.invoke.return_value = AIMessage(
        content="SD_211 développe des compétences de données."
    )
    node = CustomAgentPlugin().create_node()
    output = node(_state([HumanMessage(content="Parle-moi de SD_211")]))
    assert output["current_agent"] == "advisor"
    assert len(output["messages"]) == 1
    assert "SD_211" in output["messages"][0].content
    kwargs = mock_store.return_value.as_retriever.call_args.kwargs["search_kwargs"]
    assert kwargs["k"] == 2
    assert kwargs["filter"] == {"code_ue": "SD_211"}


@patch("src.agents.agent_advisor.agent.ChatOllama")
@patch("src.agents.agent_advisor.agent.get_vector_store")
def test_hors_documents_sans_llm(
    mock_store: MagicMock, mock_llm: MagicMock
) -> None:
    mock_store.return_value.as_retriever.return_value.invoke.return_value = []
    output = CustomAgentPlugin().create_node()(
        _state([HumanMessage(content="Quelles UE pour la BI ?")])
    )
    assert "Aucune UE" in output["messages"][0].content
    mock_llm.return_value.invoke.assert_not_called()


@patch("src.agents.agent_advisor.agent.ChatOllama")
@patch("src.agents.agent_advisor.agent.get_vector_store")
def test_erreur_chromadb(
    mock_store: MagicMock, mock_llm: MagicMock
) -> None:
    mock_store.side_effect = RuntimeError("Chroma indisponible")
    output = CustomAgentPlugin().create_node()(
        _state([HumanMessage(content="UE SD_211 ?")])
    )
    assert "indisponible" in output["messages"][0].content
    mock_llm.return_value.invoke.assert_not_called()


@patch("src.agents.agent_advisor.agent.ChatOllama")
def test_question_vide(mock_llm: MagicMock) -> None:
    output = CustomAgentPlugin().create_node()(_state([]))
    assert "Aucune question" in output["messages"][0].content
    mock_llm.return_value.invoke.assert_not_called()
