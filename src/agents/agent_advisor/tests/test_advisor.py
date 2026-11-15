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


@patch("src.agents.agent_advisor.agent.ChatOllama")
@patch("src.agents.agent_advisor.agent.get_vector_store")
def test_business_intelligence_priorite_intitule(
    mock_store: MagicMock, mock_llm: MagicMock
) -> None:
    """Le catalogue retrouve l'UE BI et écarte le syllabus cyber."""
    store = mock_store.return_value
    store._collection.get.return_value = {
        "metadatas": [
            {"code_ue": "SD_211", "source": "SD_211_Business_Intelligence_Outils.md"},
            {"code_ue": "SSI_212", "source": "SSI_212_Methodes_Analyse_Risques_Cyber.md"},
            {"code_ue": "SD_107", "source": "SD_107_Gouvernance_Ethique_Donnees.md"},
        ]
    }
    store.as_retriever.return_value.invoke.return_value = [
        Document(
            page_content="Tableaux de bord et reporting décisionnel.",
            metadata={
                "source": "SD_211_Business_Intelligence_Outils.md",
                "code_ue": "SD_211",
                "ects": 5,
            },
        )
    ]
    mock_llm.return_value.invoke.return_value = AIMessage(
        content="SD_211 est pertinente pour la Business Intelligence."
    )

    result = CustomAgentPlugin().create_node()(
        _state([HumanMessage(content="Je veux travailler en Business Intelligence")])
    )
    assert "SD_211" in result["messages"][0].content
    assert "SSI_212" not in result["messages"][0].content
    store._collection.get.assert_called_once_with(include=["metadatas"])
    kwargs = store.as_retriever.call_args.kwargs["search_kwargs"]
    assert kwargs == {"k": 2, "filter": {"code_ue": "SD_211"}}


@patch("src.agents.agent_advisor.agent.ChatOllama")
@patch("src.agents.agent_advisor.agent.get_vector_store")
def test_absence_catalogue_recherche_semantique(
    mock_store: MagicMock, mock_llm: MagicMock
) -> None:
    """Si l'accès catalogue n'est pas disponible, le RAG classique reste utilisable."""
    store = mock_store.return_value
    store._collection.get.side_effect = RuntimeError("Catalogue indisponible")
    store.as_retriever.return_value.invoke.return_value = [_doc("SD_107")]
    mock_llm.return_value.invoke.return_value = AIMessage(content="UE documentée : SD_107")

    result = CustomAgentPlugin().create_node()(
        _state([HumanMessage(content="Quelles UE pour la gestion des données ?")])
    )
    assert "SD_107" in result["messages"][0].content
    assert store.as_retriever.call_args.kwargs["search_kwargs"] == {"k": 2}


@patch("src.agents.agent_advisor.agent.ChatOllama")
@patch("src.agents.agent_advisor.agent.get_vector_store")
def test_plusieurs_codes_ue_exactement_filtres(
    mock_store: MagicMock, mock_llm: MagicMock
) -> None:
    """Les recherches multi-UE utilisent chacune un filtre exact et k=2."""
    store = mock_store.return_value
    store.as_retriever.return_value.invoke.side_effect = [
        [_doc("SD_322")],
        [_doc("SD_106")],
    ]
    mock_llm.return_value.invoke.return_value = AIMessage(content="Deux UE retrouvées.")
    output = CustomAgentPlugin().create_node()(
        _state([HumanMessage(content="ECTS des UE SD_322 et SD_106 ?")])
    )
    assert output["current_agent"] == "advisor"
    calls = store.as_retriever.call_args_list
    assert len(calls) == 2
    assert [call.kwargs["search_kwargs"] for call in calls] == [
        {"k": 2, "filter": {"code_ue": "SD_322"}},
        {"k": 2, "filter": {"code_ue": "SD_106"}},
    ]


@patch("src.agents.agent_advisor.agent.ChatOllama")
@patch("src.agents.agent_advisor.agent.get_vector_store")
def test_ct02_ects_semestre_sans_second_llm(
    mock_store: MagicMock, mock_llm: MagicMock
) -> None:
    """Les ECTS et le semestre proviennent des métadonnées, sans inférence."""
    mock_store.return_value.as_retriever.return_value.invoke.return_value = [
        Document(
            page_content="UE de sécurité systèmes et réseaux.",
            metadata={
                "code_ue": "SSI_107", "source": "SSI_107_Securite_Systemes_Reseaux.md",
                "semestre": "S6", "ects": 5,
            },
        ),
    ]
    output = CustomAgentPlugin().create_node()(
        _state([HumanMessage(content=(
            "Combien de crédits ECTS rapporte l'UE SSI_107 et "
            "à quel semestre est-elle dispensée ?"
        ))])
    )
    answer = str(output["messages"][0].content)
    assert "[Agent advisor]" in answer
    assert "SSI_107" in answer
    assert "5 ECTS" in answer
    assert "S6" in answer
    mock_llm.return_value.invoke.assert_not_called()


@patch("src.agents.agent_advisor.agent.ChatOllama")
@patch("src.agents.agent_advisor.agent.get_vector_store")
def test_ct11_total_et_solde_calcules_sans_second_llm(
    mock_store: MagicMock, mock_llm: MagicMock
) -> None:
    """Le total et le solde sont calculés en Python, pas par Ollama."""
    mock_store.return_value.as_retriever.return_value.invoke.side_effect = [
        [_doc("SD_321", 2)], [_doc("SSI_322", 5)], [_doc("MGT_102", 7)],
    ]
    output = CustomAgentPlugin().create_node()(
        _state([HumanMessage(content=(
            "Si je choisis les UEs SD_321, SSI_322 et MGT_102, "
            "calcule le total d'ECTS et le solde pour atteindre 30 ECTS."
        ))])
    )
    answer = str(output["messages"][0].content)
    assert "14 ECTS" in answer
    assert "Reste 16 ECTS" in answer
    mock_llm.return_value.invoke.assert_not_called()


@patch("src.agents.agent_advisor.agent.ChatOllama")
@patch("src.agents.agent_advisor.agent.get_vector_store")
def test_total_incomplet_ne_devient_pas_un_total_fictif(
    mock_store: MagicMock, mock_llm: MagicMock
) -> None:
    """Une UE sans ECTS valide bloque le calcul complet."""
    mock_store.return_value.as_retriever.return_value.invoke.return_value = [
        _doc("SD_321", "inconnu")
    ]
    output = CustomAgentPlugin().create_node()(
        _state([HumanMessage(content="Calcule le total ECTS de SD_321")])
    )
    answer = str(output["messages"][0].content)
    assert "insuffisantes" in answer
    assert "Total :" not in answer
    mock_llm.return_value.invoke.assert_not_called()


@patch("src.agents.agent_advisor.agent.ChatOllama")
@patch("src.agents.agent_advisor.agent.get_vector_store")
def test_demande_de_recommandation_reste_confiee_au_llm(
    mock_store: MagicMock, mock_llm: MagicMock
) -> None:
    """Une vraie demande de conseil ne doit pas être traitée comme un calcul."""
    mock_store.return_value.as_retriever.return_value.invoke.return_value = [
        _doc("SD_321", 2)
    ]
    mock_llm.return_value.invoke.return_value = AIMessage(content="Conseil fondé sur SD_321.")
    CustomAgentPlugin().create_node()(
        _state([HumanMessage(content="Recommande SD_321 pour un métier; combien d'ECTS ?")])
    )
    mock_llm.return_value.invoke.assert_called_once()

@patch("src.agents.agent_advisor.agent.ChatOllama")
@patch("src.agents.agent_advisor.agent.get_vector_store")
def test_ct10_recherche_distincte_s9_s10(
    mock_store: MagicMock, mock_llm: MagicMock
) -> None:
    """SD_322 reste en S9 et SD_327 reste en S10, sans appel SLM."""
    store = mock_store.return_value
    store._collection.get.return_value = {
        "metadatas": [
            {"code_ue": "SD_322", "source": "SD_322_Big_Data_Smart_Data.md", "semestre": "S9"},
            {"code_ue": "SD_327", "source": "SD_327_Machine_Learning_Operations_MLOps.md", "semestre": "S10"},
        ]
    }
    store.as_retriever.return_value.invoke.side_effect = [
        [Document(page_content="Big Data Spark", metadata={
            "code_ue": "SD_322", "source": "SD_322_Big_Data_Smart_Data.md",
            "semestre": "S9", "ects": 5, "filiere": "Data Science",
        })],
        [Document(page_content="MLOps MLflow", metadata={
            "code_ue": "SD_327", "source": "SD_327_Machine_Learning_Operations_MLOps.md",
            "semestre": "S10", "ects": 3, "filiere": "Data Science",
        })],
    ]
    question = (
        "Je souhaite me spécialiser en Big Data & MLOps en dernière année, "
        "quelles filières et UEs sélectionner au Semestre 9 et 10 ?"
    )
    output = CustomAgentPlugin().create_node()(_state([HumanMessage(content=question)]))
    answer = str(output["messages"][0].content)
    assert "S9 :" in answer and "SD_322" in answer
    assert "S10 :" in answer and "SD_327" in answer
    assert answer.index("SD_322") < answer.index("S10 :") < answer.index("SD_327")
    kwargs = [c.kwargs["search_kwargs"] for c in store.as_retriever.call_args_list]
    assert kwargs == [
        {"k": 2, "filter": {"$and": [{"semestre": "S9"}, {"code_ue": "SD_322"}]}},
        {"k": 2, "filter": {"$and": [{"semestre": "S10"}, {"code_ue": "SD_327"}]}},
    ]
    mock_llm.return_value.invoke.assert_not_called()


@patch("src.agents.agent_advisor.agent.ChatOllama")
@patch("src.agents.agent_advisor.agent.get_vector_store")
def test_ct10_semestre_introuvable_sans_invention(
    mock_store: MagicMock, mock_llm: MagicMock
) -> None:
    """Un syllabus S9 ne doit jamais être présenté comme un cours S10."""
    store = mock_store.return_value
    store._collection.get.return_value = {
        "metadatas": [
            {"code_ue": "SD_322", "source": "SD_322_Big_Data_Smart_Data.md", "semestre": "S9"},
        ]
    }
    store.as_retriever.return_value.invoke.side_effect = [
        [Document(page_content="Big Data", metadata={
            "code_ue": "SD_322", "source": "SD_322_Big_Data_Smart_Data.md", "semestre": "S9"
        })],
        [],
    ]
    question = "Quelles filières et UE sélectionner au Semestre 9 et 10 en Big Data MLOps ?"
    output = CustomAgentPlugin().create_node()(_state([HumanMessage(content=question)]))
    answer = str(output["messages"][0].content)
    s10_section = answer.split("S10 :", maxsplit=1)[1].split("Seules les UE", maxsplit=1)[0]
    assert "Aucune UE pertinente confirmée" in s10_section
    assert "SD_322" not in s10_section
    mock_llm.return_value.invoke.assert_not_called()


@patch("src.agents.agent_advisor.agent.ChatOllama")
@patch("src.agents.agent_advisor.agent.get_vector_store")
def test_ct10_ne_pas_accepter_mauvais_semestre(
    mock_store: MagicMock, mock_llm: MagicMock
) -> None:
    """Même si une récupération retourne un mauvais semestre, on l'écarte."""
    store = mock_store.return_value
    store._collection.get.return_value = {"metadatas": []}
    s9_document = Document(page_content="Big Data", metadata={
        "code_ue": "SD_322", "source": "SD_322_Big_Data_Smart_Data.md", "semestre": "S9"
    })
    store.as_retriever.return_value.invoke.side_effect = [[s9_document], [s9_document]]
    output = CustomAgentPlugin().create_node()(_state([HumanMessage(
        content="Je souhaite me spécialiser en Big Data MLOps au Semestre 9 et 10"
    )]))
    answer = str(output["messages"][0].content)
    assert "S9 :" in answer and "SD_322" in answer
    assert "Aucune UE pertinente confirmée" in answer.split("S10 :", maxsplit=1)[1]
    mock_llm.return_value.invoke.assert_not_called()
