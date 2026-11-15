"""Agent Advisor : orientation par syllabus, RAG ciblé et calcul ECTS fiable."""

import os
import re
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.core.base_agent import BaseAgentPlugin
from src.core.ingestion import get_vector_store
from src.core.state import AgentState

from .tools import format_domain_response, process_domain_logic


def _extract_codes(question: str) -> list[str]:
    """Repérer les codes UE dans la question sans doublons."""
    return list(
        dict.fromkeys(
            code.upper()
            for code in re.findall(r"\b[A-Z]{2,5}_\d{3}\b", question, re.IGNORECASE)
        )
    )


def _extract_filters(question: str) -> dict[str, str]:
    """Construire des filtres uniquement à partir de valeurs explicites."""
    filters: dict[str, str] = {}
    semester = re.search(r"\bS(?:[1-9]|1[0-2])\b", question, re.IGNORECASE)
    if semester:
        filters["semestre"] = semester.group(0).upper()

    # Une filière doit être écrite explicitement : « filière : SD ».
    # Ne pas deviner un identifiant de filière depuis un métier.
    track = re.search(r"\bfili[eè]re\s*:\s*([A-Za-z0-9_-]+)", question, re.IGNORECASE)
    if track:
        filters["filiere"] = track.group(1)
    return filters


def _chroma_filter(values: dict[str, str]) -> dict[str, Any] | None:
    """Construire une clause ChromaDB valide pour 0, 1 ou N filtres."""
    if not values:
        return None
    if len(values) == 1:
        key, value = next(iter(values.items()))
        return {key: value}
    return {"$and": [{key: value} for key, value in values.items()]}


class CustomAgentPlugin(BaseAgentPlugin):
    """Conseiller pédagogique basé exclusivement sur les syllabus indexés."""

    @property
    def name(self) -> str:
        return "advisor"

    @property
    def description(self) -> str:
        return (
            "Agent d'orientation académique : recommande les UE et parcours selon "
            "les métiers, compétences, semestres et filières ; recherche des codes "
            "UE précis et calcule les crédits ECTS des syllabus."
        )

    def create_node(self) -> Callable[[AgentState], dict[str, Any]]:
        """Créer le nœud LangGraph sans ouvrir ChromaDB avant son premier appel."""
        llm = ChatOllama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:3b"),
            temperature=0.0,
            num_predict=400,
        )
        vector_store: Any = None

        def node(state: AgentState) -> dict[str, Any]:
            nonlocal vector_store
            messages = state.get("messages", [])
            if not messages:
                return {
                    "messages": [AIMessage(content="[advisor] Aucune question reçue.")],
                    "current_agent": self.name,
                }

            last_message = messages[-1]
            user_query = str(last_message.content).strip()
            if not user_query:
                return {
                    "messages": [AIMessage(content="[advisor] Question vide.")],
                    "current_agent": self.name,
                }

            codes = _extract_codes(user_query)
            filters = _extract_filters(user_query)
            docs: list[Document] = []
            retrieval_error = False

            try:
                if vector_store is None:
                    vector_store = get_vector_store(reindex_if_empty=False)

                if codes:
                    # Une requête filtrée par code ; k=2 au maximum par code UE.
                    for code in codes:
                        requested = {**filters, "code_ue": code}
                        kwargs: dict[str, Any] = {"k": 2}
                        metadata_filter = _chroma_filter(requested)
                        if metadata_filter is not None:
                            kwargs["filter"] = metadata_filter
                        retriever = vector_store.as_retriever(search_kwargs=kwargs)
                        docs.extend(retriever.invoke(user_query))
                else:
                    kwargs = {"k": 2}
                    metadata_filter = _chroma_filter(filters)
                    if metadata_filter is not None:
                        kwargs["filter"] = metadata_filter
                    retriever = vector_store.as_retriever(search_kwargs=kwargs)
                    docs = retriever.invoke(user_query)
            except Exception:
                retrieval_error = True
                docs = []

            tool_results = process_domain_logic(user_query, docs, requested_codes=codes)
            if retrieval_error:
                answer = "Recherche documentaire indisponible. Réessaie plus tard."
            elif not docs:
                answer = format_domain_response(tool_results)
            elif codes and tool_results["codes_manquants"]:
                # Ne pas laisser le modèle inventer les UE absentes.
                answer = format_domain_response(tool_results)
            else:
                # Chaque extrait reste associé à ses propres métadonnées.
                blocks = []
                for doc in docs:
                    metadata = doc.metadata or {}
                    blocks.append(
                        "SOURCE : " + str(metadata.get("source") or "inconnue")
                        + "\nCODE_UE : " + str(metadata.get("code_ue") or "inconnu")
                        + "\nSEMESTRE : " + str(metadata.get("semestre") or "inconnu")
                        + "\nCONTENU :\n" + doc.page_content[:2400]
                    )
                context = "\n\n--- DOCUMENT SUIVANT ---\n\n".join(blocks)
                prompt = (
                    "Tu es Advisor, conseiller académique. Réponds en français.\n"
                    "Utilise EXCLUSIVEMENT les extraits de syllabus et les résultats "
                    "Python ci-dessous.\n"
                    "Règles :\n"
                    "1. N'invente ni UE, ni compétence, ni prérequis, ni ECTS.\n"
                    "2. N'attribue jamais les compétences d'une UE à une autre.\n"
                    "3. Écarte les documents hors sujet au lieu de les recommander.\n"
                    "4. N'annonce un total ECTS que si total_ects n'est pas null.\n"
                    "5. Si une information manque, signale cette limite.\n"
                    "6. Réponse concise : UE pertinentes, justification, sources.\n\n"
                    "EXEMPLE ONE-SHOT (illustration fictive, pas un vrai syllabus) :\n"
                    "Question : Quelle compétence apporte l'UE EX_101 ?\n"
                    "Contexte : EX_101 : apprendre SQL. EX_102 : gérer la sécurité réseau.\n"
                    "Bonne réponse : EX_101 développe SQL. Je n'attribue pas "
                    "la sécurité réseau à EX_101.\n\n"
                    f"DOCUMENTS :\n{context}\n\n"
                    f"RÉSULTATS PYTHON :\n{tool_results}\n"
                )
                try:
                    response = llm.invoke([SystemMessage(content=prompt), last_message])
                    content = response.content
                    answer = content.strip() if isinstance(content, str) else ""
                    if not answer:
                        answer = format_domain_response(tool_results)
                except Exception:
                    answer = format_domain_response(tool_results)

            sources = sorted(
                {
                    str(doc.metadata.get("source"))
                    for doc in docs
                    if doc.metadata and doc.metadata.get("source")
                }
            )
            if sources:
                answer += "\n\nSources consultées : " + ", ".join(sources)
            return {
                "messages": [AIMessage(content=f"[{self.name}] {answer}")],
                "current_agent": self.name,
            }

        return node
