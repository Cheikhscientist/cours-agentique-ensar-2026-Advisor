"""Advisor : RAG ciblé par intitulé ou code UE et calcul ECTS déterministe."""

import os
import re
import unicodedata
from collections.abc import Callable
from pathlib import PurePath
from typing import Any, cast

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.core.base_agent import BaseAgentPlugin
from src.core.ingestion import get_vector_store
from src.core.state import AgentState

from .tools import format_domain_response, process_domain_logic

# Ces mots décrivent la question, mais ne caractérisent pas le domaine d'une UE.
_STOP_WORDS = {
    "academique", "acquerir", "avec", "avoir", "competence", "competences",
    "conseille", "conseiller", "cours", "dans", "developper", "dois", "domaine",
    "donne", "donnes", "elles", "etudiant", "faire", "filiere", "formation",
    "pour", "quelle", "quelles", "quels", "recommande", "recommandes",
    "recommandation", "souhaite", "travailler", "unite", "utiles", "veux",
    "metier", "parcours", "suivre", "etre", "sont", "viser", "visees", "acquises",
}


def _normalized_words(value: str) -> list[str]:
    """Normaliser accents, ponctuation et casse pour comparer les intitulés."""
    plain = unicodedata.normalize("NFKD", value)
    plain = "".join(char for char in plain if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", plain.lower())


def _extract_codes(question: str) -> list[str]:
    """Extraire les codes UE mentionnés explicitement, sans doublons."""
    return list(
        dict.fromkeys(
            code.upper()
            for code in re.findall(r"\b[A-Z]{2,5}_\d{3}\b", question, re.IGNORECASE)
        )
    )


def _extract_filters(question: str) -> dict[str, str]:
    """Utiliser exclusivement les informations de filière et semestre explicites."""
    result: dict[str, str] = {}
    semester = re.search(r"\bS(?:[1-9]|1[0-2])\b", question, re.IGNORECASE)
    if semester:
        result["semestre"] = semester.group(0).upper()
    track = re.search(r"\bfili[eè]re\s*:\s*([A-Za-z0-9_-]+)", question, re.IGNORECASE)
    if track:
        result["filiere"] = track.group(1)
    return result


def _extract_semesters(question: str) -> list[str]:
    """Extraire S9/S10, y compris « au Semestre 9 et 10 »."""
    semesters = [
        f"S{int(match)}"
        for match in re.findall(
            r"\b(?:semestres?|s)\s*(1[0-2]|[1-9])\b", question, re.IGNORECASE
        )
    ]
    pair = re.search(
        r"\bsemestres?\s+(1[0-2]|[1-9])\s*(?:et|ou|,|/|-)\s*"
        r"(?:s(?:emestre)?\s*)?(1[0-2]|[1-9])\b",
        question,
        re.IGNORECASE,
    )
    if pair:
        semesters.extend((f"S{int(pair.group(1))}", f"S{int(pair.group(2))}"))
    return list(dict.fromkeys(semesters))


def _verified_semester_docs(docs: list[Document], semester: str) -> list[Document]:
    """Ne conserver que les documents effectivement indexés au semestre demandé."""
    return [
        doc
        for doc in docs
        if str((doc.metadata or {}).get("semestre", "")).upper() == semester
    ]


def _format_semester_recommendations(
    semesters: list[str], docs: list[Document]
) -> str:
    """Réponse vérifiable : un semestre n'hérite jamais des UE de l'autre."""
    lines = ["UE documentées par semestre pour votre orientation :"]
    for semester in semesters:
        lines.append(f"{semester} :")
        seen: set[str] = set()
        found = False
        for doc in docs:
            meta = doc.metadata or {}
            code = str(meta.get("code_ue") or "").upper()
            if (
                str(meta.get("semestre") or "").upper() != semester
                or not code
                or code in seen
            ):
                continue
            seen.add(code)
            found = True
            source = str(meta.get("source") or "")
            title = " ".join(_title_terms(source)) if source else ""
            track = str(meta.get("filiere") or "non renseignée")
            ects = meta.get("ects")
            details = f"- {code}"
            if title:
                details += f" — {title}"
            details += f" ; filière : {track}"
            if ects is not None:
                details += f" ; ECTS indiqués : {ects}"
            lines.append(details + ".")
        if not found:
            lines.append("- Aucune UE pertinente confirmée par les syllabus consultés.")
    lines.append("Seules les UE dont le semestre est vérifié dans les métadonnées sont proposées.")
    return "\n".join(lines)


def _chroma_filter(values: dict[str, str]) -> dict[str, Any] | None:
    """Fabriquer une clause where valide pour ChromaDB."""
    if not values:
        return None
    if len(values) == 1:
        key, value = next(iter(values.items()))
        return {key: value}
    return {"$and": [{key: value} for key, value in values.items()]}


def _title_terms(source: str) -> list[str]:
    """Extraire les termes utiles d'un nom de syllabus, sans son code UE."""
    name = PurePath(source.replace("\\", "/")).name
    stem = name.removesuffix(".md")
    stem = re.sub(r"^[A-Za-z]{2,5}_\d{3}[_-]?", "", stem)
    return [term for term in _normalized_words(stem) if term not in _STOP_WORDS]


def _title_score(question: str, source: str) -> int:
    """Classer les intitulés sur la présence des mots métier de la question."""
    wanted = {
        word for word in _normalized_words(question)
        if len(word) >= 3 and word not in _STOP_WORDS
    }
    title = set(_title_terms(source))
    matched = wanted & title
    # Deux mots métier ou un terme spécialisé long : critère conservateur.
    if len(matched) >= 2:
        return len(matched)
    if len(matched) == 1 and len(next(iter(matched))) >= 5:
        return 1
    return 0


def _catalog_codes(vector_store: Any, question: str, filters: dict[str, str]) -> list[str]:
    """Trouver les intitulés de syllabus correspondants depuis les métadonnées Chroma.

    L'index des seules métadonnées est consulté sans charger le texte des chunks.
    Cette optimisation facultative utilise l'accès Chroma interne ; si celui-ci
    n'est pas accessible, l'agent se rabat sur la recherche sémantique.
    """
    collection = getattr(vector_store, "_collection", None)
    if collection is None:
        return []
    try:
        catalog = collection.get(include=["metadatas"])
    except Exception:
        return []
    if not isinstance(catalog, dict):
        return []
    raw = catalog.get("metadatas")
    if not isinstance(raw, list):
        return []

    candidates: dict[str, int] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        if any(str(item.get(key, "")) != value for key, value in filters.items()):
            continue
        code = str(item.get("code_ue") or "").strip().upper()
        source = str(item.get("source") or "")
        if not code or not source:
            continue
        score = _title_score(question, source)
        if score:
            candidates[code] = max(candidates.get(code, 0), score)
    return sorted(candidates, key=lambda code: (-candidates[code], code))[:2]


def _retrieve_code(
    vector_store: Any, question: str, code: str, filters: dict[str, str]
) -> list[Document]:
    """Chercher au plus deux chunks de l'UE demandée."""
    metadata_filter = _chroma_filter({**filters, "code_ue": code})
    kwargs: dict[str, Any] = {"k": 2}
    if metadata_filter is not None:
        kwargs["filter"] = metadata_filter
    retriever = vector_store.as_retriever(search_kwargs=kwargs)
    return cast(list[Document], retriever.invoke(question))


def _retrieve_semantic(
    vector_store: Any, question: str, filters: dict[str, str]
) -> list[Document]:
    """Recherche sémantique standard, limitée à deux chunks."""
    kwargs: dict[str, Any] = {"k": 2}
    metadata_filter = _chroma_filter(filters)
    if metadata_filter is not None:
        kwargs["filter"] = metadata_filter
    return cast(list[Document], vector_store.as_retriever(search_kwargs=kwargs).invoke(question))



def _is_factual_ects_question(question: str, codes: list[str]) -> bool:
    """Limiter la réponse directe aux questions factuelles sur des UE nommées."""
    if not codes:
        return False
    words = set(_normalized_words(question))
    if "ects" not in words:
        return False
    # Ne pas court-circuiter les conseils de parcours ou les comparaisons.
    if words & {
        "recommande", "recommandes", "recommander", "conseille", "conseiller",
        "comparaison", "compare", "comparer", "hesite", "competences",
        "competence", "metier", "filiere", "filieres", "parcours",
        "prerequis", "contenu", "pourquoi", "choix", "options",
    }:
        return False
    return bool(words & {"combien", "calcule", "calculer", "calcul", "total", "solde", "credits", "rapporte"})


def _format_factual_ects(question: str, results: dict[str, Any]) -> str:
    """Construire la réponse depuis les métadonnées validées, sans LLM."""
    if results.get("total_ects") is None:
        return format_domain_response(results)

    ues = results.get("ue_retrouvees", [])
    lines = ["Crédits ECTS vérifiés dans les syllabus :"]
    for ue in ues:
        code = str(ue["code_ue"])
        ects = int(ue["ects"])
        semester = str(ue.get("semestre") or "")
        detail = f"- {code} : {ects} ECTS"
        if semester:
            detail += f", semestre {semester}"
        lines.append(detail + ".")

    total = int(results["total_ects"])
    lines.append(f"Total : {total} ECTS.")
    normalized = unicodedata.normalize("NFKD", question)
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    target = re.search(
        r"\b(?:atteindre|objectif(?:\s+de)?|sur)\s+(\d+)\s*ECTS\b",
        normalized,
        re.IGNORECASE,
    )
    if target:
        objective = int(target.group(1))
        remaining = objective - total
        if remaining >= 0:
            lines.append(f"Reste {remaining} ECTS pour atteindre {objective} ECTS.")
        else:
            lines.append(f"Le total dépasse {objective} ECTS de {-remaining} ECTS.")
    return "\n".join(lines)


class CustomAgentPlugin(BaseAgentPlugin):
    """Conseiller d'orientation académique appuyé sur les syllabus indexés."""

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
        """Retourner le nœud LangGraph, avec initialisation paresseuse de ChromaDB."""
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
            question = str(last_message.content).strip()
            if not question:
                return {
                    "messages": [AIMessage(content="[advisor] Question vide.")],
                    "current_agent": self.name,
                }

            explicit_codes = _extract_codes(question)
            filters = _extract_filters(question)
            semesters = _extract_semesters(question)
            multi_semester_orientation = (
                len(semesters) >= 2
                and not explicit_codes
                and bool(
                    set(_normalized_words(question))
                    & {
                        "specialiser", "recommande", "recommandes", "selectionner",
                        "choisir", "filieres", "filiere", "parcours", "orientation",
                    }
                )
            )
            docs: list[Document] = []
            retrieval_error = False
            try:
                if vector_store is None:
                    vector_store = get_vector_store(reindex_if_empty=False)

                if multi_semester_orientation:
                    # Une recherche indépendante par semestre, sans transfert d'UE.
                    for semester in semesters:
                        semester_filters = {**filters, "semestre": semester}
                        codes = _catalog_codes(vector_store, question, semester_filters)
                        semester_docs: list[Document] = []
                        if codes:
                            for code in codes:
                                semester_docs.extend(
                                    _retrieve_code(
                                        vector_store, question, code, semester_filters
                                    )
                                )
                        else:
                            semester_docs = _retrieve_semantic(
                                vector_store, question, semester_filters
                            )
                        docs.extend(_verified_semester_docs(semester_docs, semester))
                elif explicit_codes:
                    # Tous les codes explicitement demandés sont recherchés.
                    for code in explicit_codes:
                        docs.extend(_retrieve_code(vector_store, question, code, filters))
                else:
                    # Intitulé correspondant dans le catalogue : priorité au titre.
                    named_codes = _catalog_codes(vector_store, question, filters)
                    if named_codes:
                        for code in named_codes:
                            docs.extend(_retrieve_code(vector_store, question, code, filters))
                    else:
                        docs = _retrieve_semantic(vector_store, question, filters)
            except Exception:
                retrieval_error = True
                docs = []

            # Ne pas envoyer des UE de risque cyber pour une question BI quand
            # une correspondance par intitulé documentaire a été trouvée.
            results = process_domain_logic(question, docs, requested_codes=explicit_codes)
            if retrieval_error:
                answer = "Recherche documentaire indisponible. Réessaie plus tard."
            elif multi_semester_orientation:
                answer = _format_semester_recommendations(semesters, docs)
            elif not docs or (explicit_codes and results["codes_manquants"]):
                answer = format_domain_response(results)
            elif _is_factual_ects_question(question, explicit_codes):
                answer = _format_factual_ects(question, results)
            else:
                blocks: list[str] = []
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
                    "Utilise EXCLUSIVEMENT les extraits et résultats Python fournis.\n"
                    "RÈGLES :\n"
                    "1. N'invente aucune UE, compétence, condition ou crédit ECTS.\n"
                    "2. Chaque compétence doit appartenir au code UE de son document.\n"
                    "3. Ne recommande PAS une UE simplement parce qu'elle est présente "
                    "dans les extraits : vérifie le lien direct avec le métier demandé.\n"
                    "4. Si une UE est hors sujet, écarte-la de la recommandation.\n"
                    "5. Ne donne de total ECTS que si total_ects est disponible.\n"
                    "6. Si une information est absente, indique-le clairement.\n"
                    "7. Donne les UE pertinentes avec leurs compétences exactes et sources.\n"
                    "8. Ne propose jamais une UE absente des documents fournis.\n"
                    "9. Évite les répétitions et termine les comparaisons par une conclusion courte.\n\n"
                    "EXEMPLE ONE-SHOT FICTIF (pas un vrai syllabus) :\n"
                    "Question : Je veux travailler en analyse de données.\n"
                    "Documents : EX_101 = analyse SQL ; EX_102 = gestion d'incidents cyber.\n"
                    "Réponse correcte : EX_101 est pertinent grâce à SQL. "
                    "Je ne recommande pas EX_102 pour ce métier.\n\n"
                    f"DOCUMENTS :\n{context}\n\n"
                    f"RÉSULTATS PYTHON :\n{results}\n"
                )
                try:
                    response = llm.invoke([SystemMessage(content=prompt), last_message])
                    content = response.content
                    answer = content.strip() if isinstance(content, str) else ""
                    if not answer:
                        answer = format_domain_response(results)
                except Exception:
                    answer = format_domain_response(results)

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
                "messages": [AIMessage(content=f"[Agent {self.name}] {answer}")],
                "current_agent": self.name,
            }

        return node
