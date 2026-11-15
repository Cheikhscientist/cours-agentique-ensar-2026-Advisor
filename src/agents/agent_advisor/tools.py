"""Outils métiers déterministes de l'agent d'orientation Advisor."""

import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool


@tool
def calculer_total_ects(credits: list[int]) -> int:
    """Additionner les crédits ECTS de plusieurs UE distinctes.

    Args:
        credits: Liste de crédits ECTS entiers, sans doublons d'UE.

    Returns:
        Nombre total de crédits ECTS.

    Raises:
        ValueError: Si un crédit est négatif.
    """
    if any(credit < 0 for credit in credits):
        raise ValueError("Un crédit ECTS ne peut pas être négatif.")
    return sum(credits)


def _parse_ects(value: Any) -> int | None:
    """Lire un nombre d'ECTS non négatif sans inventer de valeur."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 and value.is_integer() else None
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*(?:ECTS)?\s*", value, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def process_domain_logic(
    query: str,
    retrieved_docs: list[Document],
    requested_codes: list[str] | None = None,
) -> dict[str, Any]:
    """Dédoublonner les UE et calculer leurs ECTS à partir des métadonnées.

    Les ECTS ne sont jamais estimés depuis le texte généré par le LLM.
    Un total incomplet n'est pas présenté comme le total demandé.
    """
    expected = list(dict.fromkeys(code.upper() for code in (requested_codes or [])))
    seen: set[str] = set()
    ues: list[dict[str, Any]] = []

    for doc in retrieved_docs:
        metadata = doc.metadata or {}
        code = str(metadata.get("code_ue") or "").strip().upper()
        source = str(metadata.get("source") or "").strip()
        identity = code or source
        if not identity or identity in seen:
            continue
        seen.add(identity)
        ues.append(
            {
                "code_ue": code,
                "source": source,
                "semestre": str(metadata.get("semestre") or ""),
                "filiere": str(metadata.get("filiere") or ""),
                "ects": _parse_ects(metadata.get("ects")),
            }
        )

    found_codes = {item["code_ue"] for item in ues}
    missing_codes = [code for code in expected if code not in found_codes]
    missing_ects = any(item["ects"] is None for item in ues)
    complete = bool(ues) and not missing_codes and not missing_ects
    total = (
        calculer_total_ects.invoke(
            {"credits": [item["ects"] for item in ues if item["ects"] is not None]}
        )
        if complete
        else None
    )

    if not ues:
        summary = "Aucune UE documentée n'a été retrouvée."
        status = "no_documents"
    elif not complete:
        summary = "Informations insuffisantes pour confirmer le total ECTS."
        status = "incomplete"
    else:
        summary = f"{len(ues)} UE distincte(s), {total} ECTS au total."
        status = "success"

    return {
        "status": status,
        "question": query,
        "documents_analyzed": len(retrieved_docs),
        "ue_retrouvees": ues,
        "nombre_ue": len(ues),
        "codes_demandes": expected,
        "codes_manquants": missing_codes,
        "total_ects": total,
        "ects_complets": complete,
        "summary": summary,
    }


def format_domain_response(tool_results: dict[str, Any]) -> str:
    """Réponse de secours lisible en cas d'échec ou d'absence du SLM."""
    summary = str(tool_results.get("summary", "Information indisponible."))
    missing = tool_results.get("codes_manquants", [])
    if missing:
        summary += " UE non retrouvées : " + ", ".join(missing) + "."
    return summary
