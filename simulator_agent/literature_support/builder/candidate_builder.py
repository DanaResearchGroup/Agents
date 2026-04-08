"""Build experiment candidates from evidence snippets.

Groups evidence by experiment family (apparatus), attaches observables,
and determines which candidates are simulatable.
"""

from __future__ import annotations

from collections import defaultdict

from models import EvidenceSnippet, ExperimentCandidate


# Sections where a family anchor is strong enough to create a candidate
_ANCHOR_SECTIONS = {"methods", "results", "caption", "unknown"}

# Page proximity for associating conditions with an anchor
_PAGE_RADIUS = 1

# Minimum conditions needed alongside a family anchor to be simulatable
_MIN_CONDITIONS_FOR_SIMULATABLE = 2


def build_experiment_candidates(
    evidence: list[EvidenceSnippet],
) -> list[ExperimentCandidate]:
    """Build experiment candidates from evidence snippets.

    Algorithm:
    1. Find experiment_family anchors (apparatus mentions)
    2. Merge anchors with same family
    3. Collect nearby conditions + observables for each group
    4. Filter: intro-only anchors don't become candidates
    5. Sort: primary (methods) first, then by confidence
    """
    families = [e for e in evidence if e.kind == "experiment_family"]
    observables = [e for e in evidence if e.kind == "observable_type"]
    conditions = [e for e in evidence if e.kind not in ("experiment_family", "observable_type")]

    if not families:
        return _no_anchor_fallback(conditions, observables)

    family_groups = _merge_anchors(families)

    candidates: list[ExperimentCandidate] = []
    candidate_idx = 0

    for family_name, group_anchors in family_groups.items():
        strong_anchors = [a for a in group_anchors if a.section in _ANCHOR_SECTIONS]
        if not strong_anchors:
            continue

        page_min = min(a.page_num for a in group_anchors)
        page_max = max(a.page_num for a in group_anchors)

        nearby_conds = _collect_nearby(conditions, page_min, page_max, group_anchors)
        nearby_obs = _collect_nearby(observables, page_min, page_max, group_anchors)

        # Split observables: confirmed (methods/results, high weight) vs possible
        confirmed, possible = _classify_observables(nearby_obs)

        best_anchor = max(group_anchors, key=lambda a: a.section_weight)
        is_primary = best_anchor.section == "methods"

        all_evidence = list(group_anchors) + nearby_obs + nearby_conds
        avg_conf = sum(e.confidence for e in all_evidence) / len(all_evidence)

        cond_kinds = set(c.kind for c in nearby_conds)
        has_temp = "temperature" in cond_kinds
        has_pressure = "pressure" in cond_kinds
        simulatable = is_primary and len(cond_kinds) >= _MIN_CONDITIONS_FOR_SIMULATABLE and has_temp

        # Planning priority
        if is_primary and simulatable:
            priority = "high"
        elif is_primary or simulatable:
            priority = "medium"
        else:
            priority = "low"

        notes: list[str] = []
        sections_involved = {a.section for a in group_anchors if a.section}
        if len(sections_involved) > 1:
            notes.append(f"found in sections: {', '.join(sorted(sections_involved))}")
        if is_primary:
            notes.append("primary experiment (from methods/experimental)")
        if not simulatable and is_primary:
            missing = []
            if not has_temp:
                missing.append("temperature")
            if not has_pressure:
                missing.append("pressure")
            if missing:
                notes.append(f"missing for simulation: {', '.join(missing)}")

        candidate_idx += 1
        candidates.append(ExperimentCandidate(
            candidate_id=f"candidate_{candidate_idx}",
            experiment_family=family_name,
            confirmed_observables=confirmed,
            possible_observables=possible,
            section=best_anchor.section,
            page_start=page_min,
            page_end=page_max,
            evidence=all_evidence,
            confidence=round(avg_conf, 3),
            is_primary=is_primary,
            simulatable=simulatable,
            planning_priority=priority,
            notes=notes,
        ))

    candidates.sort(key=lambda c: (
        not c.is_primary,
        not c.simulatable,
        -len(c.evidence),  # more evidence = more dominant in the paper
        -c.confidence,
    ))
    return candidates


def _merge_anchors(
    anchors: list[EvidenceSnippet],
) -> dict[str, list[EvidenceSnippet]]:
    groups: dict[str, list[EvidenceSnippet]] = defaultdict(list)
    for anchor in anchors:
        groups[anchor.value_text].append(anchor)
    return dict(groups)


# Sections that count as "strong" for observable confirmation
_STRONG_OBS_SECTIONS = {"methods", "results"}
# Minimum section weight for an observable to be "confirmed"
_CONFIRMED_OBS_WEIGHT = 0.8


def _classify_observables(
    obs_snippets: list[EvidenceSnippet],
) -> tuple[list[str], list[str]]:
    """Split observables into confirmed vs possible based on evidence strength.

    Confirmed: found in methods/results with high section weight.
    Possible: mentioned nearby but in weaker sections (captions, intro).
    """
    confirmed: set[str] = set()
    possible: set[str] = set()

    for obs in obs_snippets:
        if obs.section in _STRONG_OBS_SECTIONS and obs.section_weight >= _CONFIRMED_OBS_WEIGHT:
            confirmed.add(obs.value_text)
        else:
            possible.add(obs.value_text)

    # If an observable is confirmed, remove it from possible
    possible -= confirmed
    return sorted(confirmed), sorted(possible)


def _collect_nearby(
    snippets: list[EvidenceSnippet],
    page_min: int,
    page_max: int,
    anchors: list[EvidenceSnippet],
) -> list[EvidenceSnippet]:
    anchor_sections = {a.section for a in anchors if a.section}
    expanded_min = page_min - _PAGE_RADIUS
    expanded_max = page_max + _PAGE_RADIUS

    nearby = []
    for s in snippets:
        if not (expanded_min <= s.page_num <= expanded_max):
            continue
        if s.section in anchor_sections or abs(s.page_num - page_min) <= _PAGE_RADIUS:
            nearby.append(s)
    return nearby


def _no_anchor_fallback(
    conditions: list[EvidenceSnippet],
    observables: list[EvidenceSnippet],
) -> list[ExperimentCandidate]:
    all_ev = conditions + observables
    if not all_ev:
        return []

    page_min = min(e.page_num for e in all_ev)
    page_max = max(e.page_num for e in all_ev)
    avg_conf = sum(e.confidence for e in all_ev) / len(all_ev)
    confirmed, possible = _classify_observables(observables)

    return [ExperimentCandidate(
        candidate_id="candidate_1",
        experiment_family="unknown",
        confirmed_observables=confirmed,
        possible_observables=possible,
        section=None,
        page_start=page_min,
        page_end=page_max,
        evidence=all_ev,
        confidence=round(avg_conf, 3),
        is_primary=False,
        simulatable=False,
        planning_priority="low",
        notes=["no experiment family detected — conditions only"],
    )]
