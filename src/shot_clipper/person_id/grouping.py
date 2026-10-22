"""Group a session's goals by shooter, then suggest a name per group.

Within one session people keep their clothes, so appearance groups goals
well: on the owner's three tagged sessions, groups at GROUP_DISTANCE were
69-86% pure, at roughly two to three groups per real player. Across weeks
clothes change, so names are only *suggested*, and only when one name is
clearly closer than the next (NAME_MARGIN): about 40% of goals get a
suggestion, ~68% of them right on that test. A person confirms or fixes the
name once per group in the label UI's Players tab, and every confirmed name
joins the gallery that names next week's groups.

Any goal in a group that a person already tagged names the whole group -
the one-tap-per-player path within a session.
"""
from collections import Counter, defaultdict

import numpy as np

GROUP_DISTANCE = 0.12  # cosine distance, average linkage
NAME_MARGIN = 0.03     # best name's similarity minus the runner-up's
# What "some other person" typically scores: the most similar WRONG player
# among ~12 had median similarity 0.865 (owner's tags, 26 Sep vs 17 Oct).
# The runner-up is never taken as lower than this, so a gallery that knows
# only one or two people can't hand their names to everyone.
OTHER_PERSON_SIM = 0.86


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def group(embs: list, distance: float = GROUP_DISTANCE) -> list[int]:
    """Group index per embedding (0 = largest group)."""
    if not embs:
        return []
    if len(embs) == 1:
        return [0]
    from sklearn.cluster import AgglomerativeClustering

    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=distance, metric="cosine",
                                  linkage="average").fit_predict(np.stack([unit(e) for e in embs]))
    order = [g for g, _ in Counter(lab).most_common()]
    remap = {g: i for i, g in enumerate(order)}
    return [remap[g] for g in lab]


def prototypes(entries) -> dict[str, list[np.ndarray]]:
    """name -> one mean embedding per (session, name): a person's look in
    each week they were tagged, since clothes change between weeks."""
    by = defaultdict(list)
    for e in entries:
        by[(e["session"], e["name"])].append(unit(e["emb"]))
    out = defaultdict(list)
    for (_, name), vs in by.items():
        out[name].append(unit(np.mean(vs, axis=0)))
    return dict(out)


def name_groups(members: dict, gallery: list[dict], margin: float = NAME_MARGIN) -> dict:
    """members: group id -> [(emb, tag or None)]; gallery: [{session, name,
    emb}]. Returns group id -> {"name", "confidence", "how"}, where how is
    "tagged" (a member already carries a person's tag), "matched" (gallery
    look-alike, clear margin) or None (no suggestion)."""
    protos = prototypes(gallery)
    names = sorted(protos)
    out = {}
    for gid, items in members.items():
        tags = Counter(t for _, t in items if t)
        if tags:
            name, n = tags.most_common(1)[0]
            out[gid] = {"name": name, "confidence": round(n / len(items), 3), "how": "tagged"}
            continue
        if not names:
            out[gid] = {"name": None, "confidence": 0.0, "how": None}
            continue
        c = unit(np.mean([unit(e) for e, _ in items], axis=0))
        sims = np.array([max(float(c @ p) for p in protos[n]) for n in names])
        order = np.argsort(-sims)
        runner_up = max(float(sims[order[1]]) if len(names) > 1 else 0.0, OTHER_PERSON_SIM)
        gap = float(sims[order[0]]) - runner_up
        if gap >= margin:
            out[gid] = {"name": names[order[0]], "confidence": round(gap, 3), "how": "matched"}
        else:
            out[gid] = {"name": None, "confidence": round(gap, 3), "how": None}
    return out
