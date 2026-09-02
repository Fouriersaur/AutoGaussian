"""
VERDICTS AND THE TWO LIBRARIES (Sec. 4, Sec. 6 of the algorithm flow).

The search does *not* produce a single "invalid" bucket.  Failing to find a
witness and proving that no witness exists are different statements -- the
existential/universal asymmetry of Sec. 5 -- and the difference has to survive
into the data structure, because only one of the two may prune a subtree:

    VALID            a witness exists and is stored
    INVALID_PROV     the oracle did not find a witness      (provisional)
    INVALID_CERT     a certificate proved no witness exists (certified)
    INVALID_DEFAULT  escalation ran and resolved nothing    (provisional)

``INVALID_PROV`` and ``INVALID_DEFAULT`` both carry ``certified=False``; they
differ only in how much effort was spent.  ``n_uncertified`` counts them, and
is the completeness caveat reported with every run (Sec. 8).
"""

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "Verdict",
    "InvalidEntry",
    "Libraries",
    "graph_key",
    "key_to_graph",
    "REASON_FIT",
    "REASON_HURWITZ",
    "REASON_PROVISIONAL",
    "REASON_DEFAULT",
    "REASON_FIT_RANGE",
    "REASON_PBH",
    "REASON_PASSIVE",
    "REASON_BATH",
]

REASON_FIT = "fit"                  # provisional: the fit tolerance was not met
REASON_HURWITZ = "hurwitz"          # provisional: fit met, no stable witness found
REASON_PROVISIONAL = "provisional"  # not escalated at all (not on a pruning frontier)
REASON_DEFAULT = "default"          # escalated, nothing fired
REASON_FIT_RANGE = "fit_range"      # certified: target outside the graph's range
REASON_PBH = "pbh_dark_mode"        # certified: structurally non-stabilizable
REASON_PASSIVE = "passive_range"    # certified: passive graph cannot squeeze
REASON_BATH = "bath_unhosted"       # certified: no live channel carries the declared bath


class Verdict(Enum):
    """Oracle / escalation outcomes.  The oracle may only emit the first two."""

    VALID = "valid"
    INVALID_PROV = "invalid_provisional"
    INVALID_CERT = "invalid_certified"
    INVALID_DEFAULT = "invalid_default"

    @property
    def is_valid(self):
        return self is Verdict.VALID

    @property
    def certified(self):
        """``True`` only for a verdict backed by a real infeasibility proof."""
        return self is Verdict.INVALID_CERT


def graph_key(graph):
    """Hashable key of a graph vector (its raw int8 bytes)."""
    return bytes(np.asarray(graph, dtype="int8"))


def key_to_graph(key):
    return np.frombuffer(key, dtype="int8").copy()


@dataclass
class InvalidEntry:
    """One entry of the invalid library.

    Parameters
    ----------
    graph : int8 array
    certified : bool
        ``True`` **iff** an infeasibility certificate fired.  This is the flag
        the search consults before pruning: an uncertified entry never
        condemns its subgraphs (Sec. 6, invariant 3).
    reason : str
        One of the ``REASON_*`` constants.
    info : dict
        Diagnostics of the failed attempt (best loss reached, abscissa, ...).
    """

    graph: np.ndarray
    certified: bool = False
    reason: str = REASON_PROVISIONAL
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self):
        return graph_key(self.graph)

    def __str__(self):
        return "%s (%s)" % ("CERTIFIED" if self.certified else "uncertified", self.reason)


@dataclass
class Libraries:
    """The two libraries produced by :func:`autogaussian.search.discover`.

    ``valid`` maps a graph key to the witness run (the info dict of the
    successful optimisation, containing ``x``, ``loss_reached`` and
    ``max_real_eigenvalue``).  ``invalid`` maps a graph key to an
    :class:`InvalidEntry`.
    """

    valid: Dict[bytes, Dict[str, Any]] = field(default_factory=dict)
    invalid: Dict[bytes, InvalidEntry] = field(default_factory=dict)
    num_oracle_calls: int = 0
    num_escalations: int = 0
    # memo for certificates that do not depend on the graph (the physicality
    # SDP is a statement about the target alone -- solve it once per run)
    cache: Dict[str, Any] = field(default_factory=dict)

    # -- bookkeeping -------------------------------------------------------

    def add_valid(self, graph, witness):
        key = graph_key(graph)
        previous = self.invalid.pop(key, None)
        if previous is not None and previous.certified:
            # a certificate said no witness exists and one turned up anyway:
            # the certificate is wrong, and silence here would hide it
            warnings.warn("graph %r was certified invalid (%s) but a witness was "
                          "found; the certificate is unsound"
                          % (np.asarray(graph).tolist(), previous.reason),
                          RuntimeWarning)
        self.valid[key] = witness

    def add_invalid(self, entry):
        key = entry.key
        if key in self.valid:
            raise ValueError("graph is already in the valid library")
        self.invalid[key] = entry

    def __contains__(self, graph):
        key = graph_key(graph)
        return key in self.valid or key in self.invalid

    def verdict_of(self, graph):
        key = graph_key(graph)
        if key in self.valid:
            return Verdict.VALID
        entry = self.invalid.get(key)
        if entry is None:
            return None
        if entry.certified:
            return Verdict.INVALID_CERT
        if entry.reason == REASON_DEFAULT:
            return Verdict.INVALID_DEFAULT
        return Verdict.INVALID_PROV

    # -- views used by the search -----------------------------------------

    def valid_graphs(self):
        return np.array([key_to_graph(k) for k in self.valid], dtype="int8")

    def certified_invalid_graphs(self):
        """The **only** graphs allowed to condemn their subgraphs (Sec. 6)."""
        rows = [key_to_graph(k) for k, e in self.invalid.items() if e.certified]
        return np.array(rows, dtype="int8")

    def uncertified_entries(self) -> List[InvalidEntry]:
        return [e for e in self.invalid.values() if not e.certified]

    def n_uncertified(self) -> int:
        """Number of invalid entries that were *not* proved infeasible."""
        return sum(1 for e in self.invalid.values() if not e.certified)

    def n_certified(self) -> int:
        return sum(1 for e in self.invalid.values() if e.certified)

    def minimal_valid(self):
        """Minimal elements of the valid library -- the irreducible graphs."""
        from autogaussian.graph import any_subgraph_of

        graphs = self.valid_graphs()
        if graphs.size == 0:
            return np.zeros((0, 0), dtype="int8")
        graphs = np.unique(graphs, axis=0)
        keep = []
        for idx, graph in enumerate(graphs):
            others = np.delete(graphs, idx, axis=0)
            if not any_subgraph_of(graph, others):
                keep.append(graph)
        return np.array(keep, dtype="int8")

    # -- reporting (Sec. 8) ------------------------------------------------

    def completeness_statement(self, space=None) -> str:
        """The statement Sec. 8 requires every run to emit."""
        n = self.n_uncertified()
        if n == 0:
            return ("The list of irreducible graphs is complete and certified: "
                    "every rejected graph carries an infeasibility certificate.")
        lines = ["The list of irreducible graphs is complete up to at most %i "
                 "false negative%s (graphs rejected without a certificate):"
                 % (n, "" if n == 1 else "s")]
        for entry in self.uncertified_entries():
            if space is not None:
                elements = ", ".join(space.describe(entry.graph)) or "empty graph"
            else:
                elements = np.array2string(np.asarray(entry.graph))
            lines.append("    [%s] %s" % (entry.reason, elements))
        return "\n".join(lines)

    def summary(self) -> str:
        return ("valid: %i    invalid: %i (certified %i, uncertified %i)    "
                "oracle calls: %i    escalations: %i"
                % (len(self.valid), len(self.invalid), self.n_certified(),
                   self.n_uncertified(), self.num_oracle_calls, self.num_escalations))
