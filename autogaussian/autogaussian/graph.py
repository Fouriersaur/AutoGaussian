"""
Discrete graph lattice (Sec. 2 / Sec. 6 of the algorithm flow).

A candidate device is an integer vector over *slots*.  Because the covariance
oracle works in the full 2N x 2N Nambu structure, a mode pair carries **two**
independent edges (unlike AUTOSCATTER's reduced N x N picture, where the
particle/hole assignment of the modes decides whether an edge is a
beam-splitter or a squeezer):

    block 0 ("normal", from ``g``)     : detunings on the diagonal,
                                         beam-splitter couplings off-diagonal
    block 1 ("anomalous", from ``nu``) : on-site (degenerate) squeezing on the
                                         diagonal, two-mode squeezing off-diagonal

Slot values -- identical semantics to AUTOSCATTER, so complexity is additive
and the validity monotonicity of Sec. 6 is preserved:

    0 = NO_COUPLING          1 = DETUNING / COUPLING_WITHOUT_PHASE (real)
    2 = COUPLING_WITH_PHASE (complex)

The graph vector is the concatenation of the upper triangle of block 0 and the
upper triangle of block 1, length ``N(N+1)``.

Monotonicity used by the search:  ``g1 <= g2`` elementwise  <=>  ``g1`` is a
subgraph of ``g2``.
"""

from itertools import product

import numpy as np

__all__ = [
    "NO_COUPLING",
    "DETUNING",
    "COUPLING_WITHOUT_PHASE",
    "COUPLING_WITH_PHASE",
    "GraphSpace",
    "is_subgraph_of_any",
    "any_subgraph_of",
    "characterize_architecture",
    "characterize_architectures",
    "min_number_of_pumps",
    "plot_graph",
    "plot_list_of_graphs",
]

NO_COUPLING = 0
DETUNING = 1
COUPLING_WITHOUT_PHASE = 1
COUPLING_WITH_PHASE = 2

SLOT_DETUNING = "detuning"
SLOT_BEAMSPLITTER = "beamsplitter"
SLOT_ONSITE_SQUEEZING = "onsite_squeezing"
SLOT_TWO_MODE_SQUEEZING = "two_mode_squeezing"


class GraphSpace:
    """Enumerates the slots of an ``N``-mode Nambu device and the values each
    slot may take, honouring the structural constraints of Sec. 1.4.

    Parameters
    ----------
    num_modes : int
    allow_detunings : bool
    allow_beamsplitters : bool
    allow_two_mode_squeezing : bool
    allow_onsite_squeezing : bool
    forbidden_couplings : list of (i, j)
        Pairs that may not couple *at all* (e.g. "two signal modes may not
        couple directly -- must be mediated").
    passive_only_couplings : list of (i, j)
        Pairs where no active/parametric (squeezing) link is allowed.
    real_only_couplings : list of (i, j)
        Pairs whose couplings must be real (no synthetic flux on that edge).
    forbidden_detunings, forbidden_onsite_squeezing : list of int
    """

    def __init__(
        self,
        num_modes,
        allow_detunings=True,
        allow_beamsplitters=True,
        allow_two_mode_squeezing=True,
        allow_onsite_squeezing=True,
        forbidden_couplings=(),
        passive_only_couplings=(),
        real_only_couplings=(),
        forbidden_detunings=(),
        forbidden_onsite_squeezing=(),
    ):
        self.num_modes = int(num_modes)
        self.forbidden_couplings = set(frozenset(p) for p in forbidden_couplings)
        self.passive_only_couplings = set(frozenset(p) for p in passive_only_couplings)
        self.real_only_couplings = set(frozenset(p) for p in real_only_couplings)
        self.forbidden_detunings = set(forbidden_detunings)
        self.forbidden_onsite_squeezing = set(forbidden_onsite_squeezing)

        self.slots = []            # list of (kind, i, j)
        self.possible_values = []  # list of list[int]

        triu = list(zip(*np.triu_indices(self.num_modes)))

        # block 0: detunings + beam-splitter couplings
        for i, j in triu:
            i, j = int(i), int(j)
            if i == j:
                allowed = [NO_COUPLING]
                if allow_detunings and i not in self.forbidden_detunings:
                    allowed.append(DETUNING)
                self.slots.append((SLOT_DETUNING, i, j))
            else:
                allowed = [NO_COUPLING]
                if allow_beamsplitters and frozenset((i, j)) not in self.forbidden_couplings:
                    allowed.append(COUPLING_WITHOUT_PHASE)
                    if frozenset((i, j)) not in self.real_only_couplings:
                        allowed.append(COUPLING_WITH_PHASE)
                self.slots.append((SLOT_BEAMSPLITTER, i, j))
            self.possible_values.append(allowed)

        # block 1: on-site + two-mode squeezing
        for i, j in triu:
            i, j = int(i), int(j)
            if i == j:
                allowed = [NO_COUPLING]
                if allow_onsite_squeezing and i not in self.forbidden_onsite_squeezing:
                    allowed.append(COUPLING_WITHOUT_PHASE)
                    allowed.append(COUPLING_WITH_PHASE)
                self.slots.append((SLOT_ONSITE_SQUEEZING, i, j))
            else:
                allowed = [NO_COUPLING]
                pair = frozenset((i, j))
                active_ok = (
                    allow_two_mode_squeezing
                    and pair not in self.forbidden_couplings
                    and pair not in self.passive_only_couplings
                )
                if active_ok:
                    allowed.append(COUPLING_WITHOUT_PHASE)
                    if pair not in self.real_only_couplings:
                        allowed.append(COUPLING_WITH_PHASE)
                self.slots.append((SLOT_TWO_MODE_SQUEEZING, i, j))
            self.possible_values.append(allowed)

        self.num_slots = len(self.slots)
        self.max_values = np.array([max(v) for v in self.possible_values], dtype="int8")
        self.max_complexity = int(self.max_values.sum())

    # -- basic graphs ------------------------------------------------------

    def fully_connected(self):
        """The maximal (most complex) graph -- the root of the search."""
        return self.max_values.copy()

    def empty(self):
        return np.zeros(self.num_slots, dtype="int8")

    def num_possible_graphs(self):
        n = 1
        for v in self.possible_values:
            n *= len(v)
        return n

    # -- enumeration -------------------------------------------------------

    def iter_graphs_at_complexity(self, level):
        """Yield every admissible graph whose complexity (sum of slot values)
        equals ``level``.

        Recursive enumeration with suffix-sum pruning -- avoids materialising
        the full ``prod_i |values_i|`` product, which is what makes N >= 4
        tractable at low complexity.
        """
        values = self.possible_values
        n = self.num_slots
        suffix_max = np.zeros(n + 1, dtype=int)
        for idx in range(n - 1, -1, -1):
            suffix_max[idx] = suffix_max[idx + 1] + max(values[idx])

        current = np.zeros(n, dtype="int8")

        def rec(idx, remaining):
            if idx == n:
                if remaining == 0:
                    yield current.copy()
                return
            for v in values[idx]:
                if v <= remaining and remaining - v <= suffix_max[idx + 1]:
                    current[idx] = v
                    for out in rec(idx + 1, remaining - v):
                        yield out
            current[idx] = 0

        for out in rec(0, int(level)):
            yield out

    def all_graphs(self):
        """Materialise every admissible graph (only for small ``N``)."""
        return np.array(list(product(*self.possible_values)), dtype="int8")

    # -- readout helpers ---------------------------------------------------

    def describe(self, graph):
        """Human-readable list of the active elements of ``graph``."""
        out = []
        for value, (kind, i, j) in zip(graph, self.slots):
            if value == NO_COUPLING:
                continue
            phase = "complex" if value == COUPLING_WITH_PHASE else "real"
            if kind == SLOT_DETUNING:
                out.append("detuning Delta_%i" % i)
            elif kind == SLOT_BEAMSPLITTER:
                out.append("beam-splitter %i-%i (%s)" % (i, j, phase))
            elif kind == SLOT_ONSITE_SQUEEZING:
                out.append("on-site squeezing %i (%s)" % (i, phase))
            else:
                out.append("two-mode squeezing %i-%i (%s)" % (i, j, phase))
        return out

    def to_blocks(self, graph):
        """Split a graph vector into the two ``N x N`` upper-triangular blocks
        (normal, anomalous) as dense integer matrices."""
        n = self.num_modes
        triu = np.triu_indices(n)
        half = len(triu[0])
        normal = np.zeros((n, n), dtype="int8")
        anomalous = np.zeros((n, n), dtype="int8")
        normal[triu] = graph[:half]
        anomalous[triu] = graph[half:]
        return normal, anomalous


# ---------------------------------------------------------------------------
# monotonicity helpers (Sec. 6)
# ---------------------------------------------------------------------------

def is_subgraph_of_any(graph, supergraphs):
    """True if ``graph`` is a subgraph of at least one row of ``supergraphs``.

    Used to discard a candidate that is contained in a known INVALID graph
    (every subgraph of an invalid graph is invalid).
    """
    supergraphs = np.asarray(supergraphs)
    if supergraphs.size == 0 or graph is None:
        return False
    supergraphs = np.atleast_2d(supergraphs)
    return bool(np.any(np.all(supergraphs - np.asarray(graph) >= 0, axis=-1)))


def any_subgraph_of(graph, subgraphs):
    """True if at least one row of ``subgraphs`` is a subgraph of ``graph``.

    Used to discard a candidate that already contains a known VALID graph
    (every extension of a valid graph is valid, hence not irreducible).
    """
    subgraphs = np.asarray(subgraphs)
    if subgraphs.size == 0 or graph is None:
        return False
    subgraphs = np.atleast_2d(subgraphs)
    return bool(np.any(np.all(np.asarray(graph) - subgraphs >= 0, axis=-1)))


# ---------------------------------------------------------------------------
# Sec. 7.1 -- complexity metadata
# ---------------------------------------------------------------------------

def characterize_architecture(graph, space):
    """Count the ingredients of one graph."""
    num_detunings = 0
    num_passive = 0
    num_passive_complex = 0
    num_onsite_squeezing = 0
    num_two_mode_squeezing = 0
    for value, (kind, i, j) in zip(graph, space.slots):
        if value == NO_COUPLING:
            continue
        if kind == SLOT_DETUNING:
            num_detunings += 1
        elif kind == SLOT_BEAMSPLITTER:
            num_passive += 1
            if value == COUPLING_WITH_PHASE:
                num_passive_complex += 1
        elif kind == SLOT_ONSITE_SQUEEZING:
            num_onsite_squeezing += 1
        else:
            num_two_mode_squeezing += 1
    return {
        "num_detunings": num_detunings,
        "num_beamsplitters": num_passive,
        "num_complex_beamsplitters": num_passive_complex,
        "num_onsite_squeezing": num_onsite_squeezing,
        "num_two_mode_squeezing": num_two_mode_squeezing,
        "num_couplings": num_passive + num_two_mode_squeezing + num_onsite_squeezing,
        "complexity": int(np.sum(graph)),
    }


def min_number_of_pumps(graph, space):
    """Minimum number of parametric pumps needed to realise ``graph``.

    Modes are labelled by drive frequency class; two modes sharing a label are
    operated at the same frequency.  Rules:

    * real beam-splitter between equal labels  -> no pump
    * beam-splitter between different labels   -> one pump
    * *complex* beam-splitter                  -> labels must differ (a
      synthetic phase is imprinted by a pump)  -> one pump
    * two-mode squeezing                       -> always one pump (at the sum
      frequency)
    * on-site (degenerate) squeezing           -> always one pump (at 2 omega)
    * detuning                                 -> no pump

    Returns ``(num_pumps, list_of_optimal_labelings)``.
    """
    n = space.num_modes
    edges = []
    for value, (kind, i, j) in zip(graph, space.slots):
        if value == NO_COUPLING:
            continue
        edges.append((kind, i, j, int(value)))

    def count(labels):
        num = 0
        for kind, i, j, value in edges:
            if kind == SLOT_DETUNING:
                continue
            if kind == SLOT_ONSITE_SQUEEZING:
                num += 1
                continue
            if kind == SLOT_TWO_MODE_SQUEEZING:
                num += 1
                continue
            # beam-splitter
            if value == COUPLING_WITH_PHASE and labels[i] == labels[j]:
                return None  # a complex passive coupling needs a pump-imprinted phase
            if labels[i] != labels[j]:
                num += 1
        return num

    best = None
    best_labels = []
    # canonical (restricted-growth) labelings only
    ranges = [range(idx + 1) for idx in range(n)]
    for labels in product(*ranges):
        labels = np.array(labels)
        result = count(labels)
        if result is None:
            continue
        if best is None or result < best:
            best = result
            best_labels = [labels]
        elif result == best:
            best_labels.append(labels)
    if best is None:
        return np.inf, []
    return best, best_labels


def characterize_architectures(graphs, space):
    """Vectorised :func:`characterize_architecture` + pump count."""
    infos = []
    for graph in graphs:
        info = characterize_architecture(graph, space)
        info["min_number_of_pumps"], _ = min_number_of_pumps(graph, space)
        infos.append(info)
    keys = infos[0].keys() if infos else []
    return {key: np.array([info[key] for info in infos]) for key in keys}


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_graph(
    graph,
    space,
    node_colors=None,
    positions=None,
    ax=None,
    color_passive="black",
    color_passive_complex="green",
    color_squeezing="blue",
    edge_width=2,
):
    """Draw one device graph.

    black = real beam-splitter, green = complex beam-splitter (synthetic flux),
    blue = squeezing (two-mode edge, or a self-loop marker for on-site
    squeezing).  Detuned modes are drawn with a thicker node border.
    """
    import networkx as nx

    n = space.num_modes
    if node_colors is None:
        node_colors = ["tab:blue"] * n

    G = nx.Graph()
    G.add_nodes_from(range(n))
    detuned = set()
    onsite = set()
    edge_colors = {}
    for value, (kind, i, j) in zip(graph, space.slots):
        if value == NO_COUPLING:
            continue
        if kind == SLOT_DETUNING:
            detuned.add(i)
        elif kind == SLOT_ONSITE_SQUEEZING:
            onsite.add(i)
        elif kind == SLOT_BEAMSPLITTER:
            color = color_passive_complex if value == COUPLING_WITH_PHASE else color_passive
            edge_colors[(i, j)] = color
        else:
            edge_colors[(i, j)] = color_squeezing
    for (i, j), color in edge_colors.items():
        G.add_edge(i, j, color=color)

    if positions is None:
        positions = nx.circular_layout(G)

    borders = [3.0 if idx in detuned else 0.5 for idx in range(n)]
    shapes = ["s" if idx in onsite else "o" for idx in range(n)]

    nx.draw_networkx_edges(
        G, positions, ax=ax,
        edge_color=[G[u][v]["color"] for u, v in G.edges],
        width=edge_width,
    )
    for shape in set(shapes):
        idxs = [idx for idx in range(n) if shapes[idx] == shape]
        nx.draw_networkx_nodes(
            G, positions, ax=ax, nodelist=idxs, node_shape=shape,
            node_color=[node_colors[idx] for idx in idxs],
            linewidths=[borders[idx] for idx in idxs], edgecolors="black",
        )
    nx.draw_networkx_labels(G, positions, ax=ax)
    if ax is not None:
        ax.axis("off")
    return G


def plot_list_of_graphs(graphs, space, graphs_per_row=5, size=2.5, **kwargs):
    import matplotlib.pyplot as plt

    num_rows = (len(graphs) - 1) // graphs_per_row + 1
    num_cols = min(len(graphs), graphs_per_row)
    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(size * num_cols, size * num_rows),
                             squeeze=False)
    for ax in axes.flatten():
        ax.axis("off")
    for idx, graph in enumerate(graphs):
        plot_graph(graph, space, ax=axes.flatten()[idx], **kwargs)
    return fig, axes
