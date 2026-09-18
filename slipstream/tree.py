"""The round's draft tree.

The client holds the whole tree: tokens, the drafter's path probabilities, and
the acceptance probabilities it estimates. The provider holds only what
Appendix C allows it to hold, each position's parent and whether its hidden
state has arrived and run, which is enough to build the attention mask and to
apply a verdict. `DraftTree` is the client's view; `ProviderTree` in
`slipstream/provider/state.py` is the provider's.

Definitions follow Section 2.2 and Section 4.2:

  path probability q_v  product of the drafter's token probabilities along the
                        path from the root to v
  rank                  position of a node within its level, ordered by q_v,
                        rank 1 being the most probable
  trunk                 the chain that follows the most probable child at
                        every level; everything else is a branch
"""

from __future__ import annotations

from dataclasses import dataclass, field

ROOT = 0


@dataclass
class Node:
    """One draft token. `position` is what crosses the network."""

    position: int
    parent: int
    depth: int
    # Token identifier and path probability never leave the client.
    token: int | None = None
    q: float = 1.0
    # Acceptance probability, estimated by the client from past verdicts.
    alpha: float = 0.0
    children: list[int] = field(default_factory=list)
    # Rank within the node's level, 1 being the most probable. Assigned by
    # `assign_ranks`.
    rank: int = 1
    # Bookkeeping the scheduler and the transport fill in.
    admitted: bool = False
    sent: bool = False
    in_first_segment: bool = False
    output_received: bool = False
    accepted: bool = False


class DraftTree:
    """A tree of draft tokens rooted at the last accepted token.

    The root is position 0 and carries the token the previous round accepted
    last. Its hidden state travels with the first segment, so a first segment
    of `s` draft tokens carries `s + 1` hidden states (Section 4.2).
    """

    def __init__(self, root_token: int, round_id: int = 0) -> None:
        self.round_id = round_id
        self.nodes: list[Node] = [
            Node(position=ROOT, parent=ROOT, depth=0, token=root_token, q=1.0, alpha=1.0)
        ]

    # ---------------------------------------------------------------- basics

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, position: int) -> Node:
        return self.nodes[position]

    @property
    def root(self) -> Node:
        return self.nodes[ROOT]

    def add_children(
        self, parent: int, tokens: list[int], probs: list[float]
    ) -> list[int]:
        """Add one drafting step's children, produced together from one
        softmax (Section 4.2). Returns their positions."""
        out = []
        p = self.nodes[parent]
        for token, prob in zip(tokens, probs):
            position = len(self.nodes)
            node = Node(
                position=position,
                parent=parent,
                depth=p.depth + 1,
                token=int(token),
                q=p.q * float(prob),
            )
            self.nodes.append(node)
            p.children.append(position)
            out.append(position)
        return out

    def ancestors(self, position: int) -> list[int]:
        """Positions from the root down to `position`, inclusive."""
        chain = [position]
        while chain[-1] != ROOT:
            chain.append(self.nodes[chain[-1]].parent)
        chain.reverse()
        return chain

    def descendants(self, position: int) -> list[int]:
        """`position` and everything under it, in breadth-first order."""
        out, frontier = [position], [position]
        while frontier:
            nxt = []
            for p in frontier:
                for c in self.nodes[p].children:
                    out.append(c)
                    nxt.append(c)
            frontier = nxt
        return out

    def levels(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for node in self.nodes:
            out.setdefault(node.depth, []).append(node.position)
        return out

    def assign_ranks(self) -> None:
        """Order each level by path probability and record the rank. Ties keep
        the drafting order, so ranks are deterministic."""
        for _, positions in self.levels().items():
            ordered = sorted(positions, key=lambda p: (-self.nodes[p].q, p))
            for rank, p in enumerate(ordered, start=1):
                self.nodes[p].rank = rank

    def trunk(self) -> list[int]:
        """The chain that follows the most probable child at every level,
        excluding the root."""
        out: list[int] = []
        cur = ROOT
        while self.nodes[cur].children:
            cur = max(self.nodes[cur].children, key=lambda p: (self.nodes[p].q, -p))
            out.append(cur)
        return out

    def is_branch(self, position: int) -> bool:
        return position not in set(self.trunk()) and position != ROOT

    # ------------------------------------------------------- one-shot rerank

    def frontier(self) -> list[int]:
        """Drafted tokens that have no children yet, the candidates the
        best-first drafter may expand next."""
        return [n.position for n in self.nodes if not n.children and n.position != ROOT]

    def best_first_candidate(self) -> int | None:
        """The drafted token with the highest path probability that has no
        children yet (Section 4.2). The root is expanded first."""
        if not self.nodes[ROOT].children:
            return ROOT
        pending = self.frontier()
        if not pending:
            return None
        return max(pending, key=lambda p: (self.nodes[p].q, -p))

    def connected_top_m(self, m: int) -> list[int]:
        """The `m` draft tokens with the highest path probability that form a
        connected tree with the root, the rerank of Section 2.2.

        Greedy by path probability: a node may enter only once its parent has,
        which holds automatically because a child's q is at most its parent's.
        """
        chosen: set[int] = {ROOT}
        order = sorted(
            (n for n in self.nodes if n.position != ROOT),
            key=lambda n: (-n.q, n.position),
        )
        out: list[int] = []
        for node in order:
            if len(out) >= m:
                break
            if node.parent in chosen:
                chosen.add(node.position)
                out.append(node.position)
        return out

    def subtree(self, positions: list[int]) -> "DraftTree":
        """A copy holding the root and `positions`, renumbered from 0 in the
        same order. Used to ship a reranked one-shot tree."""
        keep = [ROOT] + [p for p in positions if p != ROOT]
        remap = {old: new for new, old in enumerate(keep)}
        out = DraftTree(root_token=self.nodes[ROOT].token or 0, round_id=self.round_id)
        out.nodes[ROOT].q = self.nodes[ROOT].q
        for old in keep[1:]:
            src = self.nodes[old]
            position = len(out.nodes)
            out.nodes.append(
                Node(
                    position=position,
                    parent=remap[src.parent],
                    depth=src.depth,
                    token=src.token,
                    q=src.q,
                    alpha=src.alpha,
                )
            )
            out.nodes[remap[src.parent]].children.append(position)
        out.assign_ranks()
        return out

    # ------------------------------------------------------------- attention

    def ancestor_mask(self, positions: list[int]) -> list[list[bool]]:
        """Exact ancestor mask for a batch of positions.

        Entry (i, j) is True when `positions[j]` is an ancestor of
        `positions[i]` or is `positions[i]` itself, which is exactly the set of
        tree positions a draft token may attend to. The provider builds the
        same mask from parent pointers alone.
        """
        index = {p: i for i, p in enumerate(positions)}
        mask = [[False] * len(positions) for _ in positions]
        for i, p in enumerate(positions):
            for a in self.ancestors(p):
                j = index.get(a)
                if j is not None:
                    mask[i][j] = True
        return mask

    def path_stats(self) -> dict:
        """Counts the motivation study reports: nodes per level and per rank."""
        self.assign_ranks()
        per_rank: dict[int, int] = {}
        for node in self.nodes:
            if node.position == ROOT:
                continue
            per_rank[node.rank] = per_rank.get(node.rank, 0) + 1
        return {
            "nodes": len(self.nodes) - 1,
            "depth": max(n.depth for n in self.nodes),
            "per_rank": per_rank,
            "trunk": self.trunk(),
        }


def candidate_pool_size(depth: int, width: int) -> int:
    """Draft tokens a dynamic tree of this shape can offer, the candidate pool
    of Section 3.2.

    The drafter proposes the `width` most probable next tokens, and for each of
    them `width` next tokens, of which it expands only the `width` with the
    highest path probability, repeating for `depth` levels (Section 2.2). So
    the first level contributes `width` candidates and every level after it
    `width * width`:

        pool = width + (depth - 1) * width ** 2

    which is 22 for a (6, 2) tree, 84 for a (6, 4) tree, and 84 again for a
    (10, 3) tree, the three pools the budget sweep runs into.
    """
    if depth <= 0 or width <= 0:
        return 0
    return width + (depth - 1) * width * width
