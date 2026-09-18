"""The one-shot methods: a chain or a tree, shipped in one piece.

Section 6.1. Both draft with the multi-token-prediction module and ship the
draft in one piece, so every drafted token is on the critical path. Their
budget adapts online: a client records the goodput that each budget from 3 to
22, the range of Section 3.2, gave over its recent rounds, and uses the best,
so each method settles at the best budget for its load. That is the working
point draft-size controllers seek, and it is why One-shot tree stands for them
in the evaluation.

One-shot tree uses the same best-first tree as Slipstream, with four children
per expanded token, and reranks it to the budget the way Section 2.2
describes. One-shot chain keeps one guess per position.
"""

from __future__ import annotations

import random
import time

from ..client.engine import RequestState, RoundResult
from ..scheduler import Frame, Klass
from ..tree import ROOT, DraftTree
from .base import Method


class BudgetSearch:
    """Online search over the draft budget.

    The client keeps the goodput each budget gave over its recent rounds and
    uses the best, trying a neighbour now and then so the estimate follows the
    load.
    """

    def __init__(
        self,
        low: int = 3,
        high: int = 22,
        explore: float = 0.1,
        weight: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.low, self.high = low, high
        self.explore = explore
        self.weight = weight
        self.rng = random.Random(seed)
        self.goodput: dict[int, float] = {}
        self.current = low

    def choose(self) -> int:
        unseen = [b for b in range(self.low, self.high + 1) if b not in self.goodput]
        if unseen:
            self.current = unseen[0]
        elif self.rng.random() < self.explore:
            step = self.rng.choice([-2, -1, 1, 2])
            self.current = max(self.low, min(self.high, self.current + step))
        else:
            self.current = max(self.goodput, key=self.goodput.get)
        return self.current

    def observe(self, budget: int, tokens: int, round_ms: float) -> None:
        if round_ms <= 0:
            return
        goodput = tokens / (round_ms / 1e3)
        old = self.goodput.get(budget)
        self.goodput[budget] = (
            goodput if old is None else (1 - self.weight) * old + self.weight * goodput
        )

    def best(self) -> int:
        return max(self.goodput, key=self.goodput.get) if self.goodput else self.low


class OneShot(Method):
    """Shared by the chain and the tree."""

    shape = "tree"
    name = "One-shot tree"

    def __init__(self, engine, adaptive: bool = True) -> None:
        super().__init__(engine)
        low, high = engine.cfg.decode.budget_search
        self.search = BudgetSearch(low=low, high=high, seed=engine.cfg.seed)
        self.adaptive = adaptive

    def budget(self) -> int:
        if self.adaptive:
            return self.search.choose()
        return self.engine.cfg.decode.draft_budget

    def run_round(self, state: RequestState) -> RoundResult:
        engine = self.engine
        started = time.monotonic()
        budget = self.budget()
        tree = self.new_tree(state)
        self.send_root(state, tree)

        # --- draft the whole tree before sending any of it ----------------
        drafter_hidden = {ROOT: state.seed_hidden}
        drafted = 0
        while drafted < budget:
            candidate = self._candidate(tree, budget)
            if candidate is None:
                break
            seed = drafter_hidden.get(candidate)
            if seed is None:
                break
            width = 1 if self.shape == "chain" else engine.cfg.drafter.children_per_step
            proposal = engine.drafter.propose(
                seed,
                tree[candidate].token,
                engine.absolute_position(state, tree[candidate].depth),
                width=width,
            )
            positions = tree.add_children(candidate, proposal.tokens, proposal.probs)
            for position, hidden in zip(positions, proposal.hidden):
                drafter_hidden[position] = hidden
            drafted += len(positions)

        # The rerank keeps the `budget` nodes with the highest path
        # probability, which form a connected tree.
        keep = (
            tree.connected_top_m(budget)
            if self.shape == "tree"
            else [p for p in tree.trunk()][:budget]
        )
        keep_set = set(keep)

        # --- ship it in one piece ------------------------------------------
        shipped = [ROOT] + [p for p in sorted(keep_set) if p != ROOT]
        for index, position in enumerate(shipped):
            node = tree[position]
            hidden = (
                self._root_hidden
                if position == ROOT
                else self.encode_node(state, tree, position)
            )
            frame = Frame(
                request_id=state.request_id,
                round_id=tree.round_id,
                position=position,
                parent=node.parent,
                alpha=node.q,
                klass=Klass.FIRST_SEGMENT,
                precision="bf16",
                output_precision="bf16",
                payload_bytes=2 * int(hidden.numel()),
            )
            node.sent = True
            engine.send(
                state, frame, hidden, first_segment_end=(index == len(shipped) - 1)
            )

        # --- one verification ----------------------------------------------
        engine.wait_for(state, shipped, timeout_s=30.0)
        accepted, cursor, target, matched = engine.verify(state, tree, keep_set | {ROOT})
        bonus = target if matched is None else None
        commit_leaf = accepted[-1] if accepted else ROOT
        if engine.output_of(state, commit_leaf) is None:
            commit_leaf = ROOT
        accepted_tokens = [tree[p].token for p in accepted]
        engine.close_round(state, tree, accepted, bonus, commit_leaf)

        round_ms = (time.monotonic() - started) * 1e3
        advanced = len(accepted) + (1 if bonus is not None else 0)
        self.search.observe(budget, advanced, round_ms)
        engine.scheduler.drop_round(tree.round_id)

        return RoundResult(
            accepted_positions=accepted,
            accepted_tokens=[t for t in accepted_tokens if t is not None],
            bonus_token=bonus,
            first_segment_length=len(shipped) - 1,
            drafted=drafted,
            sent=len(shipped),
            continuations=0,
            round_ms=round_ms,
            commit_leaf=commit_leaf,
        )

    def _candidate(self, tree: DraftTree, budget: int) -> int | None:
        if self.shape == "chain":
            trunk = tree.trunk()
            if len(trunk) >= budget:
                return None
            return trunk[-1] if trunk else ROOT
        return tree.best_first_candidate()


class OneShotChain(OneShot):
    shape = "chain"
    name = "One-shot chain"


class OneShotTree(OneShot):
    shape = "tree"
    name = "One-shot tree"
