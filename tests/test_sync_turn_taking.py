"""Tests for sync rounds and their speaking order (TurnPolicy).

Every sync round is turn-taking: it runs one agent at a time, in a per-round order,
and each agent drains its queue when its own turn arrives — so a later speaker sees
what earlier speakers said in the SAME round. TurnPolicy only decides who goes when.
"""

import json
import logging
import tempfile
from pathlib import Path

import pytest

from magelab.org_config import OrgSettings, WireNotifications
from magelab.orchestrator import Orchestrator, TurnPolicy
from magelab.registry_config import AgentConfig, RoleConfig
from magelab.state.database import Database
from magelab.state.registry import Registry
from magelab.state.task_schemas import Task, TaskStatus
from magelab.state.task_store import TaskStore
from magelab.state.wire_store import WireStore

from .conftest import open_db_for_query

_test_logger = logging.getLogger("test")

ASSEMBLY = ["facilitator", "deliberator_1", "deliberator_2", "deliberator_3"]


def _make_org(tmp_dir: Path):
    roles = {
        "member": RoleConfig(
            name="member",
            role_prompt="You deliberate.",
            tools=["worker", "claude_basic", "communication"],
            model="test",
            max_turns=10,
        ),
    }
    agents = {a: AgentConfig(agent_id=a, role="member") for a in ASSEMBLY}
    db = Database(tmp_dir / "org.db")
    db.init_run_meta(org_name="test", org_config="{}")
    store = TaskStore(framework_logger=_test_logger, db=db)
    registry = Registry(framework_logger=_test_logger, db=db)
    registry.register_config(roles, agents)

    from .conftest import MockRunner

    runner = MockRunner()
    wire_store = WireStore(framework_logger=_test_logger, db=db, wire_notifications=WireNotifications.EVENT)
    orch = Orchestrator(store, registry, runner, wire_store, db, 30.0, "Test org", "/test/workspace")
    return store, registry, runner, wire_store, db, orch


def _load_turn_orders(db: Database) -> list[dict]:
    """Read run_turn_orders through a fresh connection (the run closes the DB)."""
    conn = open_db_for_query(db)
    rows = conn.execute("SELECT * FROM run_turn_orders ORDER BY id").fetchall()
    conn.close()
    return [
        {"round": r["round_num"], "seed": r["seed"], "turn_order": json.loads(r["turn_order"]),
         "spoke": json.loads(r["spoke"])}
        for r in rows
    ]


def _chatter(wire_store: WireStore, sender: str, body: str = "hi"):
    """Side effect: keep the assembly wire alive so rounds keep firing."""

    async def _effect():
        await wire_store.add_message("assembly", sender=sender, body=body)

    return _effect


def _kickoff(wire_id: str = "assembly"):
    return [{"wire_id": wire_id, "participants": ASSEMBLY, "sender": "Organizers", "body": "Begin."}]


# =============================================================================
# Ordering
# =============================================================================


class TestTurnOrder:
    @pytest.mark.asyncio
    async def test_agents_run_one_at_a_time_in_drawn_order(self, tmp_path):
        """Every dispatch of a round completes before the next agent starts."""
        _, _, runner, _, db, orch = _make_org(tmp_path)
        policy = TurnPolicy(order="random", seed=7, first=("facilitator",))

        await orch.run(initial_messages=_kickoff(), sync=True, sync_max_rounds=1, turn_policy=policy)

        round_1 = orch.turn_orders[0]
        assert round_1[0] == "facilitator"
        assert sorted(round_1) == sorted(ASSEMBLY)
        # Agents ran in exactly the drawn order, one at a time.
        assert [c[0] for c in runner.calls] == round_1

    @pytest.mark.asyncio
    async def test_pinned_agents_open_every_round(self, tmp_path):
        store, _, runner, wire_store, db, orch = _make_org(tmp_path)
        policy = TurnPolicy(order="random", seed=99, first=("facilitator",))

        # Keep the room talking for several rounds: each speaker re-broadcasts.
        for a in ASSEMBLY:
            runner.side_effects[a] = [_chatter(wire_store, a) for _ in range(4)]

        await orch.run(initial_messages=_kickoff(), sync=True, sync_max_rounds=4, turn_policy=policy)

        assert len(orch.turn_orders) >= 3
        assert all(order[0] == "facilitator" for order in orch.turn_orders)

    @pytest.mark.asyncio
    async def test_order_is_reshuffled_across_rounds(self, tmp_path):
        """With enough rounds, the deliberator order is not identical every round."""
        _, _, runner, wire_store, db, orch = _make_org(tmp_path)
        policy = TurnPolicy(order="random", seed=1234, first=("facilitator",))

        for a in ASSEMBLY:
            runner.side_effects[a] = [_chatter(wire_store, a) for _ in range(8)]

        await orch.run(initial_messages=_kickoff(), sync=True, sync_max_rounds=8, turn_policy=policy)

        tails = {tuple(order[1:]) for order in orch.turn_orders}
        assert len(tails) > 1, f"order never changed across {len(orch.turn_orders)} rounds"

    @pytest.mark.asyncio
    async def test_same_seed_replays_same_orders(self, tmp_path):
        orders = []
        for i in range(2):
            run_dir = tmp_path / f"run{i}"
            run_dir.mkdir()
            _, _, runner, wire_store, db, orch = _make_org(run_dir)
            policy = TurnPolicy(order="random", seed=555, first=("facilitator",))

            for a in ASSEMBLY:
                runner.side_effects[a] = [_chatter(wire_store, a) for _ in range(5)]
            await orch.run(initial_messages=_kickoff(), sync=True, sync_max_rounds=5, turn_policy=policy)
            orders.append(orch.turn_orders)

        assert orders[0] == orders[1]

    @pytest.mark.asyncio
    async def test_config_order_is_registry_order(self, tmp_path):
        _, registry, runner, _, db, orch = _make_org(tmp_path)
        policy = TurnPolicy(order="config")

        await orch.run(initial_messages=_kickoff(), sync=True, sync_max_rounds=1, turn_policy=policy)

        assert orch.turn_orders[0] == registry.list_agent_ids()


# =============================================================================
# Within-round visibility — the point of the feature
# =============================================================================


class TestWithinRoundDelivery:
    @pytest.mark.asyncio
    async def test_later_speaker_sees_earlier_speaker_same_round(self, tmp_path):
        """A message sent by the round's first speaker reaches a later speaker in the
        SAME round — the behaviour concurrent rounds cannot produce."""
        _, _, runner, wire_store, db, orch = _make_org(tmp_path)
        policy = TurnPolicy(order="config", first=("facilitator",))

        runner.side_effects["facilitator"] = [_chatter(wire_store, "facilitator", "AGENDA-TOKEN")]

        await orch.run(initial_messages=_kickoff(), sync=True, sync_max_rounds=1, turn_policy=policy)

        order = orch.turn_orders[0]
        later = order[-1]
        later_prompts = [c[2] for c in runner.calls if c[0] == later]
        assert any("AGENDA-TOKEN" in p for p in later_prompts), (
            f"{later} spoke after facilitator but never saw the same-round message"
        )

    @pytest.mark.asyncio
    async def test_a_sync_run_takes_turns_without_being_asked_to(self, tmp_path):
        """There is no opt-out: a caller that passes no policy still gets a turn order.

        This is the regression guard for the removal of the all-at-once round. A sync
        run that drew no order used to be a legitimate mode; now it would mean some
        path is still dispatching a round without ordering it, which is exactly the
        race the removal was meant to end.
        """
        _, _, runner, wire_store, db, orch = _make_org(tmp_path)

        runner.side_effects["facilitator"] = [_chatter(wire_store, "facilitator", "AGENDA-TOKEN")]

        await orch.run(initial_messages=_kickoff(), sync=True, sync_max_rounds=1)

        assert len(orch.turn_orders) == 1
        assert sorted(orch.turn_orders[0]) == sorted(ASSEMBLY)
        rows = _load_turn_orders(db)
        assert len(rows) == 1 and rows[0]["seed"] is not None

    @pytest.mark.asyncio
    async def test_message_arriving_after_your_turn_waits_for_next_round(self, tmp_path):
        """An agent that already spoke does not get a second turn in the same round."""
        _, _, runner, wire_store, db, orch = _make_org(tmp_path)
        policy = TurnPolicy(order="config", first=("facilitator",))

        runner.side_effects[ASSEMBLY[-1]] = [_chatter(wire_store, ASSEMBLY[-1], "LATE-TOKEN")]

        await orch.run(initial_messages=_kickoff(), sync=True, sync_max_rounds=1, turn_policy=policy)

        # Round budget was 1, so nobody ever saw LATE-TOKEN.
        assert not any("LATE-TOKEN" in c[2] for c in runner.calls)
        # Exactly one turn per agent in the round.
        assert len(runner.calls) == len(ASSEMBLY)


# =============================================================================
# Recording and convergence
# =============================================================================


class TestRecordingAndConvergence:
    @pytest.mark.asyncio
    async def test_turn_orders_recorded_to_db(self, tmp_path):
        _, _, runner, _, db, orch = _make_org(tmp_path)
        policy = TurnPolicy(order="random", seed=42, first=("facilitator",))

        await orch.run(initial_messages=_kickoff(), sync=True, sync_max_rounds=1, turn_policy=policy)

        rows = _load_turn_orders(db)
        assert len(rows) == 1
        assert rows[0]["round"] == 1
        assert rows[0]["seed"] == 42
        assert rows[0]["turn_order"] == orch.turn_orders[0]
        assert rows[0]["spoke"] == orch.turn_orders[0]

    @pytest.mark.asyncio
    async def test_seed_is_drawn_and_recorded_when_unset(self, tmp_path):
        _, _, runner, _, db, orch = _make_org(tmp_path)

        await orch.run(initial_messages=_kickoff(), sync=True, sync_max_rounds=1, turn_policy=TurnPolicy())

        rows = _load_turn_orders(db)
        assert rows[0]["seed"] is not None

    @pytest.mark.asyncio
    async def test_silent_round_converges_without_recording(self, tmp_path):
        """A round in which nobody has anything queued ends the run and is not recorded."""
        _, _, runner, _, db, orch = _make_org(tmp_path)
        policy = TurnPolicy(order="config")

        await orch.run(initial_messages=_kickoff(), sync=True, sync_max_rounds=6, turn_policy=policy)

        assert orch.sync_rounds == 1
        assert len(_load_turn_orders(db)) == 1

    @pytest.mark.asyncio
    async def test_turn_policy_requires_sync(self, tmp_path):
        _, _, _, _, _, orch = _make_org(tmp_path)
        with pytest.raises(ValueError, match="turn_policy can only be specified when sync=True"):
            await orch.run(initial_messages=_kickoff(), turn_policy=TurnPolicy())


# =============================================================================
# Settings plumbing
# =============================================================================


class TestSettings:
    def test_from_settings(self):
        s = OrgSettings(
            org_name="x",
            sync=True,
            sync_max_rounds=5,
            sync_turn_order="random",
            sync_turn_seed=17,
            sync_turn_first=["facilitator"],
        )
        policy = TurnPolicy.from_settings(s)
        assert policy == TurnPolicy(order="random", seed=17, first=("facilitator",))

    def test_defaults_are_a_random_unpinned_order(self):
        assert TurnPolicy.from_settings(OrgSettings(org_name="x")) == TurnPolicy(order="random", seed=None, first=())

    def test_turn_fields_require_sync(self):
        with pytest.raises(ValueError, match="sync_turn_seed can only be specified when sync=True"):
            OrgSettings(org_name="x", sync_turn_seed=3)
        with pytest.raises(ValueError, match="sync_turn_first can only be specified when sync=True"):
            OrgSettings(org_name="x", sync_turn_first=["facilitator"])

    def test_bad_order_rejected(self):
        with pytest.raises(ValueError, match="sync_turn_order must be"):
            OrgSettings(org_name="x", sync=True, sync_max_rounds=3, sync_turn_order="rotate")


class TestDraw:
    def test_draw_is_pure_and_matches_the_run(self):
        """The order a caller can draw ahead of time is the order the round uses."""
        policy = TurnPolicy(order="random", seed=4242, first=("facilitator",))
        drawn = [policy.draw(ASSEMBLY, r) for r in range(1, 4)]
        assert drawn == [policy.draw(ASSEMBLY, r) for r in range(1, 4)]
        assert all(o[0] == "facilitator" for o in drawn)
        assert all(sorted(o) == sorted(ASSEMBLY) for o in drawn)

    def test_draw_ignores_pinned_ids_that_are_not_agents(self):
        policy = TurnPolicy(order="config", first=("ghost", "facilitator"))
        assert policy.draw(ASSEMBLY, 1) == ["facilitator"] + [a for a in ASSEMBLY if a != "facilitator"]
