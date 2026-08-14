"""Hypothesis log tests. The property that matters: nothing gets silently lost
or overwritten, and a story-free entry cannot be logged at all."""

from __future__ import annotations

import pytest

from dtest.evaluate.hypothesis_log import HypothesisEntry, append_entry, load_log, scoreboard


def _entry(**kw):
    base = dict(
        title="delivery-confirmed breakout", story="delivered shares reflect "
        "committed capital, not intraday noise, so a breakout confirmed by "
        "delivery %% should persist longer than one confirmed by raw volume",
        split="delivery", window="train", metric="mean_net_pct",
        real_value=0.42, decision="accepted",
    )
    base.update(kw)
    return HypothesisEntry(**base)


def test_story_is_mandatory():
    with pytest.raises(ValueError, match="story"):
        HypothesisEntry(title="x", story="   ", split="primary", window="train",
                        metric="m", real_value=0.0, decision="rejected")


def test_decision_must_be_a_known_value():
    with pytest.raises(ValueError, match="decision"):
        _entry(decision="maybe")


def test_window_must_be_train_val_or_test():
    with pytest.raises(ValueError, match="window"):
        _entry(window="everything")


def test_append_and_reload_round_trips(tmp_path):
    path = tmp_path / "log.csv"
    append_entry(path, _entry(title="A", decision="accepted"))
    append_entry(path, _entry(title="B", decision="rejected"))
    log = load_log(path)
    assert list(log["title"]) == ["A", "B"]
    assert list(log["decision"]) == ["accepted", "rejected"]


def test_existing_rows_are_never_mutated_by_a_later_append(tmp_path):
    path = tmp_path / "log.csv"
    append_entry(path, _entry(title="first", real_value=1.23))
    before = load_log(path)
    append_entry(path, _entry(title="second", real_value=4.56))
    after = load_log(path)
    pd_row = after[after["title"] == "first"].iloc[0]
    assert pd_row["real_value"] == pytest.approx(1.23)
    assert len(after) == len(before) + 1


def test_rejections_are_logged_with_the_same_shape_as_acceptances(tmp_path):
    path = tmp_path / "log.csv"
    append_entry(path, _entry(title="rejected one", decision="rejected",
                              placebo_max=0.8, real_value=0.3))
    log = load_log(path)
    row = log.iloc[0]
    assert row["decision"] == "rejected"
    assert row["placebo_max"] == pytest.approx(0.8)


def test_each_entry_gets_a_unique_id():
    a, b = _entry(), _entry()
    assert a.hypothesis_id != b.hypothesis_id


def test_supersedes_links_to_an_earlier_entry(tmp_path):
    path = tmp_path / "log.csv"
    append_entry(path, _entry(title="v1", decision="rejected"))
    first_id = load_log(path).iloc[0]["hypothesis_id"]
    append_entry(path, _entry(title="v2", decision="accepted", supersedes=first_id))
    log = load_log(path)
    assert log.iloc[1]["supersedes"] == first_id


def test_scoreboard_counts_by_split_and_window(tmp_path):
    path = tmp_path / "log.csv"
    append_entry(path, _entry(split="primary", window="train", decision="accepted"))
    append_entry(path, _entry(split="primary", window="train", decision="rejected"))
    append_entry(path, _entry(split="primary", window="train", decision="rejected"))
    append_entry(path, _entry(split="delivery", window="train", decision="accepted"))
    board = scoreboard(path)
    primary = board[(board["split"] == "primary") & (board["window"] == "train")].iloc[0]
    assert primary["n_tried"] == 3
    assert primary["n_accepted"] == 1
    assert primary["n_rejected"] == 2


def test_scoreboard_on_empty_log_is_empty_not_a_crash(tmp_path):
    board = scoreboard(tmp_path / "does_not_exist.csv")
    assert board.empty
