"""Synthetic rank sets exercise the arena's evidence boundaries, without models."""

import unittest

from shoplens.studio_charts import MODES, arena_data, rank_story_figure


def rows(*items):
    return [{"item_id": item, "rank": rank, "score": 100 - rank} for rank, item in enumerate(items, 1)]


class ModelArenaTests(unittest.TestCase):
    def test_membership_partitions_the_actual_union_once(self):
        data = arena_data(dict(zip(MODES, (rows("shared", "pair", "frozen"),
                                          rows("pair", "shared", "semantic"),
                                          rows("preference", "shared")))))
        self.assertTrue(data["complete"])
        self.assertEqual(data["union"], 5)
        self.assertEqual(data["all_three"], 1)
        self.assertEqual(data["two_models"], 1)
        self.assertEqual(data["exclusive"], dict.fromkeys(MODES, 1))
        self.assertEqual(data["all_three"] + data["two_models"] + sum(data["exclusive"].values()), data["union"])
        self.assertEqual(data["membership"]["pair"], MODES[:2])

    def test_failed_or_missing_model_does_not_infer_exclusivity(self):
        groups = dict(zip(MODES, (rows("a"), rows("b"), rows("a"))))
        for data in (arena_data(groups, {MODES[1]: "diagnostic failure"}),
                     arena_data({MODES[0]: rows("a"), MODES[2]: rows("a")})):
            with self.subTest(modes=data["modes"]):
                self.assertFalse(data["complete"])
                self.assertIsNone(data["all_three"])
                self.assertIsNone(data["exclusive"])
                self.assertNotIn("b", data["item_ids"])
                self.assertNotIn(MODES[1], data["ranks"])

    def test_genuine_empty_list_is_observed_not_an_error(self):
        data = arena_data(dict(zip(MODES, (rows("a"), [], rows("a")))))
        self.assertTrue(data["complete"])
        self.assertEqual(data["two_models"], 1)
        self.assertEqual(data["all_three"], 0)
        self.assertEqual(data["ranks"][MODES[1]], {})
        self.assertEqual(arena_data(dict.fromkeys(MODES, []))["union"], 0)

    def test_focus_keeps_missing_rank_unknown_and_does_not_connect_gap(self):
        groups = dict(zip(MODES, (rows("selected", "other"), rows("other"), rows("other", "selected"))))
        data = arena_data(groups)
        figure = rank_story_figure(data, {"selected": {"title": "DIAGNOSTIC product"}}, "selected")
        focus = figure.data[-1]
        self.assertEqual(focus.name, "selected")
        self.assertEqual(list(focus.y), [1, None, 2])
        self.assertFalse(focus.connectgaps)
        self.assertEqual(list(focus.text), ["#1", "", "#2"])
        self.assertEqual(len(figure.data), 2)
        self.assertFalse(any(trace.showlegend for trace in figure.data))
        self.assertEqual(list(figure.layout.xaxis.ticktext), ["Frozen", "Semantic", "Preference"])

    def test_scores_do_not_change_rank_story_and_old_selection_is_rejected(self):
        groups = {mode: rows("a", "b") for mode in MODES}
        before = rank_story_figure(arena_data(groups), {}, "a").to_json()
        for model_rows in groups.values():
            for row in model_rows:
                row["score"] = float("nan")
        self.assertEqual(before, rank_story_figure(arena_data(groups), {}, "a").to_json())
        with self.assertRaises(ValueError):
            rank_story_figure(arena_data(groups), {}, "old request item")

    def test_invalid_recorded_positions_are_rejected(self):
        for bad in ([{"item_id": "a", "rank": 0}],
                    [{"item_id": "a", "rank": 1}, {"item_id": "b", "rank": 1}],
                    [{"item_id": "a", "rank": 1}, {"item_id": "a", "rank": 2}]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                arena_data({mode: bad for mode in MODES})
