"""
Validation of the implementation against hand-calculated reference values:
  * the ID3 and C4.5 examples of the article (Play Tennis data),
  * Quinlan (1993): gain and split info with one unknown Outlook value,
  * the binomial upper limit used by error-based pruning,
  * edge cases raised in review (pure/empty nodes, input tree not modified, ...).

Run:  python test_examples.py      (or: python -m pytest test_examples.py)
"""

import copy
import importlib

import numpy as np
import pandas as pd

id3 = importlib.import_module("01_id3")
c45 = importlib.import_module("02_c45")
viz = importlib.import_module("04_visualize")

COLUMNS = ["Day", "Outlook", "Temperature", "Humidity", "Wind", "Play"]
CATEGORICAL = pd.DataFrame([
    [1, "Sunny", "Hot", "High", "Weak", "No"],
    [2, "Sunny", "Hot", "High", "Strong", "No"],
    [3, "Overcast", "Hot", "High", "Weak", "Yes"],
    [4, "Rain", "Mild", "High", "Weak", "Yes"],
    [5, "Rain", "Cool", "Normal", "Weak", "Yes"],
    [6, "Rain", "Cool", "Normal", "Strong", "No"],
    [7, "Overcast", "Cool", "Normal", "Strong", "Yes"],
    [8, "Sunny", "Mild", "High", "Weak", "No"],
    [9, "Sunny", "Cool", "Normal", "Weak", "Yes"],
    [10, "Rain", "Mild", "Normal", "Weak", "Yes"],
    [11, "Sunny", "Mild", "Normal", "Strong", "Yes"],
    [12, "Overcast", "Mild", "High", "Strong", "Yes"],
    [13, "Overcast", "Hot", "Normal", "Weak", "Yes"],
    [14, "Rain", "Mild", "High", "Strong", "No"],
], columns=COLUMNS)
NUMERIC = pd.DataFrame([
    [1, "Sunny", 85, 85, "Weak", "No"],
    [2, "Sunny", 80, 90, "Strong", "No"],
    [3, "Overcast", 83, 78, "Weak", "Yes"],
    [4, "Rain", 70, 96, "Weak", "Yes"],
    [5, "Rain", 68, 80, "Weak", "Yes"],
    [6, "Rain", 65, 70, "Strong", "No"],
    [7, "Overcast", 64, 65, "Strong", "Yes"],
    [8, "Sunny", 72, 95, "Weak", "No"],
    [9, "Sunny", 69, 70, "Weak", "Yes"],
    [10, "Rain", 75, 80, "Weak", "Yes"],
    [11, "Sunny", 75, 70, "Strong", "Yes"],
    [12, "Overcast", 72, 90, "Strong", "Yes"],
    [13, "Overcast", 81, 75, "Weak", "Yes"],
    [14, "Rain", 71, 80, "Strong", "No"],
], columns=COLUMNS)
FEATURES = ["Outlook", "Temperature", "Humidity", "Wind"]   # Day excluded


def close(a, b, tol=5e-5):
    return abs(a - b) < tol


def c45_tests(data):
    """C4.5 candidate tests on a node holding `data`, keyed by attribute."""
    data = data.copy()
    data[c45.W] = 1.0
    numeric = {f for f in FEATURES if c45.is_numeric(data[f])}
    domains = {f: sorted(NUMERIC[f].unique().tolist(), key=str) for f in FEATURES if f not in numeric}
    tests = c45.candidate_tests(data, "Play", FEATURES, domains, numeric, min_cases=1)
    return {t["feature"]: t for t in tests}, c45.choose_test(tests)


# --- measures and edge cases

def test_entropy_edge_cases():
    assert id3.entropy(pd.Series(["Yes"] * 4)) == 0.0          # pure node: exactly 0
    assert id3.entropy(pd.Series([], dtype=object)) == 0.0     # empty set
    assert c45.entropy([4, 0]) == 0.0 and c45.entropy([]) == 0.0
    assert close(id3.entropy(CATEGORICAL["Play"]), 0.9403)
    assert close(c45.entropy([9, 5]), 0.9403)


# --- ID3 example of the article ---------------------------------------------

def test_id3_gains():
    d = CATEGORICAL
    expected = {"Outlook": 0.2467, "Humidity": 0.1518, "Wind": 0.0481, "Temperature": 0.0292}
    for f, g in expected.items():
        assert close(id3.information_gain(d, f, "Play"), g, 1e-4), f
    sunny, rain = d[d.Outlook == "Sunny"], d[d.Outlook == "Rain"]
    for f, g in {"Humidity": 0.9710, "Temperature": 0.5710, "Wind": 0.0200}.items():
        assert close(id3.information_gain(sunny, f, "Play"), g, 1e-4), f
    for f, g in {"Wind": 0.9710, "Temperature": 0.0200, "Humidity": 0.0200}.items():
        assert close(id3.information_gain(rain, f, "Play"), g, 1e-4), f


def test_id3_tree():
    tree = id3.fit(CATEGORICAL, "Play", FEATURES)
    assert tree["feature"] == "Outlook"
    assert tree["branches"]["Sunny"]["feature"] == "Humidity"
    assert tree["branches"]["Rain"]["feature"] == "Wind"
    assert tree["branches"]["Overcast"]["label"] == "Yes"
    assert viz.tree_stats(tree) == {"internal": 3, "leaves": 5, "depth": 2}
    assert id3.predict(tree, CATEGORICAL) == CATEGORICAL["Play"].tolist()


# --- C4.5 example of the article (Release 7) --------------------------------

def test_c45_root_table():
    tests, chosen = c45_tests(NUMERIC)
    expected = {   # feature: (threshold, gain, split info, gain ratio) - article table
        "Outlook": (None, 0.2467, 1.5774, 0.1564),
        "Temperature": (84.0, 0.1134, 0.3712, 0.3055),
        "Humidity": (67.5, 0.0477, 0.3712, 0.1285),   # tie with 95.5 -> smaller
        "Wind": (None, 0.0481, 0.9852, 0.0488),
    }
    for f, (t, g, si, gr) in expected.items():
        test = tests[f]
        assert test["threshold"] == t, f
        assert close(test["gain"], g, 1e-4) and close(test["split_info"], si, 1e-4)
        assert close(test["gain_ratio"], gr, 1e-4), f
    assert close(np.mean([t["gain"] for t in tests.values()]), 0.1140, 1e-4)
    assert chosen["feature"] == "Outlook"     # Temperature has the best ratio but gain < average


def test_c45_sunny_and_rain_nodes():
    tests, chosen = c45_tests(NUMERIC[NUMERIC.Outlook == "Sunny"])
    assert chosen["feature"] == "Humidity" and chosen["threshold"] == 77.5
    assert close(chosen["gain_ratio"], 1.0)
    assert tests["Temperature"]["threshold"] == 70.5
    assert close(tests["Temperature"]["gain"], 0.3219, 1e-4)
    assert close(tests["Temperature"]["gain_ratio"], 0.4459, 1e-4)
    assert close(tests["Wind"]["gain"], 0.0200, 1e-4) and close(tests["Wind"]["gain_ratio"], 0.0206, 1e-4)
    gains = [t["gain"] for f, t in tests.items() if f != "Outlook"]
    assert close(np.mean(gains), 0.4376, 1e-4)

    tests, chosen = c45_tests(NUMERIC[NUMERIC.Outlook == "Rain"])
    assert chosen["feature"] == "Wind" and close(chosen["gain_ratio"], 1.0)
    assert tests["Temperature"]["threshold"] == 66.5 and tests["Humidity"]["threshold"] == 75.0
    for f in ("Temperature", "Humidity"):
        assert close(tests[f]["gain"], 0.3219, 1e-4) and close(tests[f]["gain_ratio"], 0.4459, 1e-4)
    gains = [t["gain"] for f, t in tests.items() if f != "Outlook"]
    assert close(np.mean(gains), 0.5383, 1e-4)


def test_c45_tree_and_pruning():
    raw, pruned = c45.fit(NUMERIC, "Play", FEATURES)
    assert raw["feature"] == "Outlook"
    sunny = raw["branches"]["Sunny"]
    assert sunny["feature"] == "Humidity" and sunny["threshold"] == 77.5
    assert raw["branches"]["Rain"]["feature"] == "Wind"
    assert viz.tree_stats(raw) == {"internal": 3, "leaves": 5, "depth": 2}
    assert c45.predict(raw, NUMERIC) == NUMERIC["Play"].tolist()
    assert viz.tree_to_text(pruned) == viz.tree_to_text(raw)    # pruning keeps this tree


def test_pruning_does_not_modify_input():
    rng = np.random.default_rng(0)
    noisy = NUMERIC.copy()
    noisy["Play"] = rng.permutation(noisy["Play"].values)
    data = noisy.copy()
    data[c45.W] = 1.0
    raw, _ = c45.fit(noisy, "Play", FEATURES)
    before = copy.deepcopy(raw)
    pruned = c45.prune(raw, data, "Play")
    assert raw == before
    assert viz.tree_stats(pruned)["leaves"] <= viz.tree_stats(raw)["leaves"]


def test_pruning_whole_subtree():
    """A node whose children are not all leaves can still become a leaf."""
    df = pd.DataFrame({"a": list("yyyxxxxxxyyy"),
                       "b": [2, 2, 3, 2, 2, 2, 2, 3, 1, 3, 2, 0],
                       "y": list("npppnppppppp")})
    raw, pruned = c45.fit(df, "y", ["a", "b"])
    assert viz.depth(raw) == 3                       # root's children include a subtree
    assert pruned["type"] == "leaf" and pruned["label"] == "p"


def test_pessimistic_errors():
    """U_CF(0, n): probability of 0 errors in n cases at error rate U equals CF."""
    for n in (1, 2, 6, 14):
        u = c45.pessimistic_errors(n, 0) / n
        assert close((1 - u) ** n, 0.25, 1e-9)
    assert close(c45.pessimistic_errors(1, 0), 0.75)             # a single-case leaf
    # more errors or a lower CF -> higher estimate; fractional values are allowed
    assert c45.pessimistic_errors(14, 5) > c45.pessimistic_errors(14, 4) > 4
    assert c45.pessimistic_errors(14, 5, cf=0.1) > c45.pessimistic_errors(14, 5, cf=0.25)
    assert 0.75 < c45.pessimistic_errors(2.4, 0.4) < c45.pessimistic_errors(2.4, 1)


# --- missing values ----------------------------------------------------------

def test_missing_value_quinlan_1993():
    """Quinlan (1993): Outlook of one case unknown -> gain 0.199, split info 1.809."""
    data = NUMERIC.copy()
    data.loc[data.Day == 12, "Outlook"] = np.nan
    tests, _ = c45_tests(data)
    assert close(tests["Outlook"]["gain"], 0.199, 5e-4)
    assert close(tests["Outlook"]["split_info"], 1.809, 5e-4)

    # Release 7 behaviour: the lower Outlook gain lowers the average gain to ~0.102,
    # so "Temperature <= 84" (gain 0.1134, best gain ratio) becomes eligible and wins.
    raw, _ = c45.fit(data, "Play", FEATURES)
    assert raw["feature"] == "Temperature" and raw["threshold"] == 84.0

    # Fractional weights: without Temperature the root is Outlook and the unknown
    # case is split 5/13, 3/13, 5/13 between Sunny, Overcast and Rain.
    raw, _ = c45.fit(data, "Play", ["Outlook", "Humidity", "Wind"])
    assert raw["feature"] == "Outlook"
    overcast = raw["branches"]["Overcast"]
    assert close(sum(overcast["counts"].values()), 3 + 3 / 13, 1e-4)   # 3 known + 3/13 of the unknown case
    prediction = c45.predict(raw, pd.DataFrame([{"Outlook": np.nan, "Temperature": 70,
                                                 "Humidity": 70, "Wind": "Weak"}]))
    assert prediction == ["Yes"]


def test_id3_rejects_missing_values():
    data = CATEGORICAL.copy()
    data.loc[0, "Outlook"] = np.nan
    try:
        id3.fit(data, "Play", FEATURES)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_identifier_column_is_used_when_not_excluded():
    tree = id3.fit(CATEGORICAL, "Play", ["Day", *FEATURES])
    assert tree["feature"] == "Day"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, func in tests:
        try:
            func()
            print(f"PASS  {name}")
        except AssertionError as err:
            failed += 1
            print(f"FAIL  {name}  {err}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)