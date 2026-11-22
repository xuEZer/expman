from expman.webui.result_layout import layout_results


def columns(*names):
    return [{"id": name, "label": name} for name in names]


def test_controlled_contrasts_put_metrics_next_to_their_required_axes():
    records = [
        {"x": x, "z": z, "y": z * 5, "product": x * z, "bias": 4}
        for x in (2, 1)
        for z in (1, 2)
    ]
    table = layout_results(columns("x", "z"), columns("y", "product", "bias"), records)
    fields = {c["id"]: c for c in table["columns"]}

    assert list(fields) == ["bias", "z", "y", "x", "product"]
    assert fields["y"]["depends_on"] == ["z"]
    assert fields["y"]["undetermined"] == []
    assert fields["product"]["depends_on"] == ["x", "z"]
    assert fields["bias"]["depends_on"] == fields["bias"]["undetermined"] == []
    # Preserve the YAML's level order (2, 1 for x) within each z group.
    assert table["rows"] == [records[i] for i in (0, 2, 1, 3)]


def test_confounding_and_missing_metrics_do_not_imply_independence():
    records = [
        {"x": 0, "z": 0, "y": 1, "sparse": 10},
        {"x": 1, "z": 0, "y": 1},
        {"x": 0, "z": 1, "y": 2},
    ]
    table = layout_results(columns("x", "z"), columns("y", "sparse"), records)
    fields = {c["id"]: c for c in table["columns"]}
    assert fields["y"]["depends_on"] == ["z"]
    assert fields["y"]["undetermined"] == ["x"]
    assert fields["sparse"]["depends_on"] == []
    assert fields["sparse"]["undetermined"] == ["x", "z"]
    assert len(table["rows"]) == 3

    coupled = [{"x": 1, "z": 1, "y": 12}, {"x": 2, "z": 2, "y": 30}]
    table = layout_results(columns("x", "z"), columns("y"), coupled)
    metric = table["columns"][-1]
    assert metric["id"] == "y"
    assert metric["depends_on"] == []
    assert metric["undetermined"] == ["x", "z"]


def test_settings_keep_missing_null_types_and_object_identity_distinct():
    settings = [
        {},
        {"v": None},
        {"v": True},
        {"v": 1},
        {"v": [1]},
        {"v": {"a": 1, "b": 2}},
    ]
    records = [{**setting, "y": k} for k, setting in enumerate(settings)]
    records.append({"v": {"b": 2, "a": 1}, "y": 5})

    table = layout_results(columns("v"), columns("y"), records)
    assert [c["id"] for c in table["columns"]] == ["v", "y"]
    assert table["columns"][1]["depends_on"] == ["v"]
    assert table["rows"] == records

    # Reordered keys denote the same object. Contradictory repeats with no
    # measured comparison at a different level do not establish an effect.
    records = [{"v": {"b": 2, "a": 1}, "y": 2}, {"v": {"a": 1, "b": 2}, "y": 3}, {}]
    metric = layout_results(columns("v"), columns("y"), records)["columns"][-1]
    assert metric["depends_on"] == []
    assert metric["undetermined"] == ["v"]


def test_missing_metrics_stay_missing_and_rows_survive_without_any_columns():
    records = [{"x": 1, "y": 0}, {"x": 2}]
    table = layout_results(columns("x"), columns("y"), records)
    assert table["rows"] == records
    assert table["columns"][-1]["undetermined"] == ["x"]
    assert layout_results([], [], [{}]) == {"columns": [], "rows": [{}]}
