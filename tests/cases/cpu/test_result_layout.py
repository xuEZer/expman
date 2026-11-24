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


def test_unset_and_null_are_distinct_settings_even_with_an_unmeasured_run():
    records = [{"y": 0}, {"x": None, "y": 1}, {"x": 2}]
    table = layout_results(columns("x"), columns("y"), records)
    assert table["rows"] == records
    assert table["columns"][-1]["depends_on"] == ["x"]
