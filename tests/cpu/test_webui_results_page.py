import json

from nodejs_wheel import node

from expman.webui.server import STATIC_ROOT


def evaluate(expression, payload=None):
    harness = """
    const fs = require("node:fs");
    const vm = require("node:vm");
    const context = vm.createContext({payload: JSON.parse(fs.readFileSync(0, "utf8"))});
    vm.runInContext(fs.readFileSync(process.argv[1], "utf8"), context);
    Promise.resolve(vm.runInContext(process.argv[2], context))
      .then(value => process.stdout.write(JSON.stringify(value)));
    """
    result = node(
        ["-e", harness, str(STATIC_ROOT / "results.js"), expression],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        return_completed_process=True,
    )
    return json.loads(result.stdout)


def test_each_column_merges_its_own_consecutive_values():
    rows = [
        {"x": 1},
        {"x": 2},
        {"x": 2, "y": 0},
        {"x": 1, "y": 0},
    ]
    spans = evaluate(
        """["x", "y"].map(id => Array.from(mergedCells(payload, id),
              c => [valueText(c.value, id === "y"), c.span]))""",
        rows,
    )
    assert spans == [[["1", 1], ["2", 2], ["1", 1]], [["", 2], ["0", 2]]]


def test_missing_config_keys_and_explicit_values_stay_distinct():
    values = evaluate(
        'Array.from(mergedCells(payload, "x"), c => [valueText(c.value), c.span])',
        [{}, {}, {"x": None}, {"x": "null"}, {"x": False}, {"x": 0}, {"x": ""}],
    )
    assert values == [
        ["–", 2],
        ["null", 1],
        ['"null"', 1],
        ["false", 1],
        ["0", 1],
        ['""', 1],
    ]


def test_live_refresh_preserves_the_table_and_retries_the_final_read():
    result = evaluate("""
    (async () => {
      let draws = 0;
      const requests = [];
      let rows = [{x: 1}, {x: 2}];
      const panel = {append(child) { this.firstElementChild = child; }};
      globalThis.document = {
        createElement: () => ({replaceChildren(child) { this.table = child; }})
      };
      resultTable = (columns, rows) => { draws++; return rows; };
      const ok = etag => ({
        ok: true, headers: {get: key => key === "ETag" ? etag : null},
        json: () => ({columns: [], rows})
      });
      const unchangedResponse = {status: 304, ok: false};
      let respond = () => ok('"first"');
      globalThis.fetch = (url, options) => {
        requests.push([options.cache, options.headers["If-None-Match"]]);
        return respond();
      };

      await refreshResultTable(panel, 0, true);
      const viewport = panel.firstElementChild;
      const initial = viewport.table;
      respond = () => unchangedResponse;
      await refreshResultTable(panel, 0, true);
      const unchanged = initial === viewport.table;

      let finish;
      respond = () => new Promise(resolve => { finish = resolve; });
      const live = refreshResultTable(panel, 0, true);
      await refreshResultTable(panel, 0, true);
      await refreshResultTable(panel, 0, false);
      const callsDuringRequest = requests.length;
      respond = () => ({...ok('"broken"'), json: () => JSON.parse("{")});
      finish(unchangedResponse);
      await live;
      const afterFailedFinal = [requests.length, draws, viewport.table === initial];

      respond = () => ok('"final"');
      rows = [rows[0], {...rows[1], y: 0}];
      await refreshResultTable(panel, 0, false);
      await refreshResultTable(panel, 0, false);
      const afterRetry = [requests.length, draws, panel.firstElementChild === viewport];
      respond = () => unchangedResponse;
      await refreshResultTable(panel, 0, false, true);
      await refreshResultTable(panel, 0, false);
      return {initial, unchanged, callsDuringRequest, afterFailedFinal, afterRetry,
              final: viewport.table, requests, draws};
    })()
    """)
    assert result == {
        "initial": [{"x": 1}, {"x": 2}],
        "unchanged": True,
        "callsDuringRequest": 3,
        "afterFailedFinal": [4, 1, True],
        "afterRetry": [5, 2, True],
        "final": [{"x": 1}, {"x": 2, "y": 0}],
        "requests": [
            ["no-store", tag]
            for tag in [None, '"first"', '"first"', '"first"', '"first"', '"final"']
        ],
        "draws": 2,
    }
