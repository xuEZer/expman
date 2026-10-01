import io

from expman import Batch, Pipeline, Stage, Status
from expman.progress import BatchProgress
from expman.scheduling import GpuScheduler


class First(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {}

    def process(self, data, ctx):
        return data


class Second(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"item": True}

    def process(self, data, ctx):
        return ctx.cfg["item"]


def test_eta_counts_only_unmaterialized_stage_parameter_groups(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("device: [0]\nseed: 0\nitem: 0\ngrid:\n  item: [0, 1, 2]\n")
    batch = Batch(Pipeline([First, Second]), cfg, output_dir=tmp_path / "batch")
    first = batch.experiments[0]
    batch._queue.remove(first.run_id)
    batch._stage_history.extend(
        [
            {
                "run_id": first.run_id,
                "stage": 0,
                "status": Status.SUCCEEDED.value,
                "reused": False,
                "stage_duration_seconds": 10.0,
                "dependencies": [],
            },
            {
                "run_id": first.run_id,
                "stage": 1,
                "status": Status.SUCCEEDED.value,
                "reused": False,
                "stage_duration_seconds": 20.0,
                "dependencies": [(("item",), "sample")],
            },
        ]
    )
    scheduler = GpuScheduler(batch)

    estimate = batch.estimate()

    # Stage 0 has one config-independent cache group, already completed.
    # Stage 1 still needs one representative for item=1 and item=2.
    assert estimate.lower_seconds == 40.0
    assert estimate.upper_seconds == 40.0
    # One measured sample per Stage leaves a degenerate interval; the two
    # representatives then share the single available slot.
    assert str(estimate) == "00:00:00～00:00:01"
    assert estimate.completed_samples == 2
    assert estimate.remaining_experiments == 2
    assert batch._scheduler is scheduler
    progress = BatchProgress(batch, enabled=True, interval=1)
    progress.stream = io.StringIO()
    progress._render()
    assert "Stage0:1/1 Stage1:1/3" in progress.stream.getvalue()
