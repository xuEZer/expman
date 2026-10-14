"""CLI demo pipeline: expman run examples.cli_pipeline:train examples/cli.yaml."""

from expman import Pipeline, Stage


class LoadValues(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"values": True}

    def process(self, data, ctx):
        return list(ctx.cfg["values"])


class Accumulate(Stage):
    @classmethod
    def config_dependencies(cls, cfg):
        return {"interrupt_once": True}

    def process(self, data, ctx):
        total = ctx.checkpoint_state
        if total is None:
            total = 0
        for index in ctx.resume_range(0, len(data)):
            total += data[index]
            ctx.checkpoint(index + 1, total)
            if ctx.cfg["interrupt_once"] and ctx.attempt == 1 and index == 1:
                raise KeyboardInterrupt()
        return total


train = Pipeline([LoadValues, Accumulate])
