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
        return {"interrupt_once": True, "max_iter": True}

    def init(self, ctx):
        self.total = 0

    def loop(self, data, ctx, index, max_iter=Stage.cfg("max_iter")):
        self.total += data[index]
        if ctx.cfg["interrupt_once"] and ctx.attempt == 1 and index == 1:
            raise KeyboardInterrupt()
        return self.total


train = Pipeline([LoadValues, Accumulate])
