"""Pure helper functions for the scheduler packing plan."""

PLAN_CANDIDATES_PER_STAGE = 32
PLAN_BEAM_WIDTH = 128


def _interleave(groups):
    """Round-robin candidates so one Stage cannot consume beam search first."""
    items = []
    longest = max((len(group) for group in groups), default=0)
    for index in range(longest):
        for group in groups:
            if index < len(group):
                items.append(group[index])
    return items


def _plan_score(plan, host_capacity, gpu_capacities):
    """Prefer plans that jointly leave the least normalized resource behind."""
    unused = (plan.host_left_kb / host_capacity) ** 2
    unused += sum(
        (left / capacity) ** 2
        for left, capacity in zip(plan.gpu_left_kb, gpu_capacities, strict=True)
        if capacity > 0
    )
    # Parameter diversity only breaks materially equivalent packing choices.
    return -unused + 1e-6 * plan.novelty
