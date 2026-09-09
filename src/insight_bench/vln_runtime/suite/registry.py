"""Static registry of the benchmark task families shipped by this package.

The mapping is a literal table of imported factories: no entry-point discovery,
no module-path lookup, no ``importlib`` on caller-supplied names. Adding a
family is a code change in this distribution, which is what keeps the official
CLI able to execute only code that shipped with it.
"""

from __future__ import annotations

from collections.abc import Callable

from insight_bench.vln_runtime.suite.config import BenchmarkTaskConfig
from insight_bench.vln_runtime.suite.insight_bench.config import (
    default_task_config as insight_bench_config,
)

TaskConfigFactory = Callable[[], BenchmarkTaskConfig]

TASK_CONFIGS: dict[str, TaskConfigFactory] = {
    "insight_bench": insight_bench_config,
}


def available_task_names() -> tuple[str, ...]:
    return tuple(sorted(TASK_CONFIGS))


def get_task_config(name: str) -> BenchmarkTaskConfig:
    try:
        factory = TASK_CONFIGS[name]
    except KeyError as exc:
        available = ", ".join(available_task_names())
        raise ValueError(f"unknown benchmark task {name!r}; available tasks: {available}") from exc
    return factory()
