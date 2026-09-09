"""Pure VLN evaluation runtime: episodes, scoring, motion, policy wire client.

Submodules are imported explicitly (``insight_bench.vln_runtime.scoring`` etc.); this
package init stays import-light so ``import insight_bench`` never pulls numpy.
Using this subpackage requires the ``vln`` extra (numpy, pillow, pyyaml,
requests).
"""
