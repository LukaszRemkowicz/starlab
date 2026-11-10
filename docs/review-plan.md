# Review implementation plan

The user authorized implementation after the complete baseline review, in the existing isolated worktree. Preserve the approved application behavior and use separate commits for every numbered item that changes code. Do not rewrite the reviewed publication commit again.

- [x] Complete review and record baseline, asset explanations and preservation constraints in code-review.md.
- [x] Item2: validate/document analysis lookup data and read-only ownership.22 focused tests pass.
- [x] Item3: remove genuine duplication/dead scanning work; preserve independent numerical backends.348 full-suite cases pass after shared ray/tile/image/history helpers and TIFF-drop regressions.
- [x] Item4: replace categorical branch dispatch with match where clearer. Job completion/display modes and effect directions/support dispatch now use match; numerical guards remain if.
- [x] Item5: separate/validate universal transform loading and metadata meaning. Archive/schema checks precede reconstruction; stored values/operation order remain unchanged.
- [x] Item6: decompose responsibility hotspots, correct state/lifecycle defects, add complexity gates and focused regressions.368 full-suite cases pass; real GPU/CPU correction errors remain below1.5e-7. Two documented numerical state-machine exceptions remain.
- [x] Item7: add accurate docstrings, boundary/local types and enforced annotation/type checks.
- [x] Item8: name secondary-pattern units and derive Python ray counts from its data.
- [x] Item9: document executable shader stages and maintain CPU/GPU numerical distinction; shader coverage belongs to item11.
- [x] Item10: isolate state/resource ownership in focused objects, without wrapping numerical kernels in class hierarchies.
- [x] Item11: strengthen independent oracles, one-to-one matching, event synchronization and actual GPU checks.
- [ ] Item12: verify excluded experiment files remain absent and no source/test import depends on them.
- [ ] Finish with full tests, static checks, wheel/CLI validation, real GPU verification, review of changes and a commit-to-item report.

Relevant modules and specific findings are listed in code-review.md. For a code-changing item, first add the smallest regression that exposes the defect, run it red, implement, run focused checks, inspect the diff and commit only that item. Pure behavior-preserving extraction uses existing independent regression fixtures and backend comparisons.
