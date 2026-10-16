# Working on ICASSP IPVG

- Read README.md, docs/STATUS.md and docs/AUTOMATION.md before experiments.
- Track 1 is the first delivery target; Track 2 is a separate research branch.
- Never describe synthetic smoke videos as model quality results. Label local proxy metrics; no official scoring implementation has been confirmed.
- Record code commit, model revision, dataset/split hashes, seed and resource use for each experiment. Keep evaluator changes separate from model changes.
- Keep train/dev/holdout separated by connected identity and source-video groups. Do not tune on holdout or the competition test set.
- Propose one falsifiable change at a time, compare at the same sampling/GPU budget, retain failures and use a new run_id for changed or failed experiments.
- Every launched experiment must have a timeout and a declared GPU-hour cap. Campaigns are finite. No unlimited retries.
- Raw data, generated restricted content, model weights, checkpoints, tokens, identifying logs and submission archives stay outside Git. Use ignored directories and environment credentials.
- Do not upload restricted images/video to external model services without compatible data terms. An external coding controller receives code and aggregated/redacted results.
- Honor dataset access conditions and separate code, model and face-encoder licenses. Do not sign agreements or claim registration on behalf of researchers.
- Test CPU orchestration with pytest and scripts/smoke.py. GPU integrations need a real pilot before updating their verification status.
- Do not silently change this repository's scientific hypotheses, official-rule status or successful-test claims. Update docs/STATUS.md with evidence.
