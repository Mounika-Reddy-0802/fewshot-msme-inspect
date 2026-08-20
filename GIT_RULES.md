# Git & Version Control Rules

These rules apply to the entire project unless explicitly changed.

Project: **Open-Vocabulary Few-Shot Defect Detection for MSMEs** (2-member capstone, 8 weeks)
Repo: `fewshot-msme-inspect`

## One-time setup

```bash
git config --global user.name "Mounika-Reddy-0802"
git config --global user.email "sherymounikareddy.2006@gmail.com"

# remote (set once; shown for reference)
git remote add origin https://github.com/Mounika-Reddy-0802/fewshot-msme-inspect.git
```

Always push using **my own GitHub repository and credentials**. Never suggest creating
another repository or using another account. Never change the remote unless I explicitly
ask. Assume the remote is already configured.

## Commit on every meaningful change

Every meaningful change must be committed and pushed. A meaningful change includes:
adding a feature, creating a file, deleting a file, refactoring, fixing a bug, updating
project structure, adding documentation, changing configuration, adding or updating an
experiment config, changing the model / prototype memory / adapter code, updating the
evaluation or profiling harness, adding a notebook, updating the demo app, or
implementing any week-plan milestone.

## What never goes into git

- Dataset files (MVTec AD, VisA, NEU-DET, MPDD) — download scripts only
- Feature banks and cached embeddings
- Model checkpoints and ONNX/INT8 artifacts
- Results output files (plots, CSVs) — **except** split manifests, seeds, and
  figure manifests, which MUST be committed
- Secrets, tokens, wandb keys, `.env` files

Keep all of these in `.gitignore`. If any of them get committed by mistake, remove
them in a new commit immediately.

## Workflow (for every completed task)

1. Finish the implementation.
2. Verify the project still runs correctly (script or test actually executes).
3. Review the modified files (`git status`, `git diff`).
4. Create a **small, focused commit** for only the related changes.
5. Push to the current branch on GitHub.
6. Check the push landed: `git log --oneline -3` and confirm a clean tree.
7. Then move on to the next task.

## Commit message rules

Write simple, natural, human commit messages. Keep them short, lowercase, clear, focused on
one change, and under ~50 characters when possible.

Good examples:

```
add project structure
create baseline config
add patchcore validation
add prototype memory
train adapter on mvtec
fix split generator seeding
add onnx export script
add int8 quantization
add profiling harness
fix demo layout
update requirements
update readme
remove data files from repo
freeze accuracy results v1
```

Do **not** use AI-style messages such as: "implement comprehensive…", "refactor
architecture…", "introduce advanced…", "enhance robust…", "complete implementation of…",
"add extensive…", "improve overall…", "optimize pipeline…".

## Two-member rules

- Each member commits their own work under their **own git identity**, so
  contribution history stays honest for department evaluation.
- Both members work on the same branch (`main`) unless we agree on a feature branch.
- Always `git pull` before starting a task and before every push.
- Shared interfaces (manifest format, split registry, feature-bank layout, config
  schema) are never changed unilaterally — agree first, then one member commits it.

## Reviewing the teammate's work

- After the teammate pushes, review their changes against the week plan before
  starting the next task.
- If their work deviates from the plan (unseeded splits, missing metrics, hardcoded
  paths, data files committed, broken loaders), make the **minimal correction**,
  verify it runs, commit with a normal message, and push through my account as usual.
- Corrections are always **new commits on top**. Never rewrite pushed history:
  no force-push, no rebase of pushed commits, no amending the teammate's commits.

## Push rules

- Push to the current branch after every successful commit.
- Use my GitHub account and the already-configured remote.
- After every push, report repo status: branch, last 3 commits, clean/dirty tree.

## Freeze rules

- **Week 4 — architecture freeze:** after the freeze commit, no changes to model
  architecture files; only configs, evaluation, and analysis code may change.
- **Week 7 — results freeze:** tag the repo (`results-freeze`); after this, no new
  experiments — only manuscript, report, release packaging, and demo polish commits.
- **Week 8 — release:** tag the final reproducible release (`v1.0`) with one-command
  scripts per figure and the pinned environment.
