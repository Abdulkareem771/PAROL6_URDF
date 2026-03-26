# PAROL6 Git Merge, Rebase, and IDE Guide

## Purpose

This guide explains the practical git commands we use in the PAROL6 workspace when:

- a teammate pushed new commits to the same branch
- your local branch also has commits
- the IDE shows branch divergence and merge prompts
- generated files or permissions block a merge

It is written for the exact kind of situations that happened on `GUI-vision-pipeline`.

---

## The Problem We Hit

A common situation in this repo is:

1. You are working on a branch such as `GUI-vision-pipeline`.
2. You make one or more local commits.
3. A teammate also pushes commits to `origin/GUI-vision-pipeline`.
4. Your IDE starts showing messages like:
   - "branch is ahead and behind"
   - "pull / sync / merge / rebase"
   - conflict warnings

At that point, blindly clicking buttons in the IDE can be risky because:

- the IDE may choose `merge` when you expected `rebase`
- the IDE may auto-stage files you did not mean to include
- generated files can interfere with checkout or merge
- permission problems can break the operation halfway through

In our case, one merge failed with:

```bash
error: unable to unlink old 'parol6_firmware/.pio/libdeps/debug_stage3/integrity.dat': Permission denied
error: unable to unlink old 'parol6_firmware/.pio/libdeps/debug_stage4/integrity.dat': Permission denied
error: unable to unlink old 'parol6_firmware/.pio/libdeps/teensy41_j6_test/integrity.dat': Permission denied
```

That happened because:

- `test_all` contained tracked files under `parol6_firmware/.pio/...`
- those files are generated artifacts and should not have been part of normal branch integration
- the local copies were owned by `root`, so git could not replace them

---

## What "Ahead / Behind" Means

When the IDE or `git status -sb` says:

```bash
## GUI-vision-pipeline...origin/GUI-vision-pipeline [ahead 1, behind 2]
```

it means:

- `ahead 1`: you have 1 local commit not on the remote branch
- `behind 2`: the remote branch has 2 commits you do not have locally

This is not automatically a conflict. It only means the histories diverged.

You usually have two safe choices:

- `merge`: preserve both histories and create a merge commit
- `rebase`: replay your local commits on top of the updated remote branch

---

## What Usually Happens Inside the IDE

Most IDEs are wrapping normal git commands, even if they do not show them directly.

Typical IDE buttons map roughly like this:

- **Pull**:
  - often does `git fetch` + `git merge`
  - sometimes does `git pull --rebase`
- **Sync Changes**:
  - usually fetches, then pulls, then pushes
- **Rebase Current Branch**:
  - usually does `git fetch` then `git rebase origin/<branch>`
- **Merge Branch**:
  - usually does `git merge <branch>`
- **Abort Merge / Abort Rebase**:
  - same idea as `git merge --abort` or `git rebase --abort`

The problem is that the IDE may not show:

- exactly which files are entering the merge
- whether a merge commit will be created
- whether generated files are being staged
- whether the operation is stuck because of permissions

For this repo, it is often safer to do the critical part in the terminal first, then let the IDE refresh afterward.

---

## Safe Merge Workflow

Use this when:

- you want to keep branch history clear
- you are okay with a merge commit
- you want a safe preview before finalizing

### 1. Check status

```bash
cd /home/kareem/Desktop/PAROL6_URDF
git status -sb
```

### 2. Fetch the latest remote state

```bash
git fetch origin GUI-vision-pipeline
```

### 3. Create a backup branch before integrating

```bash
git branch backup_gui_vision_pipeline_premerge_$(date +%Y%m%d)
```

This gives you an easy rollback point.

### 4. Inspect what is different

```bash
git log --oneline --left-right GUI-vision-pipeline...origin/GUI-vision-pipeline
git diff --name-status GUI-vision-pipeline..origin/GUI-vision-pipeline
```

This shows:

- which commits are only local
- which commits are only remote
- which files the remote side would bring in

### 5. Preview the merge without committing it yet

```bash
git merge --no-ff --no-commit origin/GUI-vision-pipeline
```

This is one of the safest commands in this situation because:

- git performs the merge
- you can inspect the result
- nothing is finalized until you commit

### 6. Inspect the staged merge result

```bash
git status --short
git diff --cached --name-status
git diff --cached -- parol6_vision/scripts/vision_pipeline_gui.py
```

If the staged result looks correct, finish the merge:

```bash
git commit -m "Merge remote-tracking branch 'origin/GUI-vision-pipeline' into GUI-vision-pipeline"
```

If the preview looks wrong, stop immediately:

```bash
git merge --abort
```

### 7. Verify after merge

```bash
git status -sb
git log --oneline --decorate -4
python3 -m py_compile parol6_vision/scripts/vision_pipeline_gui.py
```

---

## Safe Rebase Workflow

Use this when:

- you want a linear history
- your local commits are clean and small
- you are comfortable resolving conflicts commit-by-commit

### 1. Check status and fetch

```bash
cd /home/kareem/Desktop/PAROL6_URDF
git status -sb
git fetch origin GUI-vision-pipeline
git branch backup_gui_vision_pipeline_prerebase_$(date +%Y%m%d)
```

### 2. Inspect divergence

```bash
git log --oneline --left-right GUI-vision-pipeline...origin/GUI-vision-pipeline
```

### 3. Rebase local commits on top of the remote branch

```bash
git rebase origin/GUI-vision-pipeline
```

If there is a conflict:

```bash
git status
```

Fix the file, then:

```bash
git add <file>
git rebase --continue
```

If you decide the rebase is getting messy:

```bash
git rebase --abort
```

### 4. Verify after rebase

```bash
git status -sb
git log --oneline --decorate -6
python3 -m py_compile parol6_vision/scripts/vision_pipeline_gui.py
```

### When to prefer `rebase`

Prefer rebase when:

- the remote branch only moved a little
- your local commits are yours alone
- you plan to push a clean branch history

### When to prefer `merge`

Prefer merge when:

- the branch has teammate commits mixed in
- you want to preserve the real integration history
- you want the safest and most inspectable preview path

For shared team branches in this repo, `merge --no-commit` is often the least risky option.

---

## Recommended Commands for the IDE Situation

If the IDE says the branch diverged and you want the safest manual process:

```bash
git status -sb
git fetch origin GUI-vision-pipeline
git branch backup_gui_vision_pipeline_premerge_$(date +%Y%m%d)
git log --oneline --left-right GUI-vision-pipeline...origin/GUI-vision-pipeline
git diff --name-status GUI-vision-pipeline..origin/GUI-vision-pipeline
git merge --no-ff --no-commit origin/GUI-vision-pipeline
git diff --cached --name-status
git commit -m "Merge remote-tracking branch 'origin/GUI-vision-pipeline' into GUI-vision-pipeline"
```

If you want the linear-history version:

```bash
git status -sb
git fetch origin GUI-vision-pipeline
git branch backup_gui_vision_pipeline_prerebase_$(date +%Y%m%d)
git rebase origin/GUI-vision-pipeline
```

---

## Permission-Denied Merge Failure

### Symptom

You see errors like:

```bash
error: unable to unlink old 'parol6_firmware/.pio/libdeps/debug_stage3/integrity.dat': Permission denied
```

### Meaning

Git is trying to update or remove a file in your working tree, but your current user does not own that file.

In this repo, that usually happens when:

- PlatformIO-generated files were created as `root`
- a branch accidentally tracked generated artifacts
- the IDE or terminal tries to switch or merge through those paths

### Diagnose it

```bash
ls -l parol6_firmware/.pio/libdeps/debug_stage3/integrity.dat
ls -ld parol6_firmware/.pio/libdeps/debug_stage3
id -un
id -gn
git ls-files -- parol6_firmware/.pio/libdeps/debug_stage3/integrity.dat
git check-ignore -v parol6_firmware/.pio/libdeps/debug_stage3/integrity.dat
```

### Fix ownership

If the files should remain:

```bash
sudo chown -R $USER:$USER parol6_firmware/.pio
```

### Or delete generated cache directories

If they are only disposable generated artifacts:

```bash
sudo rm -rf parol6_firmware/.pio/libdeps/debug_stage3
sudo rm -rf parol6_firmware/.pio/libdeps/debug_stage4
sudo rm -rf parol6_firmware/.pio/libdeps/teensy41_j6_test
```

Then retry the merge.

### Important note

Do not randomly delete tracked source files. Only remove generated cache paths that you understand.

---

## What We Did in the `test_all` Merge Incident

`test_all` could not be merged directly because it tried to bring in:

- tracked `.pio` generated artifacts
- a stray `src/kinect2_ros2` gitlink

Safe solution:

1. Create a temporary worktree.
2. Merge `test_all` there with `--no-commit`.
3. Inspect the staged merge result.
4. Remove the bad generated paths from the merge result.
5. Finalize the clean merge commit.
6. Fast-forward the real branch to that clean merge commit.

That pattern is useful when the main checkout is polluted by permissions or generated files.

---

## Useful Git Commands for This Repo

### Status and history

```bash
git status -sb
git branch -vv
git log --oneline --decorate -10
git log --oneline --left-right GUI-vision-pipeline...origin/GUI-vision-pipeline
```

### Compare local vs remote

```bash
git diff --name-status GUI-vision-pipeline..origin/GUI-vision-pipeline
git diff --name-status origin/GUI-vision-pipeline..GUI-vision-pipeline
```

### Safe preview merge

```bash
git merge --no-ff --no-commit origin/GUI-vision-pipeline
git diff --cached --name-status
git merge --abort
```

### Rebase controls

```bash
git rebase origin/GUI-vision-pipeline
git rebase --continue
git rebase --abort
```

### Recovery and backup

```bash
git branch backup_before_merge_$(date +%Y%m%d)
git reflog
```

### Verify changed files in a branch

```bash
git diff --name-status current_branch..other_branch
git show --stat <commit>
```

---

## Practical Team Rules

- Do not merge `main` into your branch unless that is explicitly the goal.
- Create a backup branch before risky integration.
- Inspect staged merge results before committing.
- Do not trust IDE sync buttons blindly on shared branches.
- Treat generated files under `.pio`, `build`, `install`, and `log` with suspicion.
- If a teammate branch contains generated files, do not normalize that by merging blindly.
- Prefer terminal-based merge or rebase for the critical step, then return to the IDE.

---

## Quick Command Cheat Sheet

### Safe merge

```bash
git fetch origin GUI-vision-pipeline
git branch backup_gui_vision_pipeline_premerge_$(date +%Y%m%d)
git merge --no-ff --no-commit origin/GUI-vision-pipeline
git diff --cached --name-status
git commit -m "Merge remote-tracking branch 'origin/GUI-vision-pipeline' into GUI-vision-pipeline"
```

### Safe rebase

```bash
git fetch origin GUI-vision-pipeline
git branch backup_gui_vision_pipeline_prerebase_$(date +%Y%m%d)
git rebase origin/GUI-vision-pipeline
```

### Abort if needed

```bash
git merge --abort
git rebase --abort
```

### Fix root-owned generated files

```bash
sudo chown -R $USER:$USER parol6_firmware/.pio
```

---

## Final Advice

If the IDE says "pull", "sync", or "rebase" and you are not fully sure what it will do, stop and run:

```bash
git status -sb
git fetch origin <branch>
git log --oneline --left-right <branch>...origin/<branch>
```

Those three commands usually tell you enough to choose safely between merge, rebase, or aborting the operation.
