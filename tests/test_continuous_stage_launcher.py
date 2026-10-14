"""What `examples/run_p17_continuous_stage.sh` must do before a 2-hour run.

Three protections were on the hinge probe and missing here, each reproduced
with mock jobs before being fixed:

- every seed job could fail and the script still exited 0, because failures
  went into a printed summary and nowhere else;
- reusing an output directory and then failing every new job made the gate
  report the PREVIOUS run's seed JSONs as this run's result -- silent, and it
  reads as data;
- `plan.json` recorded no commit, so nothing tied a 2-hour result to the code
  that produced it.

The seeder, the search driver and the dependency patches are all mocked. This
is about the launcher's control flow; the real stage needs a cluster.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

STAGE = Path(__file__).resolve().parents[1] / "examples" / "run_p17_continuous_stage.sh"
DEVICES = "0,1,2,3,4,5,6,7"

MOCK = """#!/usr/bin/env bash
# `-`/`-c` are the launcher's own inline python; everything else is a job.
case "$1" in
  -|-c) exec %(python)s "$@" ;;
esac
case "$1" in
  *patch_jopendde*) exit 0 ;;
esac
if [[ "$1" == *p17_continuous_seed.py ]]; then
  out=""; method=""; seed=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --out) out="$2"; shift 2 ;;
      --method) method="$2"; shift 2 ;;
      --seed) seed="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  [[ "$MOCK_SEED" == "fail" ]] && exit 1
  [[ "$MOCK_SEED" == "partial" && "$method" != apgm ]] && exit 1
  # Exits 0 and writes nothing -- the case exit codes cannot catch.
  [[ "$MOCK_SEED" == "silent" ]] && exit 0
  [[ "$MOCK_SEED" == "silent_most" && "$method" != apgm ]] && exit 0
  if [[ "$MOCK_SEED" == "corrupt" ]]; then echo 'not json' > "$out"; exit 0; fi
  # Right filename, wrong contents -- what presence checks cannot catch.
  [[ "$MOCK_SEED" == "mislabelled" ]] && { method=somethingelse; seed=99; }
  # One seed in the truncation regime, the rest comfortably inside the budget:
  # the case a max-based verdict describes wrongly.
  wanted_override=""
  if [[ "$MOCK_SEED" == "mixed" ]]; then
    if [[ "$method" == apgm ]]; then wanted_override=29; else wanted_override=2; fi
  fi
  cat > "$out" <<EOJ
{"method": "$method",
 "seed": $seed,
 "hamming_from_parent": 2,
 "candidate_substitutions_considered": ${wanted_override:-%(wanted)s},
 "edit_budget": 5,
 "seed_sequence": "QVQL$method$seed",
 "edits": [{"margin": 0.4}, {"margin": 0.9}]}
EOJ
  exit 0
fi
if [[ "$1" == *p17_alpha_recovery.py ]]; then
  [[ "$MOCK_SEARCH" == "fail" ]] && exit 1
  exit 0
fi
exit 0
"""


@pytest.fixture
def stage(tmp_path):
    import sys

    def run(*args, seeds="ok", search="ok", wanted=3):
        mock = tmp_path / f"mockpy_{wanted}_{seeds}"
        mock.write_text(MOCK % {"python": sys.executable, "wanted": wanted})
        mock.chmod(0o755)
        env = dict(
            os.environ, PYTHON_BIN=str(mock), MOCK_SEED=seeds, MOCK_SEARCH=search
        )
        return subprocess.run(
            ["bash", str(STAGE), "--devices", DEVICES, "--seed-replicates", "2", *args],
            capture_output=True, text=True, env=env, timeout=600,
        )

    return run


def test_six_good_seed_runs_exit_clean_and_print_a_verdict(stage, tmp_path):
    out = tmp_path / "good"
    done = stage("--stages", "seed", "--output-dir", str(out))
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert "VERDICT" in (out / "seeds" / "gate.txt").read_text()


def test_every_seed_job_failing_does_not_exit_zero(stage, tmp_path):
    done = stage("--stages", "seed", "--output-dir", str(tmp_path / "all_fail"),
                 seeds="fail")
    assert done.returncode != 0, "a run in which every job died reported success"
    assert "FAILED" in done.stderr


def test_a_partial_seed_failure_is_named_and_still_fails(stage, tmp_path):
    out = tmp_path / "partial"
    done = stage("--stages", "seed", "--output-dir", str(out), seeds="partial")
    assert done.returncode != 0
    gate = (out / "seeds" / "gate.txt").read_text()
    assert "INCOMPLETE" in gate, gate
    assert "bindcraft_r0.json" in gate


def test_a_reused_output_directory_is_refused(stage, tmp_path):
    out = tmp_path / "reused"
    (out / "seeds").mkdir(parents=True)
    (out / "seeds" / "apgm_r0.json").write_text('{"method": "apgm"}')
    done = stage("--stages", "seed", "--output-dir", str(out))
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert "not empty" in done.stderr


def test_the_plan_records_the_commit_that_produced_it(stage, tmp_path):
    out = tmp_path / "prov"
    stage("--stages", "seed", "--output-dir", str(out))
    plan = json.loads((out / "plan.json").read_text())
    assert len(plan["git_commit"]) == 40, plan["git_commit"]
    assert isinstance(plan["git_dirty"], bool)
    # The nested OpenDDE checkout is permanently untracked, so a dirty flag
    # computed without --ignore-submodules=all would be the constant True.
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "--ignore-submodules=all"],
        capture_output=True, text=True, cwd=STAGE.parents[1],
    ).stdout.strip()
    assert plan["git_dirty"] is bool(tracked)


def test_a_search_cell_failure_reaches_the_exit_code(stage, tmp_path):
    done = stage("--output-dir", str(tmp_path / "search_fail"), search="fail")
    assert done.returncode != 0
    assert "FAILED" in done.stdout + done.stderr


def test_search_with_no_seeds_on_disk_does_not_look_like_a_clean_run(
    stage, tmp_path
):
    done = stage("--stages", "search", "--output-dir", str(tmp_path / "no_seeds"))
    assert done.returncode != 0, (
        "every seeded cell skipped, the parent control ran alone, and the "
        "script exited 0"
    )
    assert "SKIPPED" in done.stdout


def test_nine_runs_over_eight_devices_is_refused(stage, tmp_path):
    done = stage("--stages", "seed", "--seed-replicates", "3",
                 "--output-dir", str(tmp_path / "nine"))
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert "Refusing to start stage 'seed'" in done.stderr


def test_the_truncation_verdict_tracks_what_the_relaxation_wanted(stage, tmp_path):
    """The gate's three regimes, which are what decides whether to spend."""
    for wanted, phrase in ((29, "truncation regime"), (3, "no truncation"),
                           (12, "partial truncation")):
        out = tmp_path / f"regime_{wanted}"
        done = stage("--stages", "seed", "--output-dir", str(out), wanted=wanted)
        assert done.returncode == 0, done.stderr[-2000:]
        assert phrase in (out / "seeds" / "gate.txt").read_text()


# --------------------------------------------------------------------------- #
# A fourth review round. Each of these passed the checks added in the third.
# --------------------------------------------------------------------------- #


def test_an_external_seed_dir_is_not_overwritten_by_stage_seed(stage, tmp_path):
    """`--seed-dir` exists to REUSE seeds; stage `seed` writes into it.

    The fresh-directory guard covers only `--output-dir`, so a fresh output
    path plus an existing `--seed-dir` overwrote an earlier run's seeds and
    exited 0.
    """
    seeds = tmp_path / "earlier_seeds"
    seeds.mkdir()
    kept = seeds / "apgm_r0.json"
    kept.write_text('{"method": "apgm", "seed_sequence": "ORIGINAL"}')
    done = stage("--stages", "seed", "--output-dir", str(tmp_path / "fresh"),
                 "--seed-dir", str(seeds))
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert "ORIGINAL" in kept.read_text(), "an earlier run's seeds were overwritten"


def test_reusing_a_seed_dir_for_search_only_is_still_allowed(stage, tmp_path):
    """The refusal must not break what `--seed-dir` is for."""
    seeds = tmp_path / "reusable"
    seeds.mkdir()
    for method in ("apgm", "bindcraft", "colabdesign"):
        (seeds / f"{method}_r0.json").write_text(
            json.dumps({"method": method, "seed": 0, "hamming_from_parent": 2,
                        "candidate_substitutions_considered": 3,
                        "edit_budget": 5, "seed_sequence": f"QVQL{method}",
                        "edits": [{"margin": 0.4}]})
        )
    done = stage("--stages", "search", "--output-dir", str(tmp_path / "s_only"),
                 "--seed-dir", str(seeds))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]


@pytest.mark.parametrize("mode, marker", [
    ("silent", "INCOMPLETE"),
    ("silent_most", "INCOMPLETE"),
    ("corrupt", "UNREADABLE"),
])
def test_jobs_that_exit_zero_without_usable_results_still_fail(
    stage, tmp_path, mode, marker
):
    """An exit code cannot catch a job that writes nothing, or writes junk."""
    out = tmp_path / f"zero_exit_{mode}"
    done = stage("--stages", "seed", "--output-dir", str(out), seeds=mode)
    assert done.returncode != 0, (
        f"mode {mode}: every job exited 0 and the launcher reported success"
    )
    assert marker in (out / "seeds" / "gate.txt").read_text()


def test_repeated_devices_are_refused(stage, tmp_path):
    done = stage("--stages", "seed", "--output-dir", str(tmp_path / "same_gpu"),
                 "--devices", "0,0,0,0,0,0,0,0")
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert "repeats a device" in done.stderr


def test_an_empty_device_entry_is_refused(stage, tmp_path):
    done = stage("--stages", "seed", "--output-dir", str(tmp_path / "blank_dev"),
                 "--devices", "0,1,,3")
    assert done.returncode == 2
    assert "empty or non-numeric" in done.stderr


def test_the_dirty_check_ignores_untracked_submodule_content_and_nothing_more():
    """Why `--ignore-submodules=untracked` rather than `=all`.

    `all` would also hide a tracked edit inside the dependency and a changed
    dependency revision -- the two changes most likely to move a result. Built
    on a disposable repo pair, because the distinction is invisible in a tree
    where the submodule is clean.
    """
    import subprocess
    import tempfile

    def git(*args, cwd):
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              text=True, check=True).stdout

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        inner, outer = root / "dep", root / "super"
        for d in (inner, outer):
            d.mkdir()
            git("init", "-q", "-b", "main", cwd=d)
            git("config", "user.email", "t@t", cwd=d)
            git("config", "user.name", "t", cwd=d)
        (inner / "lib.py").write_text("x = 1\n")
        git("add", "-A", cwd=inner)
        git("commit", "-qm", "init", cwd=inner)
        (outer / "README").write_text("super\n")
        git("add", "-A", cwd=outer)
        git("commit", "-qm", "init", cwd=outer)
        git("-c", "protocol.file.allow=always", "submodule", "add", "-q",
            str(inner), "dep", cwd=outer)
        git("commit", "-qm", "add dep", cwd=outer)

        def status(level):
            return subprocess.run(
                ["git", "status", "--porcelain", f"--ignore-submodules={level}"],
                cwd=outer, capture_output=True, text=True,
            ).stdout.strip()

        # untracked content inside the dependency: both flags must hide it
        (outer / "dep" / "scratch.tmp").write_text("junk\n")
        assert status("untracked") == "", status("untracked")
        assert status("all") == ""

        # a TRACKED edit inside the dependency: only `all` hides it
        (outer / "dep" / "lib.py").write_text("x = 2\n")
        assert "dep" in status("untracked"), (
            "a tracked dependency edit must still show as dirty"
        )
        assert status("all") == "", "this is what `all` would have hidden"


def test_both_launchers_use_that_ignore_level():
    examples = STAGE.parent
    for name in ("run_p17_continuous_stage.sh", "run_p17_hinge_probe.sh"):
        source = (examples / name).read_text()
        assert "--ignore-submodules=untracked" in source, name
        assert "--ignore-submodules=all" not in source, name


# --------------------------------------------------------------------------- #
# A fifth round. All four are ways to run the script and get exit 0 for a job
# that did not happen, or happened differently from what was asked.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("flag, value, marker", [
    ("--stages", "sead", "not one of"),
    ("--stages", "", "is empty"),
    ("--cells", "", "is empty"),
    ("--cells", "apgm parent apgm", "repeats a value"),
    ("--methods", "apgm apgm", "repeats a value"),
    ("--methods", "apgmm", "not one of"),
])
def test_an_unusable_selection_is_refused_rather_than_silently_empty(
    stage, tmp_path, flag, value, marker
):
    """`--stages sead` selected no stage, ran nothing, and exited 0."""
    done = stage(flag, value, "--output-dir", str(tmp_path / "sel"))
    assert done.returncode == 2, done.stdout[-1500:] + done.stderr[-1500:]
    assert marker in done.stderr, done.stderr[-1500:]


def test_duplicate_methods_would_have_collided_on_one_output_path(
    stage, tmp_path
):
    """Two `apgm` entries mean four jobs writing to two files.

    The count suggests more data than exists, which is worse than failing.
    """
    done = stage("--stages", "seed", "--methods", "apgm apgm",
                 "--output-dir", str(tmp_path / "dupe"))
    assert done.returncode == 2
    assert "overwrite each" in done.stderr


def test_a_relative_output_dir_resolves_where_it_was_asked_for(stage, tmp_path):
    """Created before `cd "$REPO_DIR"`, then read from the repo root.

    Run from elsewhere, the results landed in one place and were looked for in
    another.
    """
    import os
    import subprocess
    import sys

    mock = tmp_path / "mockpy_rel"
    mock.write_text(MOCK % {"python": sys.executable, "wanted": 3})
    mock.chmod(0o755)
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    env = dict(os.environ, PYTHON_BIN=str(mock), MOCK_SEED="ok", MOCK_SEARCH="ok")
    done = subprocess.run(
        ["bash", str(STAGE), "--devices", DEVICES, "--seed-replicates", "2",
         "--stages", "seed", "--output-dir", "out_here"],
        capture_output=True, text=True, env=env, cwd=elsewhere, timeout=600,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert (elsewhere / "out_here" / "seeds" / "gate.txt").exists(), (
        "the relative --output-dir did not resolve against the invocation "
        f"directory; repo-relative instead? "
        f"{(STAGE.parents[1] / 'out_here').exists()}"
    )
    assert not (STAGE.parents[1] / "out_here").exists()


def test_the_gate_refuses_results_whose_identity_is_wrong(stage, tmp_path):
    """Correctly named files reporting the wrong method and seed."""
    out = tmp_path / "mislabelled"
    done = stage("--stages", "seed", "--output-dir", str(out), seeds="mislabelled")
    assert done.returncode != 0, done.stdout[-2000:]
    gate = (out / "seeds" / "gate.txt").read_text()
    assert "IDENTITY MISMATCH" in gate, gate
    assert "WRONG RESULTS: 6 of 6" in gate, gate


def test_results_carrying_the_wrong_identity_are_refused(stage, tmp_path):
    """Six correctly NAMED files reporting the wrong method and seed.

    Presence and readability were checked; identity was not.
    """
    seeds = tmp_path / "wrong_identity" / "seeds"
    seeds.mkdir(parents=True)
    for method in ("apgm", "bindcraft", "colabdesign"):
        for r in (0, 1):
            (seeds / f"{method}_r{r}.json").write_text(
                json.dumps({"method": "somethingelse", "seed": 99,
                            "hamming_from_parent": 2,
                            "candidate_substitutions_considered": 3,
                            "edit_budget": 5, "seed_sequence": "QVQL",
                            "edits": [{"margin": 0.4}]})
            )
    done = stage("--stages", "search", "--output-dir", str(tmp_path / "wi_out"),
                 "--seed-dir", str(seeds))
    gate = tmp_path / "wi_out"
    assert done.returncode != 0 or "IDENTITY MISMATCH" in done.stdout, (
        done.stdout[-2000:]
    )


# --------------------------------------------------------------------------- #
# A sixth round: validation and execution disagreeing, and a verdict stated
# more broadly than the data it was computed from.
# --------------------------------------------------------------------------- #


def test_whitespace_other_than_spaces_still_selects_the_stages(stage, tmp_path):
    """Validation word-splits on any whitespace; `has_stage` matched literal
    spaces. A tab-separated value validated as two good tokens and then
    selected NO stage, so the script ran nothing and exited 0.
    """
    out = tmp_path / "tabbed"
    done = stage("--stages", "seed\tsearch", "--output-dir", str(out))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert (out / "seeds" / "gate.txt").exists(), (
        "stage seed did not run, so validation and execution still disagree"
    )
    assert "stage search" in done.stdout, "stage search did not run either"


def test_a_reused_seed_with_the_right_method_but_the_wrong_replicate_is_refused(
    stage, tmp_path
):
    """`apgm_r0.json` carrying `"seed": 99` -- the right method, wrong run.

    That is exactly the file a half-finished earlier directory leaves behind,
    and checking only the method accepted it.
    """
    seeds = tmp_path / "wrong_replicate"
    seeds.mkdir()
    for method in ("apgm", "bindcraft", "colabdesign"):
        (seeds / f"{method}_r0.json").write_text(
            json.dumps({"method": method, "seed": 99, "hamming_from_parent": 2,
                        "candidate_substitutions_considered": 3,
                        "edit_budget": 5, "seed_sequence": f"QVQL{method}",
                        "edits": [{"margin": 0.4}]})
        )
    done = stage("--stages", "search", "--output-dir", str(tmp_path / "wr_out"),
                 "--seed-dir", str(seeds))
    assert done.returncode != 0, done.stdout[-2000:]
    assert "IDENTITY MISMATCH" in done.stdout + done.stderr
    assert "seed 99 != 0" in done.stdout + done.stderr


def test_a_max_based_verdict_does_not_claim_to_describe_every_seed(
    stage, tmp_path
):
    """One seed at 29 and five at 2 used to print "each seed is an argmax of a
    heavily budget-truncated optimum", which is false for five of the six.
    """
    out = tmp_path / "mixed"
    done = stage("--stages", "seed", "--output-dir", str(out), seeds="mixed")
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    gate = (out / "seeds" / "gate.txt").read_text()
    assert "2/6 want >= 25" in gate, gate
    assert "4/6 fit the budget" in gate, gate
    assert "truncation regime for 2 of 6 seeds" in gate, gate
    assert "so each seed is an argmax" not in gate
