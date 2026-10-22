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
  # 2026-10-14: the stage records how many positions were designable and how
  # far the relaxation moved. A seed JSON without them is a PRE-2026-10-14
  # one, which the gate reads differently, so the mock emits them by default
  # and the two legacy modes below are the exceptions.
  designable=29
  [[ "$MOCK_SEED" == "mask_changed" && "$method" == apgm ]] && designable=15
  # Every run on a DIFFERENT mask than 29, consistently: the case a gate that
  # prints the constant 29 cannot be distinguished from one that reads it.
  [[ "$MOCK_SEED" == "mask_17" ]] && designable=17
  # 0 wanted, and the two opposite causes the argmax cannot separate.
  peak=0.95; parent_mass=0.02
  if [[ "$MOCK_SEED" == "frozen" || "$MOCK_SEED" == "converged" ]]; then
    wanted_override=0
    if [[ "$MOCK_SEED" == "frozen" ]]; then peak=0.125; parent_mass=0.125
    else peak=0.981; parent_mass=0.981; fi
  fi
  edits='[{"margin": 0.4}, {"margin": 0.9}]'
  hamming=2
  if [[ "${wanted_override:-1}" == 0 ]]; then edits='[]'; hamming=0; fi
  # A heredoc, not a quoted assignment: `"` is literal inside one, and the
  # escaped-quote version of this produced JSON whose keys had lost their
  # quotes -- six files the gate reported as unreadable.
  legacy=false
  [[ "$MOCK_SEED" == "legacy_all" ]] && legacy=true
  [[ "$MOCK_SEED" == "legacy_mix" && "$method" == apgm ]] && legacy=true
  extra=""
  if [[ "$legacy" == false ]]; then
    extra="$(cat <<EOX
,
 "designable_positions": $designable,
 "relaxed_peak_probability_mean": $peak,
 "relaxed_peak_probability_min": $peak,
 "relaxed_parent_probability_mean": $parent_mass,
 "relaxed_row_sum_max_deviation": 0.0,
 "projection_tied_group": 1,
 "projection_tied_discarded": 0,
 "projection_margin_gap_at_cut": 0.01,
 "loss_at_init": 40.5,
 "loss_at_relaxed": 12.25,
 "loss_descent": 28.25,
 "loss_eval_error": null
EOX
)"
  fi
  cat > "$out" <<EOJ
{"method": "$method",
 "seed": $seed,
 "hamming_from_parent": $hamming,
 "candidate_substitutions_considered": ${wanted_override:-%(wanted)s},
 "edit_budget": 5,
 "seed_sequence": "QVQL$method$seed",
 "edits": $edits$extra}
EOJ
  exit 0
fi
if [[ "$1" == *p17_alpha_recovery.py ]]; then
  # Opt-in: unset for every other test, so none of them change behaviour.
  # Doubled percent: the fixture formats this whole script as a template.
  [[ -n "$MOCK_ARGS_DIR" ]] && printf '%%s\n' "$@" > "$MOCK_ARGS_DIR/driver_$$.args"
  [[ "$MOCK_SEARCH" == "fail" ]] && exit 1
  exit 0
fi
exit 0
"""


IDLE_EIGHT = """#!/usr/bin/env bash
case "$*" in
  *--query-gpu*) for i in 0 1 2 3 4 5 6 7; do echo "$i, 143771, 1024"; done ;;
  *) exit 1 ;;
esac
"""


def _stub_nvidia_smi(directory, body=IDLE_EIGHT):
    """Return a PATH prefix whose `nvidia-smi` is `body`."""
    binary = directory / "stubbin"
    binary.mkdir(exist_ok=True)
    stub = binary / "nvidia-smi"
    stub.write_text(body)
    stub.chmod(0o755)
    return str(binary)


@pytest.fixture
def stage(tmp_path):
    import sys

    def run(*args, seeds="ok", search="ok", wanted=3, args_dir=None):
        mock = tmp_path / f"mockpy_{wanted}_{seeds}"
        mock.write_text(MOCK % {"python": sys.executable, "wanted": wanted})
        mock.chmod(0o755)
        env = dict(
            os.environ, PYTHON_BIN=str(mock), MOCK_SEED=seeds, MOCK_SEARCH=search,
            MOCK_ARGS_DIR=str(args_dir) if args_dir else "",
        )
        env["PATH"] = _stub_nvidia_smi(tmp_path) + os.pathsep + env["PATH"]
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

        (outer / "dep" / "scratch.tmp").write_text("junk\n")
        assert status("untracked") == "", status("untracked")
        assert status("all") == ""

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
    env["PATH"] = _stub_nvidia_smi(tmp_path) + os.pathsep + env["PATH"]
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


def test_a_device_that_cannot_hold_the_preallocation_is_refused(stage, tmp_path):
    """The capacity check is wired in, and the stub above is not hiding it.

    Stubbing `nvidia-smi` to eight idle devices makes every other test here
    host-independent -- but it would also hide the check being absent
    altogether, so one case stubs a device that is nearly full.
    """
    import os
    import subprocess
    import sys

    mock = tmp_path / "mockpy_full"
    mock.write_text(MOCK % {"python": sys.executable, "wanted": 3})
    mock.chmod(0o755)
    nearly_full = """#!/usr/bin/env bash
case "$*" in
  *--query-gpu*) for i in 0 1 2 3 4 5 6 7; do echo "$i, 143771, 140000"; done ;;
  *) exit 1 ;;
esac
"""
    (tmp_path / "full").mkdir(exist_ok=True)
    env = dict(os.environ, PYTHON_BIN=str(mock), MOCK_SEED="ok", MOCK_SEARCH="ok")
    env["PATH"] = (
        _stub_nvidia_smi(tmp_path / "full", nearly_full) + os.pathsep + env["PATH"]
    )
    done = subprocess.run(
        ["bash", str(STAGE), "--devices", DEVICES, "--seed-replicates", "2",
         "--stages", "seed", "--output-dir", str(tmp_path / "nofit")],
        capture_output=True, text=True, env=env, timeout=600,
    )
    assert done.returncode == 2, done.stdout[-2000:] + done.stderr[-2000:]
    assert "DOES NOT FIT" in done.stdout
    assert "Refusing to start" in done.stderr


def test_allow_busy_gpus_proceeds_past_a_full_device(stage, tmp_path):
    import os
    import subprocess
    import sys

    mock = tmp_path / "mockpy_force"
    mock.write_text(MOCK % {"python": sys.executable, "wanted": 3})
    mock.chmod(0o755)
    nearly_full = """#!/usr/bin/env bash
case "$*" in
  *--query-gpu*) for i in 0 1 2 3 4 5 6 7; do echo "$i, 143771, 140000"; done ;;
  *) exit 1 ;;
esac
"""
    (tmp_path / "force").mkdir(exist_ok=True)
    env = dict(os.environ, PYTHON_BIN=str(mock), MOCK_SEED="ok", MOCK_SEARCH="ok")
    env["PATH"] = (
        _stub_nvidia_smi(tmp_path / "force", nearly_full) + os.pathsep + env["PATH"]
    )
    done = subprocess.run(
        ["bash", str(STAGE), "--devices", DEVICES, "--seed-replicates", "2",
         "--stages", "seed", "--allow-busy-gpus",
         "--output-dir", str(tmp_path / "forced")],
        capture_output=True, text=True, env=env, timeout=600,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "Expect an OOM" in done.stderr




def test_the_gate_reads_the_designable_count_from_the_runs(stage, tmp_path):
    """`designable 29` was a constant in the gate's own source.

    It is correct today and it is the denominator the `>= 25` threshold was
    chosen against, but nothing tied it to `CDR_RESIDUE_INDICES_1IDX`:
    changing the mask would have left the gate printing a stale denominator
    under a threshold that no longer meant what it says.
    """
    out = tmp_path / "denominator"
    done = stage("--stages", "seed", "--output-dir", str(out))
    assert done.returncode == 0, done.stdout[-2000:]
    gate = (out / "seeds" / "gate.txt").read_text()
    assert "designable 29" in gate, gate
    assert "assumed" not in gate, "the count IS recorded; nothing is assumed"

    other = tmp_path / "denominator_17"
    done = stage("--stages", "seed", "--output-dir", str(other), seeds="mask_17")
    assert done.returncode == 0, done.stdout[-2000:]
    gate = (other / "seeds" / "gate.txt").read_text()
    assert "designable 17" in gate, gate
    assert "designable 29" not in gate, gate
    assert "threshold below was chosen against 29" in gate, gate


def test_seed_runs_disagreeing_on_the_designable_count_are_refused(stage, tmp_path):
    """Two masks is two experiments, and the verdict would average them."""
    out = tmp_path / "mask_changed"
    done = stage("--stages", "seed", "--output-dir", str(out), seeds="mask_changed")
    assert done.returncode != 0, done.stdout[-2000:]
    gate = (out / "seeds" / "gate.txt").read_text()
    assert "disagree on how many positions are designable" in gate, gate
    assert "15" in gate and "29" in gate, gate
    assert "VERDICT" not in gate, "the verdict must not print over a split mask"


def test_a_seed_directory_holding_two_builds_of_the_seeder_is_refused(
    stage, tmp_path
):
    """A missing count is not a disagreeing one, and the refusal says which.

    Both exit nonzero, but naming the wrong reason sends the reader to the
    wrong file.
    """
    out = tmp_path / "legacy_mix"
    done = stage("--stages", "seed", "--output-dir", str(out), seeds="legacy_mix")
    assert done.returncode != 0, done.stdout[-2000:]
    gate = (out / "seeds" / "gate.txt").read_text()
    assert "two builds of the seeder" in gate, gate
    assert "disagree on how many positions" not in gate, gate
    assert "2 of 6 seed runs predate" in gate, gate


def test_seed_runs_from_before_the_count_was_recorded_still_read(stage, tmp_path):
    """The field is 2026-10-14; the directories from before it still gate."""
    out = tmp_path / "legacy_all"
    done = stage("--stages", "seed", "--output-dir", str(out), seeds="legacy_all")
    assert done.returncode == 0, done.stdout[-2000:]
    gate = (out / "seeds" / "gate.txt").read_text()
    assert "29 (assumed; not recorded before 2026-10-14)" in gate, gate
    assert "VERDICT" in gate, gate
    assert "not recorded" in gate


def test_zero_wanted_separates_a_converged_relaxation_from_a_frozen_one(
    stage, tmp_path
):
    """The question the real-loss run left open, and could not answer.

    On the anchored start the parent is the argmax at every designable
    position before the first step, so `0 wanted` is produced both by a
    relaxation that chose the parent and by one that never moved. At
    `--init-logit-scale 1.0` the untouched peak is 0.125; a converged one is
    near 1. The gate now says which it is looking at.
    """
    frozen = tmp_path / "frozen"
    moved = tmp_path / "converged"
    a = stage("--stages", "seed", "--output-dir", str(frozen), seeds="frozen")
    b = stage("--stages", "seed", "--output-dir", str(moved), seeds="converged")
    assert a.returncode == 0, a.stdout[-2000:]
    assert b.returncode == 0, b.stdout[-2000:]
    ga = (frozen / "seeds" / "gate.txt").read_text()
    gb = (moved / "seeds" / "gate.txt").read_text()
    for gate in (ga, gb):
        assert "0 edits kept / 0 wanted" in gate, gate
        assert "no truncation" in gate, gate
    assert "may never have left the start" in ga, ga
    assert "converged on the parent" not in ga, ga
    assert "converged on the parent" in gb, gb
    assert "may never have left the start" not in gb, gb




def test_the_edit_budget_defaults_to_five(stage, tmp_path):
    done = stage("--stages", "search", "--cells", "parent", "--dry-run",
                 "--output-dir", str(tmp_path / "default"))
    assert "edit_budget 5," in done.stdout, done.stdout
    assert "OVERRIDDEN" not in done.stdout


def test_a_raised_edit_budget_reaches_the_plan_and_the_driver(stage, tmp_path):
    out = tmp_path / "budget7"
    seen = tmp_path / "driverargs"
    seen.mkdir()
    done = stage("--stages", "search", "--cells", "parent", "--edit-budget", "7",
                 "--output-dir", str(out), args_dir=seen)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    plan = json.loads((out / "plan.json").read_text())
    assert plan["search_stage"]["edit_budget"] == 7
    assert plan["search_stage"]["edit_budget_default"] == 5
    recorded = [f.read_text().split() for f in seen.glob("driver_*.args")]
    assert recorded, "the mock driver was never invoked, so this asserts nothing"
    for argv in recorded:
        assert "--edit-budget" in argv, argv
        assert argv[argv.index("--edit-budget") + 1] == "7", argv


def test_the_override_is_named_wherever_the_budget_is_printed(stage, tmp_path):
    done = stage("--stages", "search", "--cells", "parent", "--edit-budget", "7",
                 "--dry-run", "--output-dir", str(tmp_path / "say_so"))
    assert "edit_budget 7 (OVERRIDDEN, default 5)" in done.stdout, done.stdout


@pytest.mark.parametrize("bad", ["0", "-1", "x", "5.5", "07x", ""])
def test_an_edit_budget_that_is_not_a_positive_integer_is_refused(
    stage, tmp_path, bad
):
    done = stage("--stages", "search", "--cells", "parent", "--dry-run",
                 "--edit-budget", bad,
                 "--output-dir", str(tmp_path / "bad"))
    assert done.returncode == 2, (bad, done.stdout, done.stderr)
    assert "positive integer" in done.stderr, done.stderr



def _seed_dir(tmp_path, name, extra):
    seeds = tmp_path / name
    seeds.mkdir()
    for method in ("apgm", "bindcraft", "colabdesign"):
        payload = {"method": method, "seed": 0, "hamming_from_parent": 2,
                   "candidate_substitutions_considered": 3,
                   "edit_budget": 5, "seed_sequence": f"QVQL{method}",
                   "edits": [{"margin": 0.4}]}
        payload.update(extra)
        (seeds / f"{method}_r0.json").write_text(json.dumps(payload))
    return seeds


def test_a_seed_that_beats_the_parent_is_reported_as_usable(stage, tmp_path):
    seeds = _seed_dir(tmp_path, "good", {
        "loss_at_parent": 12.0, "loss_at_rounded": 9.5,
        "rounded_gain_over_parent": 2.5, "rounded_loss_eval_error": None})
    done = stage("--stages", "search", "--cells", "bindcraft",
                 "--output-dir", str(tmp_path / "out"), "--seed-dir", str(seeds))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "beats the parent by 2.5" in done.stdout
    assert "NO GAIN" not in done.stdout + done.stderr
    assert "not whether seeding helps" not in done.stdout


@pytest.mark.parametrize("gain", [0, -0.4])
def test_a_seed_that_does_not_beat_the_parent_is_flagged(stage, tmp_path, gain):
    """The 2026-10-15 case: five edits spent, nothing bought."""
    seeds = _seed_dir(tmp_path, "flat", {
        "loss_at_parent": 12.0, "loss_at_rounded": 12.0 - gain,
        "rounded_gain_over_parent": gain, "rounded_loss_eval_error": None})
    done = stage("--stages", "search", "--cells", "bindcraft",
                 "--output-dir", str(tmp_path / "out"), "--seed-dir", str(seeds))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "NO GAIN" in done.stderr
    assert "3 free moves" in done.stderr
    assert "not whether seeding helps" in done.stdout


def test_a_seed_written_before_the_check_is_not_called_useless(stage, tmp_path):
    """Absent evidence is not evidence of a worthless seed."""
    seeds = _seed_dir(tmp_path, "legacy", {})
    done = stage("--stages", "search", "--cells", "bindcraft",
                 "--output-dir", str(tmp_path / "out"), "--seed-dir", str(seeds))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "UNKNOWN" in done.stdout
    assert "rerun --stages seed" in done.stdout
    assert "NO GAIN" not in done.stderr


def test_a_seed_whose_loss_failed_to_evaluate_names_the_reason(stage, tmp_path):
    seeds = _seed_dir(tmp_path, "nulled", {
        "loss_at_parent": None, "loss_at_rounded": None,
        "rounded_gain_over_parent": None,
        "rounded_loss_eval_error": "RuntimeError('cuda oom')"})
    done = stage("--stages", "search", "--cells", "bindcraft",
                 "--output-dir", str(tmp_path / "out"), "--seed-dir", str(seeds))
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "UNKNOWN" in done.stdout
    assert "cuda oom" in done.stdout
    assert "NO GAIN" not in done.stderr
