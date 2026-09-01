import argparse
import csv
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from mosaic.workflows.boltzgen_vhh_guided import (
    VHHDesignConfig,
    build_single_design_command,
    validate_and_normalize_cli_args,
    write_combined_refold_ranking,
)


def test_jax_boltzgen_if_is_default_decoder(tmp_path):
    config = VHHDesignConfig(
        complex_cif_path=tmp_path / "input.cif",
        binder_chain_id="A",
        target_chain_ids=["T"],
        cdr_residue_indices=[26],
    )
    assert config.sequence_decoder == "boltzgen_if"


def _command_args(input_cif: Path):
    values = {
        "mode": "v4",
        "complex_cif": input_cif,
        "boltzgen_yaml": None,
        "binder_chain": "A",
        "target_chains": ["T"],
        "cdr_indices": [26, 27],
        "budget": 5,
        "output_dir": Path("unused"),
        "seed": 0,
        "num_sampling_steps": 20,
        "start_sigma_frac": 0.3,
        "step_scale": 2.0,
        "noise_scale": 0.88,
        "lambda_max": 1.0,
        "lambda_schedule": "sigma_squared",
        "nos_inner_steps": 0,
        "nos_inner_step_size": 0.05,
        "nos_lambda_kl": 1.0,
        "nos_langevin_noise": 0.0,
        "lookahead": 0,
        "n_outer_iterations": 1,
        "search_mode": "mcmc",
        "polish_steps": 10,
        "polish_batch_size": 3,
        "mcmc_steps": 20,
        "mcmc_temp": 0.02,
        "mcmc_proposal_temp": 0.01,
        "mcmc_max_path_length": 2,
        "sequence_decoder": "boltzgen_if",
        "boltzgen_if_checkpoint": None,
        "boltzgen_if_device": "auto",
        "boltzgen_if_temperature": 0.3,
        "boltzgen_if_guidance_temperature": 0.3,
        "boltzgen_if_avoid": "C",
        "weight_boltzgen_if_prior": 0.1,
        "recycling_steps": 1,
        "refold_sampling_steps": 5,
        "refold_num_samples": 1,
        "refold_batch_size": 1,
        "refold_binder_template": 1,
        "refold_binder_template_mode": "framework",
        "ipsae_pae_cutoff": 12.0,
        "refold_rmsd_threshold": 2.5,
        "refold_backend": "boltz2",
        "weight_ablang2": 0.0,
        "weight_edit_budget": 5.0,
        "weight_boltz2_ptm_energy": 0.0,
        "weight_boltz2_interface_pae": 0.0,
        "weight_boltz2_iptm": 0.0,
        "weight_boltz2_ipsae": 0.0,
        "boltz2_guidance_recycling_steps": 0,
        "boltz2_guidance_sampling_steps": 5,
        "boltz2_guidance_target_template": 1,
        "weight_opendde_iptm": 0.0,
        "weight_opendde_contact": 0.0,
        "opendde_guidance_recycling_steps": 4,
        "opendde_contact_distance": 8.0,
        "clip_gradient_norm": 1.0,
        "skip_guidance": 0,
        "skip_polish": 0,
        "skip_refold": 1,
        "log_guidance_diagnostics": 0,
        "guidance_diagnostics_cos_threshold": 0.0,
        "guidance_diagnostics_sigma_bins": 4,
    }
    return SimpleNamespace(**values)


def test_child_command_uses_packaged_module_and_propagates_if_prior(tmp_path):
    input_cif = tmp_path / "input.cif"
    input_cif.touch()
    command = build_single_design_command(
        _command_args(input_cif), seed=11, output_dir=tmp_path / "seed_11"
    )

    assert command[:4] == [
        sys.executable,
        "-u",
        "-m",
        "mosaic.workflows.boltzgen_vhh_guided",
    ]
    assert command[command.index("--complex-cif") + 1] == str(input_cif)
    assert command[command.index("--weight-boltzgen-if-prior") + 1] == "0.1"


def test_cli_validation_requires_input(tmp_path):
    parser = argparse.ArgumentParser()
    args = SimpleNamespace(
        complex_cif=None,
        boltzgen_yaml=None,
        boltzgen_if_checkpoint=None,
        output_dir=tmp_path,
        num_designs=1,
    )
    try:
        validate_and_normalize_cli_args(parser, args)
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("missing input should fail CLI validation")


def test_combined_ranking_accepts_blank_numeric_fields(tmp_path):
    seed_dir = tmp_path / "seed_0"
    seed_dir.mkdir()
    columns = [
        "rank", "edit_count", "sample_idx", "sequence", "ipsae_min", "iptm",
        "ipae_min", "bt_ipsae", "tb_ipsae", "rmsd_pass", "refold_cif",
    ]
    with (seed_dir / "refold_ranked.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({
            "rank": "1", "edit_count": "5", "sample_idx": "0", "sequence": "AAA",
            "ipsae_min": "0.5", "iptm": "0.7", "ipae_min": "4.0",
            "bt_ipsae": "0.5", "tb_ipsae": "0.6", "rmsd_pass": "true",
            "refold_cif": "sample.cif",
        })
        writer.writerow({
            "rank": "", "edit_count": "6", "sample_idx": "1", "sequence": "ARA",
            "ipsae_min": "", "iptm": "0.2", "ipae_min": "12.0",
            "bt_ipsae": "", "tb_ipsae": "", "rmsd_pass": "false",
            "refold_cif": "sample2.cif",
        })

    count = write_combined_refold_ranking(
        tmp_path,
        [{
            "seed": 0,
            "status": "ok",
            "output_dir": str(seed_dir),
            "log": "driver.log",
        }],
    )
    assert count == 2
    rows = list(csv.DictReader((tmp_path / "combined_refold_ranked.csv").open()))
    assert rows[0]["rank"] == "1"
    assert rows[1]["rank"] == ""


def test_module_cli_rejects_missing_input_before_model_loading():
    result = subprocess.run(
        [sys.executable, "-m", "mosaic.workflows.boltzgen_vhh_guided"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "one of --complex-cif or --boltzgen-yaml is required" in result.stderr
