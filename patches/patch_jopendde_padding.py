"""Thread explicit padding masks through atom attention and diffusion.

Padding is identified by out-of-range token assignments, not ref_mask (which
records reference-conformer availability). Empty structural tokens are masked
in the refiner and diffusion attention. Apply before importing model modules.
"""

import importlib.util
from pathlib import Path

MARKER = "MOSAIC PADDING v1"
SITES = []


def site(file, label, original, replacement):
    SITES.append(
        dict(file=file, label=label, original=original, replacement=replacement)
    )


site(
    "transformer.py",
    "windowed bias",
    """        if n_queries and n_keys:
            out = self.attention(q_x=a_normed, kv_x=kv, attn_bias=bias, n_queries=n_queries, n_keys=n_keys)
""",
    """        if n_queries and n_keys:
            # MOSAIC PADDING v1: windowed attention also receives key masks.
            if extra_attn_bias is not None:
                bias = bias + extra_attn_bias
            out = self.attention(q_x=a_normed, kv_x=kv, attn_bias=bias, n_queries=n_queries, n_keys=n_keys)
""",
)
site(
    "transformer.py",
    "atom transformer",
    """        return self.diffusion_transformer(a=q, s=c, z=p, n_queries=self.n_queries, n_keys=self.n_keys)
""",
    """        # MOSAIC PADDING v1: explicit mask from encoder/decoder token bounds.
        from mosaic.opendde_padding import atom_key_bias
        mask = _ignored.get("atom_mask")
        bias = None if mask is None else atom_key_bias(mask, n_queries, n_keys, q.dtype)
        return self.diffusion_transformer(
            a=q, s=c, z=p, n_queries=self.n_queries, n_keys=self.n_keys,
            extra_attn_bias=bias,
        )
""",
)
site(
    "transformer.py",
    "encoder",
    """        q_l = self.atom_transformer(q_l, c_l, p_lm)
""",
    """        # MOSAIC PADDING v1: padded keys cannot influence real queries.
        from mosaic.opendde_padding import active_atoms
        if n_token is None:
            n_token = int(jnp.max(atom_to_token_idx)) + 1
        atom_mask = active_atoms(atom_to_token_idx, n_token)
        c_l = jnp.where(atom_mask[:, None], c_l, 0)
        q_l = jnp.where(atom_mask[:, None], q_l, 0)
        q_l = self.atom_transformer(q_l, c_l, p_lm, atom_mask=atom_mask)
        q_l = jnp.where(atom_mask[:, None], q_l, 0)
""",
)
site(
    "transformer.py",
    "decoder",
    """        q = self.atom_transformer(q, c_skip, p_skip)
        q = self.layernorm_q(q)
        return self.linear_no_bias_out(q)
""",
    """        # MOSAIC PADDING v1: use the same mask in the atom decoder.
        from mosaic.opendde_padding import active_atoms
        atom_mask = active_atoms(atom_to_token_idx, a.shape[-2])
        q = jnp.where(atom_mask[:, None], q, 0)
        c_skip = jnp.where(atom_mask[:, None], c_skip, 0)
        q = self.atom_transformer(q, c_skip, p_skip, atom_mask=atom_mask)
        q = self.layernorm_q(q)
        return jnp.where(atom_mask[:, None], self.linear_no_bias_out(q), 0)
""",
)
site(
    "model.py",
    "structural refiner",
    """        s_st, z_st = self.structural_token_refiner(
            s_st, z_st, pair_mask=None, extra_attn_bias=attn_bias
        )
""",
    """        # MOSAIC PADDING v1: atomless subtokens do not participate in attention.
        from mosaic.opendde_padding import mask_structural_inputs
        s_inputs_st, s_st, z_st, attn_bias, pair_mask = mask_structural_inputs(
            struct_feat.atom_to_token_idx, s_inputs_st, s_st, z_st, attn_bias
        )
        s_st, z_st = self.structural_token_refiner(
            s_st, z_st, pair_mask=pair_mask, extra_attn_bias=attn_bias
        )
        valid = jnp.any(pair_mask, axis=-1)
        s_st = jnp.where(valid[:, None], s_st, 0)
        z_st = jnp.where(pair_mask[..., None], z_st, 0)
""",
)
site(
    "diffusion.py",
    "initial coordinates",
    """    x_l0 = schedule[0] * jax.random.normal(init_key, batch_shape + (N_sample, N_atom, 3))
""",
    """    x_l0 = schedule[0] * jax.random.normal(init_key, batch_shape + (N_sample, N_atom, 3))
    # MOSAIC PADDING v1: padding is not part of the noisy coordinate cloud.
    from mosaic.opendde_padding import active_atoms
    atom_mask = active_atoms(feat.atom_to_token_idx, s_inputs.shape[-2])
    x_l0 = jnp.where(atom_mask[:, None], x_l0, 0)
""",
)
site(
    "diffusion.py",
    "centering",
    """        x_l = centre_random_augmentation(x_l, key=aug_key, N_sample=1)[..., 0, :, :]
""",
    """        # MOSAIC PADDING v1: center only real atoms.
        x_l = centre_random_augmentation(x_l, key=aug_key, N_sample=1, mask=atom_mask)[..., 0, :, :]
""",
)
site(
    "diffusion.py",
    "noise",
    """        x_noisy = x_l + noise_scale_lambda * delta_noise_level * jax.random.normal(noise_key, x_l.shape)
""",
    """        x_noisy = x_l + noise_scale_lambda * delta_noise_level * jax.random.normal(noise_key, x_l.shape)
        # MOSAIC PADDING v1: do not inject noise into padding.
        x_noisy = jnp.where(atom_mask[:, None], x_noisy, 0)
""",
)
site(
    "diffusion.py",
    "updated coordinates",
    """        x_l_next = x_noisy + step_scale_eta * dt[..., None, None] * delta
""",
    """        x_l_next = x_noisy + step_scale_eta * dt[..., None, None] * delta
        # MOSAIC PADDING v1: retain the invariant across all diffusion steps.
        x_l_next = jnp.where(atom_mask[:, None], x_l_next, 0)
""",
)


def patch_package(directory):
    directory = Path(directory)
    prepared = {}
    # Validate every site before writing any source file.
    for item in SITES:
        path = directory / item["file"]
        source = prepared.get(path, path.read_text())
        if source.count(item["replacement"]) == 1:
            prepared[path] = source
            continue
        marker_line = next(
            line.strip() for line in item["replacement"].splitlines() if MARKER in line
        )
        if marker_line in source or source.count(item["original"]) != 1:
            raise ValueError(f"unsupported {item['file']}: {item['label']}")
        prepared[path] = source.replace(item["original"], item["replacement"])
    for path, source in prepared.items():
        compile(source, str(path), "exec")
    for path, source in prepared.items():
        changed = source != path.read_text()
        if changed:
            path.write_text(source)
        Path(importlib.util.cache_from_source(str(path))).unlink(missing_ok=True)
        print(f"{path}: {'patched' if changed else 'already patched'} padding")


def main():
    spec = importlib.util.find_spec("jopendde")
    if spec is None or spec.origin is None:
        raise RuntimeError("jopendde is not installed")
    patch_package(Path(spec.origin).parent)


if __name__ == "__main__":
    main()
