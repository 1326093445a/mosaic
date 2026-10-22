
"""Ops for all atom representations.

Generally we employ two different representations for all atom coordinates,
one is atom37 where each heavy atom corresponds to a given position in a 37
dimensional array, This mapping is non amino acid specific, but each slot
corresponds to an atom of a given name, for example slot 12 always corresponds
to 'C delta 1', positions that are not present for a given amino acid are
zeroed out and denoted by a mask.
The other representation we employ is called atom14, this is a more dense way
of representing atoms with 14 slots. Here a given slot will correspond to a
different kind of atom depending on amino acid type, for example slot 5
corresponds to 'N delta 2' for Aspargine, but to 'C delta 1' for Isoleucine.
14 is chosen because it is the maximum number of heavy atoms for any standard
amino acid.
The order of slots can be found in 'residue_constants.residue_atoms'.
Internally the model uses the atom14 representation because it is
computationally more efficient.
The internal atom14 representation is turned into the atom37 at the output of
the network to facilitate easier conversion to existing protein datastructures.
"""

from typing import Dict, Optional
from ..common import residue_constants
from .all_atom_multimer import _make_restype_atom37_to_atom14, _make_restype_atom14_to_atom37

from . import r3, utils
import jax
import jax.numpy as jnp
import numpy as np


restype_atom14_mask = jnp.array(residue_constants.restype_atom14_mask)
restype_atom37_mask = jnp.array(residue_constants.restype_atom37_mask)
residx_atom37_to_atom14 = jnp.array(_make_restype_atom37_to_atom14())
residx_atom14_to_atom37 = jnp.array(_make_restype_atom14_to_atom37())

def squared_difference(x, y):
  return jnp.square(x - y)


def get_chi_atom_indices():
  """Returns atom indices needed to compute chi angles for all residue types.

  Returns:
    A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
    in the order specified in residue_constants.restypes + unknown residue type
    at the end. For chi angles which are not defined on the residue, the
    positions indices are by default set to 0.
  """
  chi_atom_indices = []
  for residue_name in residue_constants.restypes:
    residue_name = residue_constants.restype_1to3[residue_name]
    residue_chi_angles = residue_constants.chi_angles_atoms[residue_name]
    atom_indices = []
    for chi_angle in residue_chi_angles:
      atom_indices.append(
          [residue_constants.atom_order[atom] for atom in chi_angle])
    for _ in range(4 - len(atom_indices)):
      atom_indices.append([0, 0, 0, 0])
    chi_atom_indices.append(atom_indices)

  chi_atom_indices.append([[0, 0, 0, 0]] * 4)

  return jnp.asarray(chi_atom_indices)


def atom14_to_atom37(atom14_data: jnp.ndarray,
                     batch: Dict[str, jnp.ndarray]
                    ) -> jnp.ndarray:
  """Convert atom14 to atom37 representation."""
  assert len(atom14_data.shape) in [2, 3]
  batch_residx_atom37_to_atom14 = residx_atom37_to_atom14[batch['aatype']]
  atom37_atom_exists = restype_atom37_mask[batch['aatype']]

  atom37_data = utils.batched_gather(atom14_data,
                                     batch_residx_atom37_to_atom14,
                                     batch_dims=1)
  if len(atom14_data.shape) == 2:
    atom37_data *= atom37_atom_exists
  elif len(atom14_data.shape) == 3:
    atom37_data *= atom37_atom_exists[:, :, None].astype(atom37_data.dtype)
  return atom37_data


def atom37_to_atom14(
    atom37_data: jnp.ndarray,
    batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
  """Convert atom14 to atom37 representation."""
  assert len(atom37_data.shape) in [2, 3]
  batch_residx_atom14_to_atom37 = residx_atom14_to_atom37[batch['aatype']]
  atom14_atom_exists = restype_atom14_mask[batch['aatype']]

  atom14_data = utils.batched_gather(atom37_data,
                                     batch_residx_atom14_to_atom37,
                                     batch_dims=1)
  if len(atom37_data.shape) == 2:
    atom14_data *= atom14_atom_exists.astype(atom14_data.dtype)
  elif len(atom37_data.shape) == 3:
    atom14_data *= atom14_atom_exists[:, :, None].astype(atom14_data.dtype)
  return atom14_data


def atom37_to_frames(
    aatype: jnp.ndarray,
    all_atom_positions: jnp.ndarray,
    all_atom_mask: jnp.ndarray,
) -> Dict[str, jnp.ndarray]:
  """Computes the frames for the up to 8 rigid groups for each residue.

  The rigid groups are defined by the possible torsions in a given amino acid.
  We group the atoms according to their dependence on the torsion angles into
  "rigid groups".  E.g., the position of atoms in the chi2-group depend on
  chi1 and chi2, but do not depend on chi3 or chi4.
  Jumper et al. (2021) Suppl. Table 2 and corresponding text.

  Args:
    aatype: Amino acid type, given as array with integers.
    all_atom_positions: atom37 representation of all atom coordinates.
    all_atom_mask: atom37 representation of mask on all atom coordinates.
  Returns:
    Dictionary containing:
      * 'rigidgroups_gt_frames': 8 Frames corresponding to 'all_atom_positions'
           represented as flat 12 dimensional array.
      * 'rigidgroups_gt_exists': Mask denoting whether the atom positions for
          the given frame are available in the ground truth, e.g. if they were
          resolved in the experiment.
      * 'rigidgroups_group_exists': Mask denoting whether given group is in
          principle present for given amino acid type.
      * 'rigidgroups_group_is_ambiguous': Mask denoting whether frame is
          affected by naming ambiguity.
      * 'rigidgroups_alt_gt_frames': 8 Frames with alternative atom renaming
          corresponding to 'all_atom_positions' represented as flat
          12 dimensional array.
  """
  aatype_in_shape = aatype.shape

  aatype = jnp.reshape(aatype, [-1])
  all_atom_positions = jnp.reshape(all_atom_positions, [-1, 37, 3])
  all_atom_mask = jnp.reshape(all_atom_mask, [-1, 37])

  restype_rigidgroup_base_atom_names = np.full([21, 8, 3], '', dtype=object)

  restype_rigidgroup_base_atom_names[:, 0, :] = ['C', 'CA', 'N']

  restype_rigidgroup_base_atom_names[:, 3, :] = ['CA', 'C', 'O']

  for restype, restype_letter in enumerate(residue_constants.restypes):
    resname = residue_constants.restype_1to3[restype_letter]
    for chi_idx in range(4):
      if residue_constants.chi_angles_mask[restype][chi_idx]:
        atom_names = residue_constants.chi_angles_atoms[resname][chi_idx]
        restype_rigidgroup_base_atom_names[
            restype, chi_idx + 4, :] = atom_names[1:]

  restype_rigidgroup_mask = np.zeros([21, 8], dtype=np.float32)
  restype_rigidgroup_mask[:, 0] = 1
  restype_rigidgroup_mask[:, 3] = 1
  restype_rigidgroup_mask[:20, 4:] = residue_constants.chi_angles_mask

  lookuptable = residue_constants.atom_order.copy()
  lookuptable[''] = 0
  restype_rigidgroup_base_atom37_idx = np.vectorize(lambda x: lookuptable[x])(
      restype_rigidgroup_base_atom_names)

  residx_rigidgroup_base_atom37_idx = utils.batched_gather(
      restype_rigidgroup_base_atom37_idx, aatype)

  base_atom_pos = utils.batched_gather(
      all_atom_positions,
      residx_rigidgroup_base_atom37_idx,
      batch_dims=1)

  gt_frames = r3.rigids_from_3_points(
      point_on_neg_x_axis=r3.vecs_from_tensor(base_atom_pos[:, :, 0, :]),
      origin=r3.vecs_from_tensor(base_atom_pos[:, :, 1, :]),
      point_on_xy_plane=r3.vecs_from_tensor(base_atom_pos[:, :, 2, :])
  )

  group_exists = utils.batched_gather(restype_rigidgroup_mask, aatype)

  gt_atoms_exist = utils.batched_gather(
      all_atom_mask.astype(jnp.float32),
      residx_rigidgroup_base_atom37_idx,
      batch_dims=1)
  gt_exists = jnp.min(gt_atoms_exist, axis=-1) * group_exists

  rots = np.tile(np.eye(3, dtype=np.float32), [8, 1, 1])
  rots[0, 0, 0] = -1
  rots[0, 2, 2] = -1
  gt_frames = r3.rigids_mul_rots(gt_frames, r3.rots_from_tensor3x3(rots))

  restype_rigidgroup_is_ambiguous = np.zeros([21, 8], dtype=np.float32)
  restype_rigidgroup_rots = np.tile(np.eye(3, dtype=np.float32), [21, 8, 1, 1])

  for resname, _ in residue_constants.residue_atom_renaming_swaps.items():
    restype = residue_constants.restype_order[
        residue_constants.restype_3to1[resname]]
    chi_idx = int(sum(residue_constants.chi_angles_mask[restype]) - 1)
    restype_rigidgroup_is_ambiguous[restype, chi_idx + 4] = 1
    restype_rigidgroup_rots[restype, chi_idx + 4, 1, 1] = -1
    restype_rigidgroup_rots[restype, chi_idx + 4, 2, 2] = -1

  residx_rigidgroup_is_ambiguous = utils.batched_gather(
      restype_rigidgroup_is_ambiguous, aatype)
  residx_rigidgroup_ambiguity_rot = utils.batched_gather(
      restype_rigidgroup_rots, aatype)

  alt_gt_frames = r3.rigids_mul_rots(
      gt_frames, r3.rots_from_tensor3x3(residx_rigidgroup_ambiguity_rot))

  gt_frames_flat12 = r3.rigids_to_tensor_flat12(gt_frames)
  alt_gt_frames_flat12 = r3.rigids_to_tensor_flat12(alt_gt_frames)

  gt_frames_flat12 = jnp.reshape(gt_frames_flat12, aatype_in_shape + (8, 12))
  gt_exists = jnp.reshape(gt_exists, aatype_in_shape + (8,))
  group_exists = jnp.reshape(group_exists, aatype_in_shape + (8,))
  gt_frames_flat12 = jnp.reshape(gt_frames_flat12, aatype_in_shape + (8, 12))
  residx_rigidgroup_is_ambiguous = jnp.reshape(residx_rigidgroup_is_ambiguous,
                                               aatype_in_shape + (8,))
  alt_gt_frames_flat12 = jnp.reshape(alt_gt_frames_flat12,
                                     aatype_in_shape + (8, 12,))

  return {
      'rigidgroups_gt_frames': gt_frames_flat12,
      'rigidgroups_gt_exists': gt_exists,
      'rigidgroups_group_exists': group_exists,
      'rigidgroups_group_is_ambiguous':
          residx_rigidgroup_is_ambiguous,
      'rigidgroups_alt_gt_frames': alt_gt_frames_flat12,
  }


def atom37_to_torsion_angles(
    aatype: jnp.ndarray,
    all_atom_pos: jnp.ndarray,
    all_atom_mask: jnp.ndarray,
    placeholder_for_undefined=False,
) -> Dict[str, jnp.ndarray]:
  """Computes the 7 torsion angles (in sin, cos encoding) for each residue.

  The 7 torsion angles are in the order
  '[pre_omega, phi, psi, chi_1, chi_2, chi_3, chi_4]',
  here pre_omega denotes the omega torsion angle between the given amino acid
  and the previous amino acid.

  Args:
    aatype: Amino acid type, given as array with integers.
    all_atom_pos: atom37 representation of all atom coordinates.
    all_atom_mask: atom37 representation of mask on all atom coordinates.
    placeholder_for_undefined: flag denoting whether to set masked torsion
      angles to zero.
  Returns:
    Dict containing:
      * 'torsion_angles_sin_cos': Array with shape (B, N, 7, 2) where the final
        2 dimensions denote sin and cos respectively
      * 'alt_torsion_angles_sin_cos': same as 'torsion_angles_sin_cos', but
        with the angle shifted by pi for all chi angles affected by the naming
        ambiguities.
      * 'torsion_angles_mask': Mask for which chi angles are present.
  """

  aatype = jnp.minimum(aatype, 20)

  num_batch, num_res = aatype.shape

  pad = jnp.zeros([num_batch, 1, 37, 3], jnp.float32)
  prev_all_atom_pos = jnp.concatenate([pad, all_atom_pos[:, :-1, :, :]], axis=1)

  pad = jnp.zeros([num_batch, 1, 37], jnp.float32)
  prev_all_atom_mask = jnp.concatenate([pad, all_atom_mask[:, :-1, :]], axis=1)

  pre_omega_atom_pos = jnp.concatenate(
      [prev_all_atom_pos[:, :, 1:3, :],
       all_atom_pos[:, :, 0:2, :]
      ], axis=-2)
  phi_atom_pos = jnp.concatenate(
      [prev_all_atom_pos[:, :, 2:3, :],
       all_atom_pos[:, :, 0:3, :]
      ], axis=-2)
  psi_atom_pos = jnp.concatenate(
      [all_atom_pos[:, :, 0:3, :],
       all_atom_pos[:, :, 4:5, :]
      ], axis=-2)

  pre_omega_mask = (
      jnp.prod(prev_all_atom_mask[:, :, 1:3], axis=-1)
      * jnp.prod(all_atom_mask[:, :, 0:2], axis=-1))
  phi_mask = (
      prev_all_atom_mask[:, :, 2]
      * jnp.prod(all_atom_mask[:, :, 0:3], axis=-1))
  psi_mask = (
      jnp.prod(all_atom_mask[:, :, 0:3], axis=-1) *
      all_atom_mask[:, :, 4])

  chi_atom_indices = get_chi_atom_indices()
  atom_indices = utils.batched_gather(
      params=chi_atom_indices, indices=aatype, axis=0, batch_dims=0)
  chis_atom_pos = utils.batched_gather(
      params=all_atom_pos, indices=atom_indices, axis=-2,
      batch_dims=2)

  chi_angles_mask = list(residue_constants.chi_angles_mask)
  chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
  chi_angles_mask = jnp.asarray(chi_angles_mask)

  chis_mask = utils.batched_gather(params=chi_angles_mask, indices=aatype,
                                   axis=0, batch_dims=0)

  chi_angle_atoms_mask = utils.batched_gather(
      params=all_atom_mask, indices=atom_indices, axis=-1,
      batch_dims=2)
  chi_angle_atoms_mask = jnp.prod(chi_angle_atoms_mask, axis=[-1])
  chis_mask = chis_mask * (chi_angle_atoms_mask).astype(jnp.float32)

  torsions_atom_pos = jnp.concatenate(
      [pre_omega_atom_pos[:, :, None, :, :],
       phi_atom_pos[:, :, None, :, :],
       psi_atom_pos[:, :, None, :, :],
       chis_atom_pos
      ], axis=2)

  torsion_angles_mask = jnp.concatenate(
      [pre_omega_mask[:, :, None],
       phi_mask[:, :, None],
       psi_mask[:, :, None],
       chis_mask
      ], axis=2)

  torsion_frames = r3.rigids_from_3_points(
      point_on_neg_x_axis=r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 1, :]),
      origin=r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 2, :]),
      point_on_xy_plane=r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 0, :]))

  forth_atom_rel_pos = r3.rigids_mul_vecs(
      r3.invert_rigids(torsion_frames),
      r3.vecs_from_tensor(torsions_atom_pos[:, :, :, 3, :]))

  torsion_angles_sin_cos = jnp.stack(
      [forth_atom_rel_pos.z, forth_atom_rel_pos.y], axis=-1)
  torsion_angles_sin_cos /= jnp.sqrt(
      jnp.sum(jnp.square(torsion_angles_sin_cos), axis=-1, keepdims=True)
      + 1e-8)

  torsion_angles_sin_cos *= jnp.asarray(
      [1., 1., -1., 1., 1., 1., 1.])[None, None, :, None]

  chi_is_ambiguous = utils.batched_gather(
      jnp.asarray(residue_constants.chi_pi_periodic), aatype)
  mirror_torsion_angles = jnp.concatenate(
      [jnp.ones([num_batch, num_res, 3]),
       1.0 - 2.0 * chi_is_ambiguous], axis=-1)
  alt_torsion_angles_sin_cos = (
      torsion_angles_sin_cos * mirror_torsion_angles[:, :, :, None])

  if placeholder_for_undefined:
    placeholder_torsions = jnp.stack([
        jnp.ones(torsion_angles_sin_cos.shape[:-1]),
        jnp.zeros(torsion_angles_sin_cos.shape[:-1])
    ], axis=-1)
    torsion_angles_sin_cos = torsion_angles_sin_cos * torsion_angles_mask[
        ..., None] + placeholder_torsions * (1 - torsion_angles_mask[..., None])
    alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos * torsion_angles_mask[
        ..., None] + placeholder_torsions * (1 - torsion_angles_mask[..., None])

  return {
      'torsion_angles_sin_cos': torsion_angles_sin_cos,
      'alt_torsion_angles_sin_cos': alt_torsion_angles_sin_cos,
      'torsion_angles_mask': torsion_angles_mask
  }


def torsion_angles_to_frames(
    aatype: jnp.ndarray,
    backb_to_global: r3.Rigids,
    torsion_angles_sin_cos: jnp.ndarray
) -> r3.Rigids:
  """Compute rigid group frames from torsion angles.

  Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates" lines 2-10
  Jumper et al. (2021) Suppl. Alg. 25 "makeRotX"

  Args:
    aatype: aatype for each residue
    backb_to_global: Rigid transformations describing transformation from
      backbone frame to global frame.
    torsion_angles_sin_cos: sin and cosine of the 7 torsion angles
  Returns:
    Frames corresponding to all the Sidechain Rigid Transforms
  """
  assert len(aatype.shape) == 1
  assert len(backb_to_global.rot.xx.shape) == 1
  assert len(torsion_angles_sin_cos.shape) == 3
  assert torsion_angles_sin_cos.shape[1] == 7
  assert torsion_angles_sin_cos.shape[2] == 2

  m = utils.batched_gather(residue_constants.restype_rigid_group_default_frame,
                           aatype)
  default_frames = r3.rigids_from_tensor4x4(m)

  sin_angles = torsion_angles_sin_cos[..., 0]
  cos_angles = torsion_angles_sin_cos[..., 1]

  num_residues, = aatype.shape
  sin_angles = jnp.concatenate([jnp.zeros([num_residues, 1]), sin_angles],
                               axis=-1)
  cos_angles = jnp.concatenate([jnp.ones([num_residues, 1]), cos_angles],
                               axis=-1)
  zeros = jnp.zeros_like(sin_angles)
  ones = jnp.ones_like(sin_angles)

  all_rots = r3.Rots(ones, zeros, zeros,
                     zeros, cos_angles, -sin_angles,
                     zeros, sin_angles, cos_angles)

  all_frames = r3.rigids_mul_rots(default_frames, all_rots)

  chi2_frame_to_frame = jax.tree.map(lambda x: x[:, 5], all_frames)
  chi3_frame_to_frame = jax.tree.map(lambda x: x[:, 6], all_frames)
  chi4_frame_to_frame = jax.tree.map(lambda x: x[:, 7], all_frames)

  chi1_frame_to_backb = jax.tree.map(lambda x: x[:, 4], all_frames)
  chi2_frame_to_backb = r3.rigids_mul_rigids(chi1_frame_to_backb,
                                             chi2_frame_to_frame)
  chi3_frame_to_backb = r3.rigids_mul_rigids(chi2_frame_to_backb,
                                             chi3_frame_to_frame)
  chi4_frame_to_backb = r3.rigids_mul_rigids(chi3_frame_to_backb,
                                             chi4_frame_to_frame)

  def _concat_frames(xall, x5, x6, x7):
    return jnp.concatenate(
        [xall[:, 0:5], x5[:, None], x6[:, None], x7[:, None]], axis=-1)

  all_frames_to_backb = jax.tree.map(
      _concat_frames,
      all_frames,
      chi2_frame_to_backb,
      chi3_frame_to_backb,
      chi4_frame_to_backb)

  all_frames_to_global = r3.rigids_mul_rigids(
      jax.tree.map(lambda x: x[:, None], backb_to_global),
      all_frames_to_backb)

  return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    aatype: jnp.ndarray,
    all_frames_to_global: r3.Rigids
) -> r3.Vecs:
  """Put atom literature positions (atom14 encoding) in each rigid group.

  Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates" line 11

  Args:
    aatype: aatype for each residue.
    all_frames_to_global: All per residue coordinate frames.
  Returns:
    Positions of all atom coordinates in global frame.
  """

  residx_to_group_idx = utils.batched_gather(
      residue_constants.restype_atom14_to_rigid_group, aatype)
  group_mask = jax.nn.one_hot(
      residx_to_group_idx, num_classes=8)

  map_atoms_to_global = jax.tree.map(
      lambda x: jnp.sum(x[:, None, :] * group_mask, axis=-1),
      all_frames_to_global)

  lit_positions = r3.vecs_from_tensor(
      utils.batched_gather(
          residue_constants.restype_atom14_rigid_group_positions, aatype))

  pred_positions = r3.rigids_mul_vecs(map_atoms_to_global, lit_positions)

  mask = utils.batched_gather(residue_constants.restype_atom14_mask, aatype)
  pred_positions = jax.tree.map(lambda x: x * mask, pred_positions)

  return pred_positions


def extreme_ca_ca_distance_violations(
    pred_atom_positions: jnp.ndarray,
    pred_atom_mask: jnp.ndarray,
    residue_index: jnp.ndarray,
    max_angstrom_tolerance=1.5
    ) -> jnp.ndarray:
  """Counts residues whose Ca is a large distance from its neighbour.

  Measures the fraction of CA-CA pairs between consecutive amino acids that are
  more than 'max_angstrom_tolerance' apart.

  Args:
    pred_atom_positions: Atom positions in atom37/14 representation
    pred_atom_mask: Atom mask in atom37/14 representation
    residue_index: Residue index for given amino acid, this is assumed to be
      monotonically increasing.
    max_angstrom_tolerance: Maximum distance allowed to not count as violation.
  Returns:
    Fraction of consecutive CA-CA pairs with violation.
  """
  this_ca_pos = pred_atom_positions[:-1, 1, :]
  this_ca_mask = pred_atom_mask[:-1, 1]
  next_ca_pos = pred_atom_positions[1:, 1, :]
  next_ca_mask = pred_atom_mask[1:, 1]
  has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).astype(
      jnp.float32)
  ca_ca_distance = jnp.sqrt(
      1e-6 + jnp.sum(squared_difference(this_ca_pos, next_ca_pos), axis=-1))
  violations = (ca_ca_distance -
                residue_constants.ca_ca) > max_angstrom_tolerance
  mask = this_ca_mask * next_ca_mask * has_no_gap_mask
  return utils.mask_mean(mask=mask, value=violations)


def between_residue_bond_loss(
    pred_atom_positions: jnp.ndarray,
    pred_atom_mask: jnp.ndarray,
    residue_index: jnp.ndarray,
    aatype: jnp.ndarray,
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0
) -> Dict[str, jnp.ndarray]:
  """Flat-bottom loss to penalize structural violations between residues.

  This is a loss penalizing any violation of the geometry around the peptide
  bond between consecutive amino acids. This loss corresponds to
  Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44, 45.

  Args:
    pred_atom_positions: Atom positions in atom37/14 representation
    pred_atom_mask: Atom mask in atom37/14 representation
    residue_index: Residue index for given amino acid, this is assumed to be
      monotonically increasing.
    aatype: Amino acid type of given residue
    tolerance_factor_soft: soft tolerance factor measured in standard deviations
      of pdb distributions
    tolerance_factor_hard: hard tolerance factor measured in standard deviations
      of pdb distributions

  Returns:
    Dict containing:
      * 'c_n_loss_mean': Loss for peptide bond length violations
      * 'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned
          by CA, C, N
      * 'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned
          by C, N, CA
      * 'per_residue_loss_sum': sum of all losses for each residue
      * 'per_residue_violation_mask': mask denoting all residues with violation
          present.
  """
  assert len(pred_atom_positions.shape) == 3
  assert len(pred_atom_mask.shape) == 2
  assert len(residue_index.shape) == 1
  assert len(aatype.shape) == 1

  this_ca_pos = pred_atom_positions[:-1, 1, :]
  this_ca_mask = pred_atom_mask[:-1, 1]
  this_c_pos = pred_atom_positions[:-1, 2, :]
  this_c_mask = pred_atom_mask[:-1, 2]
  next_n_pos = pred_atom_positions[1:, 0, :]
  next_n_mask = pred_atom_mask[1:, 0]
  next_ca_pos = pred_atom_positions[1:, 1, :]
  next_ca_mask = pred_atom_mask[1:, 1]
  has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).astype(
      jnp.float32)

  c_n_bond_length = jnp.sqrt(
      1e-6 + jnp.sum(squared_difference(this_c_pos, next_n_pos), axis=-1))

  next_is_proline = (
      aatype[1:] == residue_constants.resname_to_idx['PRO']).astype(jnp.float32)
  gt_length = (
      (1. - next_is_proline) * residue_constants.between_res_bond_length_c_n[0]
      + next_is_proline * residue_constants.between_res_bond_length_c_n[1])
  gt_stddev = (
      (1. - next_is_proline) *
      residue_constants.between_res_bond_length_stddev_c_n[0] +
      next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1])
  c_n_bond_length_error = jnp.sqrt(1e-6 +
                                   jnp.square(c_n_bond_length - gt_length))
  c_n_loss_per_residue = jax.nn.relu(
      c_n_bond_length_error - tolerance_factor_soft * gt_stddev)
  mask = this_c_mask * next_n_mask * has_no_gap_mask
  c_n_loss = jnp.sum(mask * c_n_loss_per_residue) / (jnp.sum(mask) + 1e-6)
  c_n_violation_mask = mask * (
      c_n_bond_length_error > (tolerance_factor_hard * gt_stddev))

  ca_c_bond_length = jnp.sqrt(1e-6 + jnp.sum(
      squared_difference(this_ca_pos, this_c_pos), axis=-1))
  n_ca_bond_length = jnp.sqrt(1e-6 + jnp.sum(
      squared_difference(next_n_pos, next_ca_pos), axis=-1))

  c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[:, None]
  c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[:, None]
  n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[:, None]

  ca_c_n_cos_angle = jnp.sum(c_ca_unit_vec * c_n_unit_vec, axis=-1)
  gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
  gt_stddev = residue_constants.between_res_bond_length_stddev_c_n[0]
  ca_c_n_cos_angle_error = jnp.sqrt(
      1e-6 + jnp.square(ca_c_n_cos_angle - gt_angle))
  ca_c_n_loss_per_residue = jax.nn.relu(
      ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
  mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
  ca_c_n_loss = jnp.sum(mask * ca_c_n_loss_per_residue) / (jnp.sum(mask) + 1e-6)
  ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error >
                                  (tolerance_factor_hard * gt_stddev))

  c_n_ca_cos_angle = jnp.sum((-c_n_unit_vec) * n_ca_unit_vec, axis=-1)
  gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
  gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
  c_n_ca_cos_angle_error = jnp.sqrt(
      1e-6 + jnp.square(c_n_ca_cos_angle - gt_angle))
  c_n_ca_loss_per_residue = jax.nn.relu(
      c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
  mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
  c_n_ca_loss = jnp.sum(mask * c_n_ca_loss_per_residue) / (jnp.sum(mask) + 1e-6)
  c_n_ca_violation_mask = mask * (
      c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev))

  per_residue_loss_sum = (c_n_loss_per_residue +
                          ca_c_n_loss_per_residue +
                          c_n_ca_loss_per_residue)
  per_residue_loss_sum = 0.5 * (jnp.pad(per_residue_loss_sum, [[0, 1]]) +
                                jnp.pad(per_residue_loss_sum, [[1, 0]]))

  violation_mask = jnp.max(
      jnp.stack([c_n_violation_mask,
                 ca_c_n_violation_mask,
                 c_n_ca_violation_mask]), axis=0)
  violation_mask = jnp.maximum(
      jnp.pad(violation_mask, [[0, 1]]),
      jnp.pad(violation_mask, [[1, 0]]))

  return {'c_n_loss_mean': c_n_loss,
          'ca_c_n_loss_mean': ca_c_n_loss,
          'c_n_ca_loss_mean': c_n_ca_loss,
          'per_residue_loss_sum': per_residue_loss_sum,
          'per_residue_violation_mask': violation_mask
         }


def between_residue_clash_loss(
    atom14_pred_positions: jnp.ndarray,
    atom14_atom_exists: jnp.ndarray,
    atom14_atom_radius: jnp.ndarray,
    residue_index: jnp.ndarray,
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5
) -> Dict[str, jnp.ndarray]:
  """Loss to penalize steric clashes between residues.

  This is a loss penalizing any steric clashes due to non bonded atoms in
  different peptides coming too close. This loss corresponds to the part with
  different residues of
  Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

  Args:
    atom14_pred_positions: Predicted positions of atoms in
      global prediction frame
    atom14_atom_exists: Mask denoting whether atom at positions exists for given
      amino acid type
    atom14_atom_radius: Van der Waals radius for each atom.
    residue_index: Residue index for given amino acid.
    overlap_tolerance_soft: Soft tolerance factor.
    overlap_tolerance_hard: Hard tolerance factor.

  Returns:
    Dict containing:
      * 'mean_loss': average clash loss
      * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
      * 'per_atom_clash_mask': mask whether atom clashes with any other atom
          shape (N, 14)
  """
  assert len(atom14_pred_positions.shape) == 3
  assert len(atom14_atom_exists.shape) == 2
  assert len(atom14_atom_radius.shape) == 2
  assert len(residue_index.shape) == 1

  dists = jnp.sqrt(1e-10 + jnp.sum(
      squared_difference(
          atom14_pred_positions[:, None, :, None, :],
          atom14_pred_positions[None, :, None, :, :]),
      axis=-1))

  dists_mask = (atom14_atom_exists[:, None, :, None] *
                atom14_atom_exists[None, :, None, :])

  dists_mask *= (
      residue_index[:, None, None, None] < residue_index[None, :, None, None])

  c_one_hot = jax.nn.one_hot(2, num_classes=14)
  n_one_hot = jax.nn.one_hot(0, num_classes=14)
  neighbour_mask = ((residue_index[:, None, None, None] +
                     1) == residue_index[None, :, None, None])
  c_n_bonds = neighbour_mask * c_one_hot[None, None, :,
                                         None] * n_one_hot[None, None, None, :]
  dists_mask *= (1. - c_n_bonds)

  cys_sg_idx = residue_constants.restype_name_to_atom14_names['CYS'].index('SG')
  cys_sg_one_hot = jax.nn.one_hot(cys_sg_idx, num_classes=14)
  disulfide_bonds = (cys_sg_one_hot[None, None, :, None] *
                     cys_sg_one_hot[None, None, None, :])
  dists_mask *= (1. - disulfide_bonds)

  dists_lower_bound = dists_mask * (atom14_atom_radius[:, None, :, None] +
                                    atom14_atom_radius[None, :, None, :])

  dists_to_low_error = dists_mask * jax.nn.relu(
      dists_lower_bound - overlap_tolerance_soft - dists)

  mean_loss = (jnp.sum(dists_to_low_error)
               / (1e-6 + jnp.sum(dists_mask)))

  per_atom_loss_sum = (jnp.sum(dists_to_low_error, axis=[0, 2]) +
                       jnp.sum(dists_to_low_error, axis=[1, 3]))

  clash_mask = dists_mask * (
      dists < (dists_lower_bound - overlap_tolerance_hard))

  per_atom_clash_mask = jnp.maximum(
      jnp.max(clash_mask, axis=[0, 2]),
      jnp.max(clash_mask, axis=[1, 3]))

  return {'mean_loss': mean_loss,
          'per_atom_loss_sum': per_atom_loss_sum,
          'per_atom_clash_mask': per_atom_clash_mask
         }


def within_residue_violations(
    atom14_pred_positions: jnp.ndarray,
    atom14_atom_exists: jnp.ndarray,
    atom14_dists_lower_bound: jnp.ndarray,
    atom14_dists_upper_bound: jnp.ndarray,
    tighten_bounds_for_loss=0.0,
) -> Dict[str, jnp.ndarray]:
  """Loss to penalize steric clashes within residues.

  This is a loss penalizing any steric violations or clashes of non-bonded atoms
  in a given peptide. This loss corresponds to the part with
  the same residues of
  Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

  Args:
    atom14_pred_positions: Predicted positions of atoms in
      global prediction frame
    atom14_atom_exists: Mask denoting whether atom at positions exists for given
      amino acid type
    atom14_dists_lower_bound: Lower bound on allowed distances.
    atom14_dists_upper_bound: Upper bound on allowed distances
    tighten_bounds_for_loss: Extra factor to tighten loss

  Returns:
    Dict containing:
      * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
      * 'per_atom_clash_mask': mask whether atom clashes with any other atom
          shape (N, 14)
  """
  assert len(atom14_pred_positions.shape) == 3
  assert len(atom14_atom_exists.shape) == 2
  assert len(atom14_dists_lower_bound.shape) == 3
  assert len(atom14_dists_upper_bound.shape) == 3

  dists_masks = (1. - jnp.eye(14, 14)[None])
  dists_masks *= (atom14_atom_exists[:, :, None] *
                  atom14_atom_exists[:, None, :])

  dists = jnp.sqrt(1e-10 + jnp.sum(
      squared_difference(
          atom14_pred_positions[:, :, None, :],
          atom14_pred_positions[:, None, :, :]),
      axis=-1))

  dists_to_low_error = jax.nn.relu(
      atom14_dists_lower_bound + tighten_bounds_for_loss - dists)
  dists_to_high_error = jax.nn.relu(
      dists - (atom14_dists_upper_bound - tighten_bounds_for_loss))
  loss = dists_masks * (dists_to_low_error + dists_to_high_error)

  per_atom_loss_sum = (jnp.sum(loss, axis=1) +
                       jnp.sum(loss, axis=2))

  violations = dists_masks * ((dists < atom14_dists_lower_bound) |
                              (dists > atom14_dists_upper_bound))

  per_atom_violations = jnp.maximum(
      jnp.max(violations, axis=1), jnp.max(violations, axis=2))

  return {'per_atom_loss_sum': per_atom_loss_sum,
          'per_atom_violations': per_atom_violations
         }


def find_optimal_renaming(
    atom14_gt_positions: jnp.ndarray,
    atom14_alt_gt_positions: jnp.ndarray,
    atom14_atom_is_ambiguous: jnp.ndarray,
    atom14_gt_exists: jnp.ndarray,
    atom14_pred_positions: jnp.ndarray,
    atom14_atom_exists: jnp.ndarray,
) -> jnp.ndarray:
  """Find optimal renaming for ground truth that maximizes LDDT.

  Jumper et al. (2021) Suppl. Alg. 26
  "renameSymmetricGroundTruthAtoms" lines 1-5

  Args:
    atom14_gt_positions: Ground truth positions in global frame of ground truth.
    atom14_alt_gt_positions: Alternate ground truth positions in global frame of
      ground truth with coordinates of ambiguous atoms swapped relative to
      'atom14_gt_positions'.
    atom14_atom_is_ambiguous: Mask denoting whether atom is among ambiguous
      atoms, see Jumper et al. (2021) Suppl. Table 3
    atom14_gt_exists: Mask denoting whether atom at positions exists in ground
      truth.
    atom14_pred_positions: Predicted positions of atoms in
      global prediction frame
    atom14_atom_exists: Mask denoting whether atom at positions exists for given
      amino acid type

  Returns:
    Float array of shape [N] with 1. where atom14_alt_gt_positions is closer to
    prediction and 0. otherwise
  """
  assert len(atom14_gt_positions.shape) == 3
  assert len(atom14_alt_gt_positions.shape) == 3
  assert len(atom14_atom_is_ambiguous.shape) == 2
  assert len(atom14_gt_exists.shape) == 2
  assert len(atom14_pred_positions.shape) == 3
  assert len(atom14_atom_exists.shape) == 2

  pred_dists = jnp.sqrt(1e-10 + jnp.sum(
      squared_difference(
          atom14_pred_positions[:, None, :, None, :],
          atom14_pred_positions[None, :, None, :, :]),
      axis=-1))

  gt_dists = jnp.sqrt(1e-10 + jnp.sum(
      squared_difference(
          atom14_gt_positions[:, None, :, None, :],
          atom14_gt_positions[None, :, None, :, :]),
      axis=-1))
  alt_gt_dists = jnp.sqrt(1e-10 + jnp.sum(
      squared_difference(
          atom14_alt_gt_positions[:, None, :, None, :],
          atom14_alt_gt_positions[None, :, None, :, :]),
      axis=-1))

  lddt = jnp.sqrt(1e-10 + squared_difference(pred_dists, gt_dists))
  alt_lddt = jnp.sqrt(1e-10 + squared_difference(pred_dists, alt_gt_dists))

  mask = (atom14_gt_exists[:, None, :, None] *
          atom14_atom_is_ambiguous[:, None, :, None] *
          atom14_gt_exists[None, :, None, :] *
          (1. - atom14_atom_is_ambiguous[None, :, None, :]))

  per_res_lddt = jnp.sum(mask * lddt, axis=[1, 2, 3])
  alt_per_res_lddt = jnp.sum(mask * alt_lddt, axis=[1, 2, 3])

  alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).astype(jnp.float32)

  return alt_naming_is_better


def frame_aligned_point_error(
    pred_frames: r3.Rigids,
    target_frames: r3.Rigids,
    frames_mask: jnp.ndarray,
    pred_positions: r3.Vecs,
    target_positions: r3.Vecs,
    positions_mask: jnp.ndarray,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    epsilon=1e-4) -> jnp.ndarray:
  """Measure point error under different alignments.

  Jumper et al. (2021) Suppl. Alg. 28 "computeFAPE"

  Computes error between two structures with B points under A alignments derived
  from the given pairs of frames.
  Args:
    pred_frames: num_frames reference frames for 'pred_positions'.
    target_frames: num_frames reference frames for 'target_positions'.
    frames_mask: Mask for frame pairs to use.
    pred_positions: num_positions predicted positions of the structure.
    target_positions: num_positions target positions of the structure.
    positions_mask: Mask on which positions to score.
    length_scale: length scale to divide loss by.
    l1_clamp_distance: Distance cutoff on error beyond which gradients will
      be zero.
    epsilon: small value used to regularize denominator for masked average.
  Returns:
    Masked Frame Aligned Point Error.
  """
  assert pred_frames.rot.xx.ndim == 1
  assert target_frames.rot.xx.ndim == 1
  assert frames_mask.ndim == 1, frames_mask.ndim
  assert pred_positions.x.ndim == 1
  assert target_positions.x.ndim == 1
  assert positions_mask.ndim == 1

  local_pred_pos = r3.rigids_mul_vecs(
      jax.tree.map(lambda r: r[:, None], r3.invert_rigids(pred_frames)),
      jax.tree.map(lambda x: x[None, :], pred_positions))

  local_target_pos = r3.rigids_mul_vecs(
      jax.tree.map(lambda r: r[:, None], r3.invert_rigids(target_frames)),
      jax.tree.map(lambda x: x[None, :], target_positions))

  error_dist = jnp.sqrt(
      r3.vecs_squared_distance(local_pred_pos, local_target_pos)
      + epsilon)

  if l1_clamp_distance:
    error_dist = jnp.clip(error_dist, 0, l1_clamp_distance)

  normed_error = error_dist / length_scale
  normed_error *= jnp.expand_dims(frames_mask, axis=-1)
  normed_error *= jnp.expand_dims(positions_mask, axis=-2)

  normalization_factor = (
      jnp.sum(frames_mask, axis=-1) *
      jnp.sum(positions_mask, axis=-1))
  return (jnp.sum(normed_error, axis=(-2, -1)) /
          (epsilon + normalization_factor))


def _make_renaming_matrices():
  """Matrices to map atoms to symmetry partners in ambiguous case."""
  restype_3 = [
      residue_constants.restype_1to3[res] for res in residue_constants.restypes
  ]
  restype_3 += ['UNK']
  all_matrices = {res: np.eye(14, dtype=np.float32) for res in restype_3}
  for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
    correspondences = np.arange(14)
    for source_atom_swap, target_atom_swap in swap.items():
      source_index = residue_constants.restype_name_to_atom14_names[
          resname].index(source_atom_swap)
      target_index = residue_constants.restype_name_to_atom14_names[
          resname].index(target_atom_swap)
      correspondences[source_index] = target_index
      correspondences[target_index] = source_index
      renaming_matrix = np.zeros((14, 14), dtype=np.float32)
      for index, correspondence in enumerate(correspondences):
        renaming_matrix[index, correspondence] = 1.
    all_matrices[resname] = renaming_matrix.astype(np.float32)
  renaming_matrices = np.stack([all_matrices[restype] for restype in restype_3])
  return renaming_matrices


RENAMING_MATRICES = _make_renaming_matrices()


def get_alt_atom14(aatype, positions, mask):
  """Get alternative atom14 positions.

  Constructs renamed atom positions for ambiguous residues.

  Jumper et al. (2021) Suppl. Table 3 "Ambiguous atom names due to 180 degree-
  rotation-symmetry"

  Args:
    aatype: Amino acid at given position
    positions: Atom positions as r3.Vecs in atom14 representation, (N, 14)
    mask: Atom masks in atom14 representation, (N, 14)
  Returns:
    renamed atom positions, renamed atom mask
  """
  renaming_transform = utils.batched_gather(
      jnp.asarray(RENAMING_MATRICES), aatype)

  positions = jax.tree.map(lambda x: x[:, :, None], positions)
  alternative_positions = jax.tree.map(
      lambda x: jnp.sum(x, axis=1), positions * renaming_transform)

  alternative_mask = jnp.sum(mask[..., None] * renaming_transform, axis=1)

  return alternative_positions, alternative_mask
