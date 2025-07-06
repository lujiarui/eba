# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under Creative Commons Attribution-NonCommercial 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import functools
import os
import re
from collections import defaultdict
from typing import Mapping, Sequence

import biotite.structure as struc
import numpy as np
import torch
from biotite.structure import AtomArray
from biotite.structure.io import pdbx

from protenix.utils.file_io import load_gzip_pickle, read_indices_csv



def remove_numbers(s):
    """
    Remove numbers from a string.

    Args:
        s (string): input string

    Returns:
        string: a string with numbers removed.
    """
    return re.sub(r"\d+", "", s)


def get_asym_id(index: int):
    """Generate asym id same to mmcif file from index.
    Args:
        index (int): index of asym unit, start from 1
    Return:
        asym id, A,B,..Z,AA,BA,...ZZ,AAA,BAA,...ZZZ,...
    """
    index -= 1
    lower_bound = 0
    upper_bound = 26
    digits = 1
    while True:
        if index < upper_bound:
            break
        digits += 1
        upper_bound += 26**digits
        lower_bound += 26 ** (digits - 1)
    index -= lower_bound
    chars = []
    for i in range(digits):
        chars.append(chr(ord("A") + index % 26))
        index //= 26
    return "".join(chars)


def replace_elem_by_mapping_dict(input_array: np.ndarray, mapping: dict) -> np.ndarray:
    """
    Replace each element in input_array with the value in mapping.

    Args:
        input_array: np.ndarray
        mapping: a mapping dict

    Returns:
        np.ndarray: a new array with the same shape as input_array.
    """
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))

    sidx = k.argsort()  # k,v from approach #1

    k = k[sidx]
    v = v[sidx]

    idx = np.searchsorted(k, input_array.ravel()).reshape(input_array.shape)
    idx[idx == len(k)] = 0
    mask = k[idx] == input_array
    out = np.where(mask, v[idx], 0)
    return out


@functools.lru_cache
def parse_pdb_cluster_file_to_dict(
    cluster_file: str, remove_uniprot: bool = True
) -> dict[str, tuple]:
    """parse PDB cluster file, and return a pandas dataframe
    example cluster file:
    https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt

    Args:
        cluster_file (str): cluster_file path
    Returns:
        dict(str, tuple(str, str)): {pdb_id}_{entity_id} --> [cluster_id, cluster_size]
    """
    # TODO: cluster the data by ourself only from wwPDB
    pdb_cluster_dict = {}
    with open(cluster_file) as f:
        for line in f:
            pdb_clusters = []
            for ids in line.strip().split():
                if remove_uniprot:
                    if ids.startswith("AF_") or ids.startswith("MA_"):
                        continue
                pdb_clusters.append(ids)
            cluster_size = len(pdb_clusters)
            if cluster_size == 0:
                continue
            # use first member as cluster id.
            cluster_id = f"pdb_cluster_{pdb_clusters[0]}"
            for ids in pdb_clusters:
                pdb_cluster_dict[ids.lower()] = (cluster_id, cluster_size)
    return pdb_cluster_dict




def int_to_letters(n: int) -> str:
    """
    Convert int to letters.
    Useful for converting chain index to label_asym_id.

    Args:
        n (int): int number
    Returns:
        str: letters. e.g. 1 -> A, 2 -> B, 27 -> AA, 28 -> AB
    """
    result = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result


def get_inter_residue_bonds(atom_array: AtomArray) -> np.ndarray:
    """get inter residue bonds by checking chain_id and res_id

    Args:
        atom_array (AtomArray): Biotite AtomArray, must have chain_id and res_id

    Returns:
        np.ndarray: inter residue bonds, shape = (n,2)
    """
    if atom_array.bonds is None:
        return []
    idx_i = atom_array.bonds._bonds[:, 0]
    idx_j = atom_array.bonds._bonds[:, 1]
    chain_id_diff = atom_array.chain_id[idx_i] != atom_array.chain_id[idx_j]
    res_id_diff = atom_array.res_id[idx_i] != atom_array.res_id[idx_j]
    diff_mask = chain_id_diff | res_id_diff
    inter_residue_bonds = atom_array.bonds._bonds[diff_mask]
    inter_residue_bonds = inter_residue_bonds[:, :2]  # remove bond type
    return inter_residue_bonds


def get_starts_by(
    atom_array: AtomArray, by_annot: str, add_exclusive_stop=False
) -> np.ndarray:
    """get mask of env_atoms within distance to query_atoms

    Args:
        query_atoms (AtomArray): query_atoms, query_atoms not in env_atoms
        env_atoms (AtomArray):
        distance (float): within distance (Å), eg: 5

    Returns:
        array(bool): a mask of env_atoms, length = len(env_atoms)
    """
    cell_list = struc.CellList(env_atoms, cell_size=5)
    mask_2d = cell_list.get_atoms(query_atoms.coord, radius=distance, as_mask=True)
    return mask_2d.any(axis=0)


def get_inter_residue_bonds(atom_array: AtomArray) -> np.ndarray:
    """get inter residue bonds by checking chain_id and res_id

    Args:
        atom_array (AtomArray): Biotite AtomArray, must have chain_id and res_id

    Returns:
        np.ndarray: inter residue bonds, shape = (n,2)
    """
    if atom_array.bonds is None:
        return []
    idx_i = atom_array.bonds._bonds[:, 0]
    idx_j = atom_array.bonds._bonds[:, 1]
    chain_id_diff = atom_array.chain_id[idx_i] != atom_array.chain_id[idx_j]
    res_id_diff = atom_array.res_id[idx_i] != atom_array.res_id[idx_j]
    diff_mask = chain_id_diff | res_id_diff
    inter_residue_bonds = atom_array.bonds._bonds[diff_mask]
    inter_residue_bonds = inter_residue_bonds[:, :2]  # remove bond type
    return inter_residue_bonds


def get_starts_by(atom_array, by_annot, add_exclusive_stop=False):
    """get start indices by given annotation in an AtomArray

    Args:
        atom_array (AtomArray): Biotite AtomArray
        by_annot (str): annotation to group by, eg: 'chain_id', 'res_id', 'res_name'
        add_exclusive_stop (bool, optional): add exclusive stop (len(atom_array)). Defaults to False.

    Returns:
        np.ndarray: start indices of each group, shape = (n,), eg: [0, 10, 20, 30, 40]
    """
    annot = getattr(atom_array, by_annot)
    # If annotation change, a new start
    annot_change_mask = annot[1:] != annot[:-1]

    # Convert mask to indices
    # Add 1, to shift the indices from the end of a residue
    # to the start of a new residue
    starts = np.where(annot_change_mask)[0] + 1

    # The first start is not included yet -> Insert '[0]'
    if add_exclusive_stop:
        return np.concatenate(([0], starts, [atom_array.array_length()]))
    else:
        return np.concatenate(([0], starts))


def atom_select(atom_array: AtomArray, select_dict: dict, as_mask=False) -> np.ndarray:
    """return index of atom_array that match select_dict

    Args:
        atom_array (AtomArray): Biotite AtomArray
        select_dict (dict): select dict, eg: {'element': 'C'}
        as_mask (bool, optional): return mask of atom_array. Defaults to False.

    Returns:
        np.ndarray: index of atom_array that match select_dict
    """
    mask = np.ones(len(atom_array), dtype=bool)
    for k, v in select_dict.items():
        mask = mask & (getattr(atom_array, k) == v)
    if as_mask:
        return mask
    else:
        return np.where(mask)[0]


def get_ligand_polymer_bond_mask(
    atom_array: AtomArray, lig_include_ions=False
) -> np.ndarray:
    """
    Ref AlphaFold3 SI Chapter 3.7.1.
    Get bonds between the bonded ligand and its parent chain.

    Args:
        atom_array (AtomArray): biotite atom array object.
        lig_include_ions (bool): whether to include ions in the ligand.

    Returns:
        np.ndarray: bond records between the bonded ligand and its parent chain.
                    e.g. np.array([[atom1, atom2, bond_order]...])
    """
    if not lig_include_ions:
        # bonded ligand exclude ions
        unique_chain_id, counts = np.unique(
            atom_array.label_asym_id, return_counts=True
        )
        chain_id_to_count_map = dict(zip(unique_chain_id, counts))
        ions_mask = np.array(
            [
                chain_id_to_count_map[label_asym_id] == 1
                for label_asym_id in atom_array.label_asym_id
            ]
        )

        lig_mask = (atom_array.mol_type == "ligand") & ~ions_mask
    else:
        lig_mask = atom_array.mol_type == "ligand"

    # identify polymer by mol_type (protein, rna, dna, ligand)
    polymer_mask = np.isin(atom_array.mol_type, ["protein", "rna", "dna"])

    idx_i = atom_array.bonds._bonds[:, 0]
    idx_j = atom_array.bonds._bonds[:, 1]

    lig_polymer_bond_indices = np.where(
        (lig_mask[idx_i] & polymer_mask[idx_j])
        | (lig_mask[idx_j] & polymer_mask[idx_i])
    )[0]
    if lig_polymer_bond_indices.size == 0:
        # no ligand-polymer bonds
        lig_polymer_bonds = np.empty((0, 3)).astype(int)
    else:
        lig_polymer_bonds = atom_array.bonds._bonds[
            lig_polymer_bond_indices
        ]  # np.array([[atom1, atom2, bond_order]...])
    return lig_polymer_bonds


@functools.lru_cache
def parse_pdb_cluster_file_to_dict(
    cluster_file: str, remove_uniprot: bool = True
) -> dict[str, tuple]:
    """parse PDB cluster file, and return a pandas dataframe
    example cluster file:
    https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt

    Args:
        cluster_file (str): cluster_file path
    Returns:
        dict(str, tuple(str, str)): {pdb_id}_{entity_id} --> [cluster_id, cluster_size]
    """
    pdb_cluster_dict = {}
    with open(cluster_file) as f:
        for line in f:
            pdb_clusters = []
            for ids in line.strip().split():
                if remove_uniprot:
                    if ids.startswith("AF_") or ids.startswith("MA_"):
                        continue
                pdb_clusters.append(ids)
            cluster_size = len(pdb_clusters)
            if cluster_size == 0:
                continue
            # use first member as cluster id.
            cluster_id = f"pdb_cluster_{pdb_clusters[0]}"
            for ids in pdb_clusters:
                pdb_cluster_dict[ids.lower()] = (cluster_id, cluster_size)
    return pdb_cluster_dict


def get_clean_data(atom_array: AtomArray) -> AtomArray:
    """
    Removes unresolved atoms from the AtomArray.

    Args:
        atom_array (AtomArray): The input AtomArray containing atoms.

    Returns:
        AtomArray: A new AtomArray with unresolved atoms removed.
    """
    atom_array_wo_unresol = atom_array.copy()
    atom_array_wo_unresol = atom_array[atom_array.is_resolved]
    return atom_array_wo_unresol


def save_atoms_to_cif(
    output_cif_file: str, atom_array: AtomArray, entity_poly_type, pdb_id
) -> None:
    """
    Save atom array data to a CIF file.

    Args:
        output_cif_file (str): the output path for saving atom array in cif
        atom_array (AtomArray): the atom array to be saved
    """
    cifwriter = CIFWriter(atom_array, entity_poly_type)
    cifwriter.save_to_cif(
        output_path=output_cif_file,
        entry_id=pdb_id,
        include_bonds=True,
    )


def get_raw_atom_array(bioassembly_dict_fpath):
    bioassembly_dict = load_gzip_pickle(bioassembly_dict_fpath)
    atom_array = bioassembly_dict["atom_array"]
    entity_poly_type = bioassembly_dict["entity_poly_type"]
    # Note: 'charge' info isn't used but causes OST DockQ parser errors. Temp set to 0 for compatibility.
    atom_array.charge = np.zeros(len(atom_array.charge))
    return atom_array, entity_poly_type


def save_structure_cif(
    atom_array, pred_coordinate, output_fpath, entity_poly_type, pdb_id
):
    pred_atom_array = copy.deepcopy(atom_array)
    pred_pose = pred_coordinate.cpu().numpy()
    pred_atom_array.coord = pred_pose
    # save_atoms_to_cif(
    #     output_fpath,
    #     pred_atom_array,
    #     entity_poly_type,
    #     pdb_id,
    # )
    # save pred coordinates wo unresolved atoms
    if hasattr(atom_array, "is_resolved"):
        pred_atom_array_wo_unresol = get_clean_data(pred_atom_array)
        save_atoms_to_cif(
            output_fpath.replace(".cif", "_wounresol.cif"),
            pred_atom_array_wo_unresol,
            entity_poly_type,
            pdb_id,
        )


def get_lig_lig_bonds(
    atom_array: AtomArray, lig_include_ions: bool = False
) -> np.ndarray:
    """
    Get all inter-ligand bonds in order to create "token_bonds".

    Args:
        atom_array (AtomArray): biotite AtomArray object with "mol_type" attribute.
        lig_include_ions (bool, optional): . Defaults to False.

    Returns:
        np.ndarray: inter-ligand bonds, e.g. np.array([[atom1, atom2, bond_order]...])
    """
    if not lig_include_ions:
        # bonded ligand exclude ions
        unique_chain_id, counts = np.unique(
            atom_array.label_asym_id, return_counts=True
        )
        chain_id_to_count_map = dict(zip(unique_chain_id, counts))
        ions_mask = np.array(
            [
                chain_id_to_count_map[label_asym_id] == 1
                for label_asym_id in atom_array.label_asym_id
            ]
        )

        lig_mask = (atom_array.mol_type == "ligand") & ~ions_mask
    else:
        lig_mask = atom_array.mol_type == "ligand"

    chain_res_id = np.vstack((atom_array.label_asym_id, atom_array.res_id)).T
    idx_i = atom_array.bonds._bonds[:, 0]
    idx_j = atom_array.bonds._bonds[:, 1]

    ligand_ligand_bond_indices = np.where(
        (lig_mask[idx_i] & lig_mask[idx_j])
        & np.any(chain_res_id[idx_i] != chain_res_id[idx_j], axis=1)
    )[0]

    if ligand_ligand_bond_indices.size == 0:
        # no ligand-polymer bonds
        lig_polymer_bonds = np.empty((0, 3)).astype(int)
    else:
        lig_polymer_bonds = atom_array.bonds._bonds[ligand_ligand_bond_indices]
    return lig_polymer_bonds


def superimpose(
    fixed: AtomArray,
    mobile: AtomArray,
    fixed_asym_id: str,
    mobile_asym_id: str,
) -> tuple[AtomArray, struc.AffineTransformation]:
    """
    Superimpose mobile atom_array onto fixed atom_array based on label_asym_id.
    if fixed atom_array have attr `is_resolved==True`, only these atoms are used for superimposition.

    Args:
        fixed (AtomArray): fixed atom_array
        mobile (AtomArray): mobile atom_array
        fixed_asym_id (str): label_asym_id of fixed atom_array
        mobile_asym_id (str): label_asym_id of mobile atom_array

    Returns:
        AtomArray: superimposed mobile atom_array
        AffineTransformation:This object contains the affine transformation(s) that were
        applied on `mobile`.
        `AffineTransformation.apply()` can be used to transform
        another AtomArray in the same way.
    """
    assert hasattr(fixed, "label_asym_id"), "fixed atom_array must have label_asym_id"
    assert hasattr(mobile, "label_asym_id"), "mobile atom_array must have label_asym_id"

    assert np.any(
        fixed.label_asym_id == fixed_asym_id
    ), f"{fixed_asym_id=} not in fixed!"
    assert np.any(
        mobile.label_asym_id == mobile_asym_id
    ), f"{mobile_asym_id=} not in mobile!"
    fixed_filtered = fixed[fixed.label_asym_id == fixed_asym_id]
    mobile_filtered = mobile[mobile.label_asym_id == mobile_asym_id]

    assert len(fixed_filtered) == len(
        mobile_filtered
    ), f"{len(fixed_filtered)=}, {len(mobile_filtered)=}"
    diff_idx = np.where(fixed_filtered.atom_name != mobile_filtered.atom_name)[0]
    assert (
        fixed_filtered.atom_name == mobile_filtered.atom_name
    ).all(), f"atom_name between aligned chains must be the same:\n {fixed_filtered[diff_idx]}\n {mobile_filtered[diff_idx]} "

    if hasattr(fixed_filtered, "is_resolved"):
        mobile_filtered = mobile_filtered[fixed_filtered.is_resolved]
        fixed_filtered = fixed_filtered[fixed_filtered.is_resolved]

    fitted, transform = struc.superimpose(fixed_filtered, mobile_filtered)

    return transform.apply(mobile), transform


class CIFWriter:
    """
    Write AtomArray to cif.
    """

    def __init__(self, atom_array: AtomArray, entity_poly_type: dict[str, str] = None):
        """
        Args:
            atom_array (AtomArray): Biotite AtomArray object.
            entity_poly_type (dict[str, str], optional): A dict of label_entity_id to entity_poly_type. Defaults to None.
                                                         If None, "the entity_poly" and "entity_poly_seq" will not be written to the cif.
        """
        self.atom_array = atom_array
        self.entity_poly_type = entity_poly_type

    def _get_entity_poly_and_entity_poly_seq_block(self):
        entity_poly = defaultdict(list)
        for entity_id, entity_type in self.entity_poly_type.items():
            label_asym_ids = np.unique(
                self.atom_array.label_asym_id[
                    self.atom_array.label_entity_id == entity_id
                ]
            )
            label_asym_ids_str = ",".join(label_asym_ids)

            if label_asym_ids_str == "":
                # The entity not in current atom_array
                continue

            entity_poly["entity_id"].append(entity_id)
            entity_poly["pdbx_strand_id"].append(label_asym_ids_str)
            entity_poly["type"].append(entity_type)

        entity_poly_seq = defaultdict(list)
        for entity_id, label_asym_ids_str in zip(
            entity_poly["entity_id"], entity_poly["pdbx_strand_id"]
        ):
            first_label_asym_id = label_asym_ids_str.split(",")[0]
            first_asym_chain = self.atom_array[
                self.atom_array.label_asym_id == first_label_asym_id
            ]
            chain_starts = struc.get_chain_starts(
                first_asym_chain, add_exclusive_stop=True
            )
            asym_chain = first_asym_chain[
                chain_starts[0] : chain_starts[1]
            ]  # ensure the asym chain is a single chain

            res_starts = struc.get_residue_starts(asym_chain, add_exclusive_stop=False)
            asym_chain_entity_id = asym_chain[res_starts].label_entity_id.tolist()
            asym_chain_hetero = [
                "n" if not i else "y" for i in asym_chain[res_starts].hetero
            ]
            asym_chain_res_name = asym_chain[res_starts].res_name.tolist()
            asym_chain_res_id = asym_chain[res_starts].res_id.tolist()

            entity_poly_seq["entity_id"].extend(asym_chain_entity_id)
            entity_poly_seq["hetero"].extend(asym_chain_hetero)
            entity_poly_seq["mon_id"].extend(asym_chain_res_name)
            entity_poly_seq["num"].extend(asym_chain_res_id)

        block_dict = {
            "entity_poly": pdbx.CIFCategory(entity_poly),
            "entity_poly_seq": pdbx.CIFCategory(entity_poly_seq),
        }
        return block_dict

    def save_to_cif(
        self, output_path: str, entry_id: str = None, include_bonds: bool = False
    ):
        """
        Save AtomArray to cif.

        Args:
            output_path (str): Output path of cif file.
            entry_id (str, optional): The value of "_entry.id" in cif. Defaults to None.
                                      If None, the entry_id will be the basename of output_path (without ".cif" extension).
            include_bonds (bool, optional): Whether to include  bonds in the cif. Defaults to False.
                                            If set to True and `array` has associated ``bonds`` , the
                                            intra-residue bonds will be written into the ``chem_comp_bond``
                                            category.
                                            Inter-residue bonds will be written into the ``struct_conn``
                                            independent of this parameter.

        """
        if entry_id is None:
            entry_id = os.path.basename(output_path).replace(".cif", "")

        block_dict = {"entry": pdbx.CIFCategory({"id": entry_id})}
        if self.entity_poly_type is not None:
            block_dict.update(self._get_entity_poly_and_entity_poly_seq_block())

        block = pdbx.CIFBlock(block_dict)
        cif = pdbx.CIFFile({os.path.basename(output_path).replace(".cif", ""): block})
        pdbx.set_structure(cif, self.atom_array, include_bonds=include_bonds)
        block = cif.block
        atom_site = block.get("atom_site")

        occ = atom_site.get("occupancy")
        if occ is None:
            atom_site["occupancy"] = np.ones(len(self.atom_array), dtype=float)

        atom_site["label_entity_id"] = self.atom_array.label_entity_id
        cif.write(output_path)


def remove_digits_from_label_asym_id(atom_array: AtomArray) -> AtomArray:
    """
    Remove digits from the label_asym_id.
    The numerical part of "label_asym_id" is added by Meson during data processing,
    with the purpose of distinguishing unique chains within the bioassembly.

    atom_array: AtomArray object with "label_asym_id" attribute.

    Returns: AtomArray object with "label_asym_id" attribute updated.
    """
    atom_array.label_asym_id = np.vectorize(remove_numbers)(atom_array.label_asym_id)
    return atom_array


def make_dummy_feature(
    features_dict: Mapping[str, torch.Tensor],
    dummy_feats: Sequence = ["msa"],
) -> dict[str, torch.Tensor]:
    num_token = features_dict["token_index"].shape[0]
    num_atom = features_dict["atom_to_token_idx"].shape[0]
    num_msa = 1
    num_templ = 4
    num_pockets = 30
    feat_shape, _ = get_data_shape_dict(
        num_token=num_token,
        num_atom=num_atom,
        num_msa=num_msa,
        num_templ=num_templ,
        num_pocket=num_pockets,
    )
    for feat_name in dummy_feats:
        if feat_name not in ["msa", "template"]:
            cur_feat_shape = feat_shape[feat_name]
            features_dict[feat_name] = torch.zeros(cur_feat_shape)
    if "msa" in dummy_feats:
        # features_dict["msa"] = features_dict["restype"].unsqueeze(0)
        features_dict["msa"] = torch.nonzero(features_dict["restype"])[:, 1].unsqueeze(
            0
        )
        assert features_dict["msa"].shape == feat_shape["msa"]
        features_dict["has_deletion"] = torch.zeros(feat_shape["has_deletion"])
        features_dict["deletion_value"] = torch.zeros(feat_shape["deletion_value"])
        features_dict["profile"] = features_dict["restype"]
        assert features_dict["profile"].shape == feat_shape["profile"]
        features_dict["deletion_mean"] = torch.zeros(feat_shape["deletion_mean"])
        for key in [
            "prot_pair_num_alignments",
            "prot_unpair_num_alignments",
            "rna_pair_num_alignments",
            "rna_unpair_num_alignments",
        ]:
            features_dict[key] = torch.tensor(0, dtype=torch.int32)

    if "template" in dummy_feats:
        features_dict["template_restype"] = (
            torch.ones(feat_shape["template_restype"]) * 31
        )  # gap
        features_dict["template_all_atom_mask"] = torch.zeros(
            feat_shape["template_all_atom_mask"]
        )
        features_dict["template_all_atom_positions"] = torch.zeros(
            feat_shape["template_all_atom_positions"]
        )
    return features_dict


def data_type_transform(
    feat_or_label_dict: Mapping[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], AtomArray]:
    for key, value in feat_or_label_dict.items():
        if key in IntDataList:
            feat_or_label_dict[key] = value.to(torch.long)

    return feat_or_label_dict


# List of "index" or "type" data
# Their data type should be int
IntDataList = [
    "residue_index",
    "token_index",
    "asym_id",
    "entity_id",
    "sym_id",
    "ref_space_uid",
    "template_restype",
    "atom_to_token_idx",
    "atom_to_tokatom_idx",
    "frame_atom_index",
    "msa",
    "entity_mol_id",
    "mol_id",
    "mol_atom_index",
]


# shape of the data
def get_data_shape_dict(num_token, num_atom, num_msa, num_templ, num_pocket):
    """
    Generate a dictionary containing the shapes of all data.

    Args:
        num_token (int): Number of tokens.
        num_atom (int): Number of atoms.
        num_msa (int): Number of MSA sequences.
        num_templ (int): Number of templates.
        num_pocket (int): Number of pockets to the same interested ligand.

    Returns:
        dict: A dictionary containing the shapes of all data.
    """
    # Features in AlphaFold3 SI Table5
    feat = {
        # Token features
        "residue_index": (num_token,),
        "token_index": (num_token,),
        "asym_id": (num_token,),
        "entity_id": (num_token,),
        "sym_id": (num_token,),
        "restype": (num_token, 32),
        # chain permutation features
        "entity_mol_id": (num_atom,),
        "mol_id": (num_atom,),
        "mol_atom_index": (num_atom,),
        # Reference features
        "ref_pos": (num_atom, 3),
        "ref_mask": (num_atom,),
        "ref_element": (num_atom, 128),  # note: 128 elem in the paper
        "ref_charge": (num_atom,),
        "ref_atom_name_chars": (num_atom, 4, 64),
        "ref_space_uid": (num_atom,),
        # Msa features
        # "msa": (num_msa, num_token, 32),
        "msa": (num_msa, num_token),
        "has_deletion": (num_msa, num_token),
        "deletion_value": (num_msa, num_token),
        "profile": (num_token, 32),
        "deletion_mean": (num_token,),
        # Template features
        "template_restype": (num_templ, num_token),
        "template_all_atom_mask": (num_templ, num_token, 37),
        "template_all_atom_positions": (num_templ, num_token, 37, 3),
        "template_pseudo_beta_mask": (num_templ, num_token),
        "template_backbone_frame_mask": (num_templ, num_token),
        "template_distogram": (num_templ, num_token, num_token, 39),
        "template_unit_vector": (num_templ, num_token, num_token, 3),
        # Bond features
        "token_bonds": (num_token, num_token),
        "is_protein": (num_atom,),  # Atom level, not token level
        "is_rna": (num_atom,),
        "is_dna": (num_atom,),
        "is_ligand": (num_atom,),
        "distogram_rep_atom_mask": (num_atom,),
        "pae_rep_atom_mask": (num_atom,),
        "plddt_m_rep_atom_mask": (num_atom,),
        "modified_res_mask": (num_atom,),
        "bond_mask": (num_atom, num_atom),
        "resolution": (1,),
    }

    # Extra features needed
    extra_feat = {
        # Input features
        "atom_to_token_idx": (num_atom,),  # after crop
        "atom_to_tokatom_idx": (num_atom,),  # after crop
        "pae_rep_atom_mask": (num_atom,),  # same as "pae_rep_atom_mask" in label_dict
        "is_distillation": (1,),
    }

    # Label
    label = {
        "coordinate": (num_atom, 3),
        "coordinate_mask": (num_atom,),
        # "centre_atom_mask": (num_atom,),
        # "centre_centre_distance": (num_token, num_token),
        # "centre_centre_distance_mask": (num_token, num_token),
        "has_frame": (num_token,),  # move to input_feature_dict?
        "frame_atom_index": (num_token, 3),  # atom index after crop
        # Metrics
        "interested_ligand_mask": (
            num_pocket,
            num_atom,
        ),
        "pocket_mask": (
            num_pocket,
            num_atom,
        ),
    }

    # Merged
    all_feat = {**feat, **extra_feat}
    return all_feat, label


def get_lig_lig_bonds(
    atom_array: AtomArray, lig_include_ions: bool = False
) -> np.ndarray:
    """
    Get all inter-ligand bonds in order to create "token_bonds".

    Args:
        atom_array (AtomArray): biotite AtomArray object with "mol_type" attribute.
        lig_include_ions (bool, optional): . Defaults to False.

    Returns:
        np.ndarray: inter-ligand bonds, e.g. np.array([[atom1, atom2, bond_order]...])
    """
    if not lig_include_ions:
        # bonded ligand exclude ions
        unique_chain_id, counts = np.unique(
            atom_array.label_asym_id, return_counts=True
        )
        chain_id_to_count_map = dict(zip(unique_chain_id, counts))
        ions_mask = np.array(
            [
                chain_id_to_count_map[label_asym_id] == 1
                for label_asym_id in atom_array.label_asym_id
            ]
        )

        lig_mask = (atom_array.mol_type == "ligand") & ~ions_mask
    else:
        lig_mask = atom_array.mol_type == "ligand"

    chain_res_id = np.vstack((atom_array.label_asym_id, atom_array.res_id)).T
    idx_i = atom_array.bonds._bonds[:, 0]
    idx_j = atom_array.bonds._bonds[:, 1]

    ligand_ligand_bond_indices = np.where(
        (lig_mask[idx_i] & lig_mask[idx_j])
        & np.any(chain_res_id[idx_i] != chain_res_id[idx_j], axis=1)
    )[0]

    if ligand_ligand_bond_indices.size == 0:
        # no ligand-polymer bonds
        lig_polymer_bonds = np.empty((0, 3)).astype(int)
    else:
        lig_polymer_bonds = atom_array.bonds._bonds[ligand_ligand_bond_indices]
    return lig_polymer_bonds


def pdb_to_cif(input_fname: str, output_fname: str, entry_id: str = None):
    """
    Convert PDB to CIF.

    Args:
        input_fname (str): input PDB file name
        output_fname (str): output CIF file name
        entry_id (str, optional): entry id. Defaults to None.
    """
    pdbfile = PDBFile.read(input_fname)
    atom_array = pdbfile.get_structure(model=1, include_bonds=True, altloc="first")

    seq_to_entity_id = {}
    cnt = 0
    chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=True)

    # split chains by hetero
    new_chain_starts = []
    for c_start, c_stop in zip(chain_starts[:-1], chain_starts[1:]):
        new_chain_starts.append(c_start)
        chain_start_hetero = atom_array.hetero[c_start]
        hetero_diff = np.where(atom_array.hetero[c_start:c_stop] != chain_start_hetero)
        if hetero_diff[0].shape[0] > 0:
            new_chain_start = c_start + hetero_diff[0][0]
            new_chain_starts.append(new_chain_start)

    new_chain_starts += [chain_starts[-1]]

    # # split HETATM chains by res id
    new_chain_starts2 = []
    for c_start, c_stop in zip(new_chain_starts[:-1], new_chain_starts[1:]):
        new_chain_starts2.append(c_start)
        res_id_diff = np.diff(atom_array.res_id[c_start:c_stop])
        uncont_res_starts = np.where(res_id_diff >= 1)

        if uncont_res_starts[0].shape[0] > 0:
            for res_start_atom_idx in uncont_res_starts[0]:
                new_chain_start = c_start + res_start_atom_idx + 1
                # atom_array.hetero is True if "HETATM"
                if (
                    atom_array.hetero[new_chain_start]
                    and atom_array.hetero[new_chain_start - 1]
                ):
                    new_chain_starts2.append(new_chain_start)

    chain_starts = new_chain_starts2 + [chain_starts[-1]]

    label_entity_id = np.zeros(len(atom_array), dtype=np.int32)
    atom_index = np.arange(len(atom_array), dtype=np.int32)
    res_id = copy.deepcopy(atom_array.res_id)

    chain_id = copy.deepcopy(atom_array.chain_id)
    chain_count = 0
    for c_start, c_stop in zip(chain_starts[:-1], chain_starts[1:]):
        chain_count += 1
        new_chain_id = int_to_letters(chain_count)
        chain_id[c_start:c_stop] = new_chain_id

        chain_array = atom_array[c_start:c_stop]
        residue_starts = struc.get_residue_starts(chain_array, add_exclusive_stop=True)
        resname_seq = [name for name in chain_array[residue_starts[:-1]].res_name]
        resname_str = "_".join(resname_seq)
        if (
            all([name in DNA_STD_RESIDUES for name in resname_seq])
            and resname_str in seq_to_entity_id
        ):
            resname_seq = resname_seq[::-1]
            resname_str = "_".join(resname_seq)
            atom_index[c_start:c_stop] = atom_index[c_start:c_stop][::-1]

        if resname_str not in seq_to_entity_id:
            cnt += 1
            seq_to_entity_id[resname_str] = cnt
        label_entity_id[c_start:c_stop] = seq_to_entity_id[resname_str]

        res_cnt = 1
        for res_start, res_stop in zip(residue_starts[:-1], residue_starts[1:]):
            res_id[c_start:c_stop][res_start:res_stop] = res_cnt
            res_cnt += 1

    atom_array = atom_array[atom_index]

    # add label entity id
    atom_array.set_annotation("label_entity_id", label_entity_id)
    entity_poly_type = {}
    for seq, entity_id in seq_to_entity_id.items():
        resname_seq = seq.split("_")

        count = defaultdict(int)
        for name in resname_seq:
            if name in PRO_STD_RESIDUES:
                count["prot"] += 1
            elif name in DNA_STD_RESIDUES:
                count["dna"] += 1
            elif name in RNA_STD_RESIDUES:
                count["rna"] += 1
            else:
                count["other"] += 1

        if count["prot"] >= 2 and count["dna"] == 0 and count["rna"] == 0:
            entity_poly_type[entity_id] = "polypeptide(L)"
        elif count["dna"] >= 2 and count["rna"] == 0 and count["prot"] == 0:
            entity_poly_type[entity_id] = "polydeoxyribonucleotide"
        elif count["rna"] >= 2 and count["dna"] == 0 and count["prot"] == 0:
            entity_poly_type[entity_id] = "polyribonucleotide"
        else:
            # other entity type: ignoring
            continue

    # add label atom id
    atom_array.set_annotation("label_atom_id", atom_array.atom_name)

    # add label asym id
    atom_array.chain_id = chain_id  # reset chain_id
    atom_array.set_annotation("label_asym_id", atom_array.chain_id)

    # add label seq id
    atom_array.res_id = res_id  # reset res_id
    atom_array.set_annotation("label_seq_id", atom_array.res_id)

    w = CIFWriter(atom_array=atom_array, entity_poly_type=entity_poly_type)
    w.save_to_cif(
        output_fname,
        entry_id=entry_id or os.path.basename(output_fname),
        include_bonds=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb_file", type=str, required=True, help="The pdb file to parse"
    )
    parser.add_argument(
        "--cif_file", type=str, required=True, help="The cif file path to generate"
    )
    args = parser.parse_args()
    pdb_to_cif(args.pdb_file, args.cif_file)
