#coding=utf-8
import os
import sys
import copy
import json
import time
import rdkit
import torch
import random
import pickle
import torch.nn.functional as F
from torch import tensor
from rdkit import Chem
from rdkit.Chem import rdFreeSASA
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, knn, global_mean_pool, global_max_pool, radius_graph
"""
This file is used to get the geometric and multi-level structure info in a protein for our EquiPocket.
Pls intall msms(https://ccsb.scripps.edu/msms/) first.
"""


class MoleculeFeatures(object):
    def __init__(self, file_name):
        self.file_name = file_name
        if file_name[-3:] == "pdb":
            self.molecule = Chem.MolFromPDBFile(file_name)
        if file_name[-4:] == "mol2":
            self.molecule = Chem.MolFromMol2File(file_name)
        if file_name[-3:] == "sdf":
            self.molecule = Chem.SDMolSupplier(file_name)[0]

    # get_bond_length
    def get_bond_length(self, x_0, y_0, z_0, x_1, y_1, z_1):
        bond_length = ((x_0 - x_1)**2 + (y_0 - y_1)**2 + (z_0 - z_1)**2) ** 0.5
        return bond_length

    # get atom features
    def get_atom_features(self, tmp_atom, cal_sasa=False):
        tmp_data = []
        atom_index = tmp_atom.GetIdx()
        tmp_data.append(tmp_atom.GetAtomicNum())
        tmp_data.append(tmp_atom.GetFormalCharge() + 2)
        chiral_tag_list = 0
        chiral_tag = str(tmp_atom.GetChiralTag())
        if chiral_tag == "CHI_UNSPECIFIED":
            pass
        if chiral_tag == "CHI_TETRAHEDRAL_CW":
            chiral_tag_list = 1
        if chiral_tag == "CHI_TETRAHEDRAL_CCW":
            chiral_tag_list = 2
        if chiral_tag == "CHI_OTHER":
            chiral_tag_list = 3
        tmp_data.append(chiral_tag_list)
        tmp_data.append(1 if tmp_atom.GetIsAromatic()==True else 0)
        ## 1.10 判断原子是否在环上
        tmp_data.append(1 if tmp_atom.IsInRing() else 0)
        tmp_data.append(tmp_atom.GetDegree())
        x, y, z = self.molecule.GetConformer().GetAtomPosition(atom_index)
        pos = (x, y, z)
        return atom_index, tmp_data, pos

    # get edge_feature
    def get_edge_features(self, tmp_bond):
        tmp_result = []
        start_index = tmp_bond.GetBeginAtomIdx()
        end_index = tmp_bond.GetEndAtomIdx()
        # SINGLE, AROMATIC, DOUBLE, Zero
        bond_type = str(tmp_bond.GetBondType())
        bond_type_list = 0
        if bond_type == "SINGLE":
            bond_type_list = 1
        if bond_type == "DOUBLE":
            bond_type_list = 2
        if bond_type == "AROMATIC":
            bond_type_list = 3
        bond_ring = 1 if tmp_bond.IsInRing() else 0
        x_0, y_0, z_0 = self.molecule.GetConformer().GetAtomPosition(start_index)
        x_1, y_1, z_1 = self.molecule.GetConformer().GetAtomPosition(end_index)
        bond_length = self.get_bond_length(x_0, y_0, z_1, x_1, y_1, z_1)
        tmp_result = []
        tmp_result.append(bond_type_list)
        tmp_result += [bond_ring, bond_length]
        return start_index, end_index, tmp_result

    # regard molecule as graph
    def get_graph_features(self, init_index=0):
        self.all_atoms = {}
        atoms = self.molecule.GetAtoms()
        all_atom_index = []
        all_atom_features = []
        all_atom_pos = []
        for tmp_atom in atoms:
            atom_index, atom_feature, pos = self.get_atom_features(tmp_atom)
            all_atom_index.append(init_index + atom_index)
            all_atom_features.append(atom_feature)
            all_atom_pos.append(pos)
        bonds = self.molecule.GetBonds()
        all_edge_index = [[], []]
        all_edge_attr = []
        for tmp_bond in bonds:
            start_index, end_index, edge_feature = self.get_edge_features(tmp_bond)
            all_edge_index[0].append(init_index + start_index)
            all_edge_index[1].append(init_index + end_index)
            all_edge_attr.append(edge_feature)
            all_edge_index[1].append(init_index + start_index)
            all_edge_index[0].append(init_index + end_index)
            all_edge_attr.append(edge_feature)
        return all_atom_index, all_atom_features, all_atom_pos, all_edge_index, all_edge_attr

    # get_surface_feature from msms
    def get_surface(self, msms_path=""):
        atom_in_surface = []
        vert_surface = []
        pdb_file = self.file_name
        pdb_file = os.path.abspath(pdb_file)
        run_shell = "cd %s ;" % msms_path
        run_shell += "pdb_to_xyzr %s > tmp.xyzr;" % pdb_file
        run_shell += "msms -probe_radius 1.5 -if tmp.xyzr -af result -of result"
        run_result = os.system(run_shell)
        # 0: success
        if run_result == 0:
            result_area_path = os.path.join(msms_path, "result.area")
            result_face_path = os.path.join(msms_path, "result.face")
            result_vert_path = os.path.join(msms_path, "result.vert")
            # get vert
            tmp_i = 0
            f = open(result_vert_path)
            for line in f:
                tmp_i += 1
                if tmp_i <= 3:
                    continue
                line = list(map(float, line.strip().split()))
                vert_surface.append(line)
            f.close()
        return vert_surface

def get_surface_feature(vert_surface, protein_pos, mean_protein_pos):
    pos = protein_pos
    vert_pos = vert_surface[:, [0, 1, 2]]
    vert_pos = torch.unique(vert_pos, dim=0)
    vert_pos = vert_pos - mean_protein_pos
    dist_atom_pos_vert_pos = torch.cdist(vert_pos.clone(), pos)
    vert_atom = torch.argmin(dist_atom_pos_vert_pos, dim=1)
    vert_atom = vert_atom.long()
    atom_in_surface = torch.zeros(protein_pos.shape[0])
    atom_in_surface[vert_atom.unique().long()] = 1
    vert_atom_diff = vert_pos - pos[vert_atom]
    vert_num = torch.tensor(vert_atom.shape[0])
    sort_vert_atom, indices = torch.sort(vert_atom)
    vert_atom = sort_vert_atom
    vert_pos = vert_pos[indices]
    vert_atom_diff = vert_atom_diff[indices]
    vert_surface = vert_surface[indices]
    _, vert_batch = torch.unique(vert_atom, return_inverse=True)
    return vert_pos, vert_atom, vert_num, atom_in_surface, vert_atom_diff, vert_batch

def get_surface_descriptor(pos, vert_pos, vert_atom, atom_in_surface, vert_batch):
    tmp_pos = pos[atom_in_surface==1]
    # KNN for two nearest surface point
    assign_index = knn(vert_pos, vert_pos, 3)
    edge_0 = assign_index[0]
    edge_1 = assign_index[1]
    mask_edge = (edge_0 == edge_1)
    edge_0 = edge_0[~mask_edge]
    tmp_edge_0 = edge_0.clone()
    edge_0 = vert_pos[edge_0]
    edge_1 = edge_1[~mask_edge]
    tmp_edge_1 = edge_1.clone()
    edge_1 = vert_pos[edge_1]
    edge_diff = edge_0 - edge_1
    edge_diff = edge_diff.view(vert_pos.shape[0], 2, 3)
    length_edge_0 = edge_diff[:, 0, :].norm(dim=1).unsqueeze(dim=-1)
    length_edge_1 = edge_diff[:, 1, :].norm(dim=1).unsqueeze(dim=-1)
    angle_knn = F.normalize(edge_diff[:, 0, :]) * F.normalize(edge_diff[:, 1, :])
    angle_knn = torch.mul(F.normalize(edge_diff[:, 0, :]), F.normalize(edge_diff[:, 1, :])).sum(dim=1).unsqueeze(dim=-1)
    angle_knn[torch.isnan(angle_knn)] = 1
    # the former 3 features for local geometric
    knn_geometric_feature = torch.concat([length_edge_0, length_edge_1, angle_knn], dim=1)
    # the latter 4 features
    surface_center_pos = global_mean_pool(vert_pos, vert_batch)
    surface_pos_to_center = vert_pos - surface_center_pos[vert_batch]
    surface_pos_to_atom = vert_pos - pos[vert_atom]
    surface_center_to_atom = surface_center_pos - pos[atom_in_surface==1]
    dist_atom_to_surface_center = surface_center_to_atom.square().sum(dim=1).sqrt().unsqueeze(dim=-1)
    dist_surface_point_to_surface_center = surface_pos_to_center.square().sum(dim=1).sqrt().unsqueeze(dim=-1)
    dist_surface_point_to_atom = surface_pos_to_atom.square().sum(dim=1).sqrt().unsqueeze(dim=-1)
    cos_surface_point_atom = torch.mul(surface_pos_to_center, surface_center_to_atom[vert_batch]).sum(dim=1).unsqueeze(dim=-1)
    cos_surface_point_atom = cos_surface_point_atom / (dist_atom_to_surface_center[vert_batch])
    cos_surface_point_atom = cos_surface_point_atom / dist_surface_point_to_surface_center
    cos_surface_point_atom[torch.isnan(cos_surface_point_atom)] = 1
    surface_shape_geometric_feature = torch.concat([dist_surface_point_to_surface_center,
        dist_surface_point_to_atom,
        dist_atom_to_surface_center[vert_batch],
        cos_surface_point_atom], dim=1)
    surface_descriptor = torch.concat([knn_geometric_feature, surface_shape_geometric_feature], dim=1)
    return surface_descriptor, surface_center_pos

def get_protein_feature(protein_file_name, msms_path=""):
    protein = MoleculeFeatures(protein_file_name)
    atoms = protein.molecule.GetAtoms()
    # get global structure features
    all_atom_index, all_atom_features, all_atom_pos, all_edge_index, all_edge_attr = protein.get_graph_features()
    all_atom_pos = torch.tensor(all_atom_pos).float()
    mean_protein_pos = all_atom_pos.mean(dim=0)
    all_atom_pos = all_atom_pos - mean_protein_pos
    # get_surface_features
    vert_surface = protein.get_surface(msms_path=msms_path)
    vert_surface = tensor(vert_surface).float()
    vert_pos, vert_atom, vert_num, atom_in_surface, vert_atom_diff, vert_batch = get_surface_feature(vert_surface, all_atom_pos, mean_protein_pos)
    # get_surface_descriptor
    surface_descriptor, surface_center_pos = get_surface_descriptor(all_atom_pos, vert_pos, vert_atom, atom_in_surface, vert_batch)
    # trans data -> graph data
    all_atom_features = tensor(all_atom_features).float()
    all_edge_index = tensor(all_edge_index)
    all_edge_attr = tensor(all_edge_attr).float()
    graph_data = Data(x=all_atom_features,
            pos=all_atom_pos,
            edge_index=all_edge_index,
            edge_attr=all_edge_attr,
            atom_in_surface=atom_in_surface,
            vert_surface=vert_surface,
            vert_pos=vert_pos,
            vert_atom=vert_atom,
            vert_num=vert_atom,
            vert_atom_diff=vert_atom_diff,
            vert_batch=vert_batch,
            surface_center_pos=surface_center_pos,
            surface_descriptor=surface_descriptor)
    return graph_data

if __name__ == "__main__":
    # pls install msms at first
    msms_path = ""
    protein_file_name = "protein.pdb"
    tmp_graph = get_protein_feature(protein_file_name, msms_path=msms_path)
    # Data(x=[1572, 6], edge_index=[2, 3224], edge_attr=[3224, 3], pos=[1572, 3], atom_in_surface=[1572], vert_surface=[10385, 9], vert_pos=[10384, 3], vert_atom=[10384], vert_num=[10384], vert_atom_diff=[10384, 3], vert_batch=[10384], surface_center_pos=[988, 3], surface_descriptor=[10384, 7])

