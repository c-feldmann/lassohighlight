from rdkit.Chem.rdmolops import Get3DDistanceMatrix
from rdkit.Geometry.rdGeometry import Point2D
import numpy as np
from collections import defaultdict
from typing import *


def angle_to_coord(center, angle, radius) -> np.ndarray:
    """Determines a point relative to the center with distance (radius) at given angle"""
    x = radius * np.sin(angle)
    y = radius * np.cos(angle)
    x += center[0]
    y += center[1]
    return np.array([x, y])


def arch_points(radius, start_rad, end_rad, n):
    angles = np.linspace(start_rad, end_rad, n)
    x = radius * np.sin(angles)
    y = radius * np.cos(angles)
    return np.vstack([x, y]).T


def angle_between(center, pos):
    diff = pos - center
    return np.arctan2(diff[0], diff[1])


def avg_bondlen(mol):
    distance_matrix = Get3DDistanceMatrix(mol)
    bondlength_list: List = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bondlength_list.append(distance_matrix[a1, a2])
    return np.mean(bondlength_list)


class DrawAtom:
    def __init__(self, atom_id, position, radius, relative_bond_radius):
        self.atom_id = atom_id
        self.pos = position
        self.bond_ratio = relative_bond_radius
        self.radius = radius
        self.bonds = []
        self.attachment_points = None

    def add_bond(self, angle, neighbor_id, bond_id):
        self.bonds.append((angle, neighbor_id, bond_id))

    @property
    def abs_bond_radius(self):
        return self.radius * self.bond_ratio

    @property
    def padding_angle(self):
        return np.arcsin(self.abs_bond_radius / self.radius)

    def init_attachment_points(self):
        sorted_bonds = sorted(self.bonds, key=lambda x: x[0])
        attachment_points = dict()
        for i, bond_tuple in enumerate(sorted_bonds):
            bond_angle = bond_tuple[0]
            start_angle = bond_angle - self.padding_angle
            end_angle = bond_angle + self.padding_angle

            start_r = self.radius
            end_r = self.radius

            # Handling intersecting bonds
            if i == 0:
                prev_bond_angle = sorted_bonds[-1][0] - np.pi * 2
            else:
                prev_bond_angle = sorted_bonds[i - 1][0]

            if prev_bond_angle + self.padding_angle > start_angle:
                start_angle = np.mean([prev_bond_angle + self.padding_angle, start_angle])

                inter_bond_angle = bond_angle - prev_bond_angle

                rhom_side_len = self.abs_bond_radius / np.sin(inter_bond_angle)
                start_r = 2 * rhom_side_len * np.cos(inter_bond_angle / 2)

            if i + 1 == len(sorted_bonds):
                next_bond_angle = sorted_bonds[0][0] + np.pi * 2
            else:
                next_bond_angle = sorted_bonds[i + 1][0]

            if next_bond_angle - self.padding_angle < end_angle:
                end_angle = np.mean([next_bond_angle - self.padding_angle, end_angle])

                inter_bond_angle = next_bond_angle - bond_angle
                rhom_side_len = self.abs_bond_radius / np.sin(inter_bond_angle)
                end_r = 2 * rhom_side_len * np.cos(inter_bond_angle / 2)

            attachment_points[bond_tuple[2]] = [(start_angle, start_r), (end_angle, end_r)]
        self.attachment_points = attachment_points
        return self

    def get_arch_anchors(self):
        if self.bonds:
            sorted_bonds = sorted(self.bonds, key=lambda x: x[0])
            _, _, bond_keys = zip(*sorted_bonds)
            for i, k in enumerate(bond_keys):
                if i == 0:
                    start_angle = self.attachment_points[bond_keys[-1]][1][0] - np.pi * 2
                else:
                    start_angle = self.attachment_points[bond_keys[i - 1]][1][0]
                end_angle = self.attachment_points[k][0][0]
                if np.isclose(start_angle % (np.pi * 2), end_angle % (np.pi * 2)):
                    continue
                yield start_angle, end_angle


def draw_substructurematch(canvas, mol, indices, atom_radius=0.3, bond_radius_ratio=0.5, color=None):
    if not color:
        color = (0.5, 0.5, 0.5, 1)
    canvas.SetColour(color)
    conf = mol.GetConformer(0)
    canvas.SetFillPolys(False)
    avg_len = avg_bondlen(mol)
    r_ = avg_len * atom_radius
    h_ = r_ * bond_radius_ratio
    pad_angle = np.arcsin(h_ / r_)

    a_obj_dict = dict()
    for atom in mol.GetAtoms():
        a_idx = atom.GetIdx()
        if a_idx not in indices:
            continue

        atom_pos = conf.GetAtomPosition(a_idx)
        atom_pos = np.array([atom_pos.x, atom_pos.y])

        atom_obj = DrawAtom(a_idx, atom_pos, r_, bond_radius_ratio)
        for bond in atom.GetBonds():
            bond_atom1 = bond.GetBeginAtomIdx()
            bond_atom2 = bond.GetEndAtomIdx()
            neigbor_idx = bond_atom1 if bond_atom2 == a_idx else bond_atom2
            if neigbor_idx not in indices:
                continue
            neigbor_pos = conf.GetAtomPosition(neigbor_idx)
            neigbor_pos = np.array([neigbor_pos.x, neigbor_pos.y])
            atom_obj.add_bond(angle_between(atom_pos, neigbor_pos) % (np.pi * 2), neigbor_idx, bond.GetIdx())
        atom_obj.init_attachment_points()
        a_obj_dict[a_idx] = atom_obj

    added_bonds = set()
    for idx, atom in a_obj_dict.items():

        # When no bonds lead to the atom a circle is drawn
        if not atom.bonds:
            pos_list1 = arch_points(r_, 0, np.pi * 2, 60)
            pos_list1[:, 0] += atom.pos[0]
            pos_list1[:, 1] += atom.pos[1]
            points = [Point2D(*c) for c in pos_list1]
            canvas.DrawPolygon(points)

        # A arch is drawn between lines parallel to the bond
        for points in atom.get_arch_anchors():
            pos_list1 = arch_points(r_, points[0], points[1], 20)
            pos_list1[:, 0] += atom.pos[0]
            pos_list1[:, 1] += atom.pos[1]
            points = [Point2D(*c) for c in pos_list1]
            canvas.DrawPolygon(points)

        # Drawing lines parallel to each bond
        for bond in atom.bonds:
            bond_idx = bond[2]
            if bond_idx in added_bonds:
                continue
            added_bonds.add(bond_idx)

            atom_ap1 = angle_to_coord(atom.pos, *atom.attachment_points[bond_idx][0])
            atom_ap2 = angle_to_coord(atom.pos, *atom.attachment_points[bond_idx][1])
            neig = a_obj_dict[bond[1]]
            neig_ap1 = angle_to_coord(neig.pos, *neig.attachment_points[bond_idx][0])
            neig_ap2 = angle_to_coord(neig.pos, *neig.attachment_points[bond_idx][1])
            canvas.DrawLine(Point2D(*atom_ap1), Point2D(*neig_ap2))
            canvas.DrawLine(Point2D(*atom_ap2), Point2D(*neig_ap1))


def draw_multi_matches(canvas, mol, indices_set_lists, r_min=0.3, r_dist=0.13, bond_radius_ratio=0.5, color_list=None):
    if color_list is None:
        color_list = [(0.5, 0.5, 0.5)] * len(indices_set_lists)
    if len(color_list) < len(indices_set_lists):
        raise ValueError("Not enough colors for referenced substructures!")

    level_manager = defaultdict(set)
    for match_atoms, color in zip(indices_set_lists, color_list):
        used_levels = set.union(*[level_manager[a] for a in match_atoms])
        if len(used_levels) == 0:
            free_levels = {0}
        else:
            max_level = max(used_levels)
            free_levels = set(range(max_level)) - used_levels

        if free_levels:
            draw_level = min(free_levels)
        else:
            draw_level = max(used_levels) + 1

        for a in match_atoms:
            level_manager[a].add(draw_level)

        ar = r_min + r_dist * draw_level
        draw_substructurematch(canvas,
                               mol,
                               match_atoms,
                               atom_radius=ar,
                               bond_radius_ratio=max(bond_radius_ratio, ar),
                               color=color)

