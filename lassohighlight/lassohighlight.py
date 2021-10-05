import rdkit.Chem.Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdmolops import Get3DDistanceMatrix
from rdkit import Chem
from rdkit.Geometry.rdGeometry import Point2D
import numpy as np
from collections import defaultdict
from collections import namedtuple
from typing import *


def angle_to_coord(center, angle, radius) -> np.ndarray:
    """Determines a point relative to the center with distance (radius) at given angle.
    Angles are given in rad and 0 rad correspond to north of the center point.
    """
    x = radius * np.sin(angle)
    y = radius * np.cos(angle)
    x += center[0]
    y += center[1]
    return np.array([x, y])


def arch_points(radius, start_ang, end_ang, n) -> np.ndarray:
    """Returns an array of the shape (2, n) with equidistant points on the arch defined by given radius and angles.
    Angles are given in rad.
    """
    angles = np.linspace(start_ang, end_ang, n)
    x = radius * np.sin(angles)
    y = radius * np.cos(angles)
    return np.vstack([x, y]).T


def angle_between(center, pos):
    """Calculates the angle in rad between two points.
    An angle of 0 corresponds to north of the center.
    """
    diff = pos - center
    return np.arctan2(diff[0], diff[1])


def avg_bondlen(mol: Chem.Mol):
    """Calculates the average bond length of an rdkit.Chem.Mol object."""
    distance_matrix = Get3DDistanceMatrix(mol)
    bondlength_list: List = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        bondlength_list.append(distance_matrix[a1, a2])
    return np.mean(bondlength_list)


Bond = namedtuple("Bond", ["angle", "neighbour_id", "bond_id"])


class AnchorManager:
    """AnchorManager is an invisible overlay for RDKit Atoms storing positions for arches and bond-attachment-points.
    """

    def __init__(self, position, radius, relative_bond_radius):
        """
        """

        self.pos = position
        self.bond_ratio = relative_bond_radius
        self.radius = radius
        self.bonds: List[Bond] = []
        self.bond_attachment_points: Optional[List[Tuple]] = None

    def add_bond(self, angle, neighbor_id, bond_id):
        self.bonds.append(Bond(angle, neighbor_id, bond_id))

    @property
    def abs_bond_radius(self):
        return self.radius * self.bond_ratio

    @property
    def padding_angle(self):
        return np.arcsin(self.abs_bond_radius / self.radius)

    def generate_attachment_points(self):
        sorted_bonds = sorted(self.bonds, key=lambda x: x.angle)
        self.bond_attachment_points = dict()
        for i, bond in enumerate(sorted_bonds):
            start_angle = bond.angle - self.padding_angle
            end_angle = bond.angle + self.padding_angle

            start_r = self.radius
            end_r = self.radius

            # Handling intersecting bonds
            # # Intersections with previous bonds
            if i == 0:  # For first bond the previous bond is the last bond. Subtracting 2 pi.
                prev_bond_angle = sorted_bonds[-1].angle - np.pi * 2
            else:
                prev_bond_angle = sorted_bonds[i - 1].angle

            # #  If both points intersect the mean angle is calculated.
            if prev_bond_angle + self.padding_angle >= start_angle:

                start_angle = np.mean([prev_bond_angle + self.padding_angle, start_angle])

                inter_bond_angle = bond.angle - prev_bond_angle

                rhom_side_len = self.abs_bond_radius / np.sin(inter_bond_angle)
                # Radius is altered to match the intersecting position
                start_r = 2 * rhom_side_len * np.cos(inter_bond_angle / 2)

            # # Intersections with following bonds
            if i + 1 == len(sorted_bonds):
                next_bond_angle = sorted_bonds[0].angle + np.pi * 2
            else:
                next_bond_angle = sorted_bonds[i + 1].angle

            if next_bond_angle - self.padding_angle <= end_angle:
                end_angle = np.mean([next_bond_angle - self.padding_angle, end_angle])

                inter_bond_angle = next_bond_angle - bond.angle
                rhom_side_len = self.abs_bond_radius / np.sin(inter_bond_angle)
                end_r = 2 * rhom_side_len * np.cos(inter_bond_angle / 2)

            self.bond_attachment_points[bond.bond_id] = [(start_angle, start_r), (end_angle, end_r)]
        return self

    def get_arch_anchors(self) -> Iterator[Tuple[float, float]]:
        """Points between bonds which are drawn as arch."""
        if self.bonds:
            sorted_bonds = sorted(self.bonds, key=lambda x: x[0])
            _, _, bond_keys = zip(*sorted_bonds)
            for i, k in enumerate(bond_keys):
                if i == 0:
                    start_angle = self.bond_attachment_points[bond_keys[-1]][1][0] - np.pi * 2
                else:
                    start_angle = self.bond_attachment_points[bond_keys[i - 1]][1][0]
                end_angle = self.bond_attachment_points[k][0][0]
                if np.isclose(start_angle % (np.pi * 2), end_angle % (np.pi * 2)):
                    continue
                yield start_angle, end_angle


ColorTuple = Union[Tuple[float, float, float, float], Tuple[float, float, float]]


def draw_substructurematch(canvas, mol, indices, atom_radius=0.3, relative_bond_width=0.5, line_width=2, color=None
                           ) -> None:
    """ Draws the substructure defined by (atom-) `indices`, as lasso-highlight onto `canvas`.

    Parameters
    ----------
    canvas : rdMolDraw2D.MolDraw2D
        RDKit Canvas, where highlighting is drawn to.

    mol: Chem.Mol
        Atoms from the molecule `mol` are takes as positional reference for the highlighting.

    indices: Union[list, str]
        Atom indices delineating highlighted substructure.

    atom_radius: float
        Radius of the circle around atoms. Length is relative to average bond length (1 = avg. bond len).

    relative_bond_width: float
        Distance of line to "bond" (line segment between the two atoms). Size is relative to `atom_radius`.

    line_width: int
        width of drawn lines.

    color: ColorTuple
           Tuple with RGBA or RGB values specifying the color of the highlighting.

    Returns
    -------
    None
    """

    prior_lw = canvas.LineWidth()
    canvas.SetLineWidth(line_width)

    # Setting color
    # #  Default color is gray.
    if not color:
        color = (0.5, 0.5, 0.5, 1)
    canvas.SetColour(color)

    # Selects first conformer and calculates the mean bond length
    conf = mol.GetConformer(0)
    avg_len = avg_bondlen(mol)
    r_ = avg_len * atom_radius

    a_obj_dict = dict()  # Dictionary for atoms delineating highlighted substructure.
    for atom in mol.GetAtoms():
        a_idx = atom.GetIdx()
        if a_idx not in indices:
            continue

        # 2D-coordinates of Atom
        atom_pos = conf.GetAtomPosition(a_idx)
        atom_pos = np.array([atom_pos.x, atom_pos.y])

        # Initializing an AnchorManager centered at the atom position
        anchor_point_manager = AnchorManager(atom_pos, r_, relative_bond_width)

        # Adding Bonds to the Anchor manager
        for bond in atom.GetBonds():
            bond_atom1 = bond.GetBeginAtomIdx()
            bond_atom2 = bond.GetEndAtomIdx()
            neigbor_idx = bond_atom1 if bond_atom2 == a_idx else bond_atom2
            if neigbor_idx not in indices:
                continue
            neigbor_pos = conf.GetAtomPosition(neigbor_idx)
            neigbor_pos = np.array([neigbor_pos.x, neigbor_pos.y])
            bond_angle = angle_between(atom_pos, neigbor_pos)
            bond_angle = bond_angle % (2*np.pi)  # Assuring 0 <= bond_angle <= 2 pi
            anchor_point_manager.add_bond(bond_angle, neigbor_idx, bond.GetIdx())
        anchor_point_manager.generate_attachment_points()
        a_obj_dict[a_idx] = anchor_point_manager

    added_bonds = set()
    for idx, anchor_point_manager in a_obj_dict.items():

        # A circle is drawn to atoms without outgoing connections
        if not anchor_point_manager.bonds:
            pos_list1 = arch_points(r_, 0, np.pi * 2, 60)
            pos_list1[:, 0] += anchor_point_manager.pos[0]
            pos_list1[:, 1] += anchor_point_manager.pos[1]
            points = [Point2D(*c) for c in pos_list1]
            canvas.DrawPolygon(points)

        # A arch is drawn between lines parallel to the bond
        for points in anchor_point_manager.get_arch_anchors():
            pos_list1 = arch_points(r_, points[0], points[1], 20)
            pos_list1[:, 0] += anchor_point_manager.pos[0]
            pos_list1[:, 1] += anchor_point_manager.pos[1]
            points = [Point2D(*c) for c in pos_list1]
            canvas.DrawPolygon(points)

        # Drawing lines parallel to each bond
        for bond in anchor_point_manager.bonds:
            if bond.bond_id in added_bonds:
                continue
            added_bonds.add(bond.bond_id)
            bnd_points = anchor_point_manager.bond_attachment_points[bond.bond_id]

            atom_ap1 = angle_to_coord(anchor_point_manager.pos, *bnd_points[0])
            atom_ap2 = angle_to_coord(anchor_point_manager.pos, *bnd_points[1])
            neig = a_obj_dict[bond.neighbour_id]
            neig_ap1 = angle_to_coord(neig.pos, *neig.bond_attachment_points[bond.bond_id][0])
            neig_ap2 = angle_to_coord(neig.pos, *neig.bond_attachment_points[bond.bond_id][1])
            canvas.DrawLine(Point2D(*atom_ap1), Point2D(*neig_ap2))
            canvas.DrawLine(Point2D(*atom_ap2), Point2D(*neig_ap1))
    # restoring prior line width
    canvas.SetLineWidth(prior_lw)


def draw_multi_matches(canvas, mol, indices_set_lists, r_min=0.3, r_dist=0.13, relative_bond_width=0.5, color_list=None,
                       line_width=2):
    """

    Parameters
    ----------
    canvas : rdMolDraw2D.MolDraw2D
        RDKit Canvas, where highlighting is drawn to.

    mol: Chem.Mol
        Atoms from the molecule `mol` are takes as positional reference for the highlighting.

    indices_set_lists: List[Union[list, str]]
        Atom indices delineating highlighted substructure.

    r_min: float
        Radius of the smallest circle around atoms. Length is relative to average bond length (1 = avg. bond len).

    r_dist: float
        Incremental increase of radius for the next substructure.

    relative_bond_width: float
        Distance of line to "bond" (line segment between the two atoms). Size is relative to `atom_radius`.

    line_width: int
        width of drawn lines.

    color_list: List[ColorTuple]
           List of tuples with RGBA or RGB values specifying the color of the highlighting.

    Returns
    -------
    None

    Returns
    -------

    """
    # If no colors are given, all substructures are depicted in gray.
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
                               relative_bond_width=max(relative_bond_width, ar),
                               color=color,
                               line_width=line_width)

