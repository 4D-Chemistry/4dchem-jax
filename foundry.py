#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2026 Dr David A Sinclair <david@s-hull.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Commercial licensing inquiries: Buy the author a beer (or a cask).

import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_quaternion_cnt(n, m, num_atoms):
    """Generates a Chiral CNT and seeds pi-electrons using Unit Quaternions."""
    print(f"Generating ({n},{m}) CNT lattice...")
    a_cc = 1.42
    a = a_cc * np.sqrt(3)
    a1 = np.array([a * np.sqrt(3)/2, a / 2])
    a2 = np.array([a * np.sqrt(3)/2, -a / 2])
    
    Ch = n * a1 + m * a2
    circumference = np.linalg.norm(Ch)
    radius = circumference / (2 * np.pi)
    
    cos_theta = (2*n + m) / (2 * np.sqrt(n**2 + n*m + m**2))
    chiral_angle = np.arccos(cos_theta)
    
    area_per_atom = (3 * np.sqrt(3) * a_cc**2) / 4
    required_area = num_atoms * area_per_atom
    estimated_length = (required_area / circumference) * 1.5 
    
    n_steps = int(circumference / a_cc) * 3
    m_steps = int(estimated_length / a_cc) * 3
    
    points = []
    for i in range(-n_steps, n_steps):
        for j in range(-m_steps, m_steps):
            p1 = i * a1 + j * a2
            p2 = i * a1 + j * a2 + np.array([a_cc, 0])
            points.extend([p1, p2])
            
    points = np.array(points)
    
    rotation_matrix = np.array([
        [np.cos(chiral_angle), -np.sin(chiral_angle)],
        [np.sin(chiral_angle),  np.cos(chiral_angle)]
    ])
    rotated_points = np.dot(points, rotation_matrix.T)
    
    x_2d = rotated_points[:, 0]
    y_2d = rotated_points[:, 1]
    
    valid_mask = (x_2d >= 0) & (x_2d < circumference) & (y_2d >= 0)
    x_filtered = x_2d[valid_mask]
    y_filtered = y_2d[valid_mask]
    
    phi = (x_filtered / circumference) * 2 * np.pi
    X_3d = radius * np.cos(phi)
    Y_3d = radius * np.sin(phi)
    Z_3d = y_filtered
    
    carbon_atoms = np.column_stack((X_3d, Y_3d, Z_3d))
    carbon_atoms = carbon_atoms[carbon_atoms[:, 2].argsort()]
    carbon_atoms = carbon_atoms[:num_atoms]
    
    electrons = []
    for idx, (cx, cy, cz) in enumerate(carbon_atoms):
        nx, ny = cx / radius, cy / radius
        ex = cx + 0.8 * nx
        ey = cy + 0.8 * ny
        ez = cz
        
        if idx % 2 == 0:
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
            phase = 0.0
        else:
            qw, qx, qy, qz = 0.0, 1.0, 0.0, 0.0
            phase = np.pi
            
        delta = 0.00386 
        electrons.append((ex, ey, ez, qw, qx, qy, qz, phase, delta))
        
    return carbon_atoms, np.array(electrons), radius

def export_quaternion_cml(carbon_atoms, electrons, n, m, filename="cnt_quaternion_seed.cml"):
    """Writes the geometry to the CML XML format."""
    print(f"Exporting JAX initialization state to {filename}...")
    cml = ET.Element("cml", xmlns="http://www.xml-cml.org/schema")
    molecule = ET.SubElement(cml, "molecule", id=f"cnt_{n}_{m}")
    atomArray = ET.SubElement(molecule, "atomArray")
    
    for idx, (x, y, z) in enumerate(carbon_atoms):
        ET.SubElement(atomArray, "atom", id=f"a{idx+1}", elementType="C", 
                      x3=f"{x:.4f}", y3=f"{y:.4f}", z3=f"{z:.4f}")
        
    for idx, e in enumerate(electrons):
        x, y, z, qw, qx, qy, qz, phase, delta = e
        e_node = ET.SubElement(atomArray, "atom", id=f"e{idx+1}", elementType="E", 
                               x3=f"{x:.4f}", y3=f"{y:.4f}", z3=f"{z:.4f}")
        ET.SubElement(e_node, "scalar", dictRef="minkowski:qw", dataType="xsd:double").text = f"{qw:.1f}"
        ET.SubElement(e_node, "scalar", dictRef="minkowski:qx", dataType="xsd:double").text = f"{qx:.1f}"
        ET.SubElement(e_node, "scalar", dictRef="minkowski:qy", dataType="xsd:double").text = f"{qy:.1f}"
        ET.SubElement(e_node, "scalar", dictRef="minkowski:qz", dataType="xsd:double").text = f"{qz:.1f}"
        ET.SubElement(e_node, "scalar", dictRef="minkowski:delta", dataType="xsd:double").text = str(delta)
        ET.SubElement(e_node, "scalar", dictRef="minkowski:phase", dataType="xsd:double").text = f"{phase:.4f}"

    xmlstr = minidom.parseString(ET.tostring(cml)).toprettyxml(indent="  ")
    xmlstr = '\n'.join([line for line in xmlstr.split('\n') if line.strip()])
    with open(filename, "w") as f:
        f.write(xmlstr)

if __name__ == "__main__":
    # Test parameters: A classic (12,4) semiconducting chiral tube
    n, m = 12, 4
    num_atoms = 240
    carbons, electrons, R = generate_quaternion_cnt(n, m, num_atoms)
    export_quaternion_cml(carbons, electrons, n, m)
