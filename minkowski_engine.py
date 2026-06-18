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

import jax
import jax.numpy as jnp
import optax
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import numpy as np
import time

# --- Physical Constants ---
K_E = 14.3996        
DELTA_0 = 0.00386      
EPS = 1e-6           
CORE_RADIUS = 0.45   

# Magnetic Phase-Locking Constants
LAM_MAG = 0.42       
DELTA_MAG = 0.026    
J_MAG = 0.5          
K_ANISO = 0.2        

# Spring Constants 
K_TENSION = 1000.0   
K_COMPRESS = 50.0    

def parse_cml(filename="cnt_quaternion_seed.cml"):
    """Parses the CML file and extracts the Carbon and Electron tensors."""
    print(f"Loading {filename}...")
    tree = ET.parse(filename)
    root = tree.getroot()
    ns = {'cml': 'http://www.xml-cml.org/schema'}
    carbons, e_pos, e_quat, e_delta, e_phase = [], [], [], [], []
    
    for atom in root.findall('.//cml:atom', ns):
        elem = atom.attrib['elementType']
        x, y, z = float(atom.attrib['x3']), float(atom.attrib['y3']), float(atom.attrib['z3'])
        
        if elem == 'C':
            carbons.append([x, y, z])
        elif elem == 'E':
            e_pos.append([x, y, z])
            scalars = {s.attrib['dictRef']: float(s.text) for s in atom.findall('cml:scalar', ns)}
            e_quat.append([scalars['minkowski:qw'], scalars['minkowski:qx'], 
                           scalars['minkowski:qy'], scalars['minkowski:qz']])
            e_delta.append(scalars['minkowski:delta'])
            e_phase.append(scalars['minkowski:phase'])

    return (jnp.array(carbons), 
            {'pos': jnp.array(e_pos), 'quat': jnp.array(e_quat), 
             'delta': jnp.array(e_delta), 'phase': jnp.array(e_phase)})

@jax.jit
def normalize_quaternions(q):
    norm_sq = jnp.sum(q**2, axis=-1, keepdims=True)
    return q / jnp.sqrt(norm_sq + 1e-12)

@jax.jit
def quat_to_loop_normal(q):
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    nx = 2.0 * (qx * qz + qw * qy)
    ny = 2.0 * (qy * qz - qw * qx)
    nz = 1.0 - 2.0 * (qx**2 + qy**2)
    return jnp.stack([nx, ny, nz], axis=-1)

@jax.jit
def compute_total_energy(params, carbon_pos):
    pos = params['pos']
    quat = normalize_quaternions(params['quat'])
    delta = params['delta']
    phase = params['phase']
    N = pos.shape[0]
    
    diff_eN = pos[:, None, :] - carbon_pos[None, :, :]
    dist_eN_sq = jnp.sum(diff_eN**2, axis=-1)
    v_eN = -K_E / jnp.sqrt(dist_eN_sq + delta[:, None]**2 + CORE_RADIUS**2)
    energy_lattice = jnp.sum(v_eN)

    diff_ee = pos[:, None, :] - pos[None, :, :]
    dist_ee_sq = jnp.sum(diff_ee**2, axis=-1)
    safe_dist_ee_sq = dist_ee_sq + jnp.eye(N) * 1e-12
    dist_ee = jnp.sqrt(safe_dist_ee_sq)
    
    mask = jnp.triu(jnp.ones((N, N)), k=1)
    delta_avg = (delta[:, None] + delta[None, :]) / 2.0
    v_ee = 0.5 * K_E * (1.0 / (dist_ee + EPS) + 1.0 / jnp.sqrt(safe_dist_ee_sq + delta_avg**2))
    energy_repulsion = jnp.sum(v_ee * mask)

    normals = quat_to_loop_normal(quat)
    easy_axis = jnp.array([0.0, 0.0, 1.0]) 
    alignment = jnp.sum(normals * easy_axis, axis=-1)
    energy_anisotropy = jnp.sum(K_ANISO * (1.0 - alignment**2))

    spin_alignment = jnp.einsum('id,jd->ij', normals, normals)
    phase_interaction = jnp.cos(phase[:, None] - phase[None, :])
    v_mag_potential = (LAM_MAG * K_E) / jnp.sqrt(safe_dist_ee_sq + DELTA_MAG**2)
    e_mag = v_mag_potential * (spin_alignment * phase_interaction)
    energy_magnetic = jnp.sum(e_mag * mask)

    strain = delta - DELTA_0
    energy_spring = jnp.sum(jnp.where(strain > 0, 0.5 * K_TENSION * strain**2, 0.5 * K_COMPRESS * strain**2))

    return energy_lattice + energy_repulsion + energy_anisotropy + energy_magnetic + energy_spring

def create_step_fn(optimizer):
    @jax.jit
    def step(params, opt_state, carbon_pos):
        loss, grads = jax.value_and_grad(compute_total_energy)(params, carbon_pos)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_params['quat'] = normalize_quaternions(new_params['quat'])
        new_params['phase'] = jnp.mod(new_params['phase'], 2 * jnp.pi)
        return new_params, new_opt_state, loss
    return step

def export_relaxed_cml(carbon_pos, params, filename="cnt_relaxed.cml"):
    print(f"Exporting relaxed state to {filename}...")
    carbons = np.asarray(carbon_pos)
    pos = np.asarray(params['pos'])
    quat = np.asarray(params['quat'])
    delta = np.asarray(params['delta'])
    phase = np.asarray(params['phase'])
    
    cml = ET.Element("cml", xmlns="http://www.xml-cml.org/schema")
    molecule = ET.SubElement(cml, "molecule", id="cnt_relaxed")
    atomArray = ET.SubElement(molecule, "atomArray")
    
    for idx, (x, y, z) in enumerate(carbons):
        ET.SubElement(atomArray, "atom", id=f"a{idx+1}", elementType="C", x3=f"{x:.4f}", y3=f"{y:.4f}", z3=f"{z:.4f}")
        
    for idx in range(len(pos)):
        x, y, z = pos[idx]
        qw, qx, qy, qz = quat[idx]
        e_node = ET.SubElement(atomArray, "atom", id=f"e{idx+1}", elementType="E", x3=f"{x:.4f}", y3=f"{y:.4f}", z3=f"{z:.4f}")
        ET.SubElement(e_node, "scalar", dictRef="minkowski:qw", dataType="xsd:double").text = f"{qw:.4f}"
        ET.SubElement(e_node, "scalar", dictRef="minkowski:qx", dataType="xsd:double").text = f"{qx:.4f}"
        ET.SubElement(e_node, "scalar", dictRef="minkowski:qy", dataType="xsd:double").text = f"{qy:.4f}"
        ET.SubElement(e_node, "scalar", dictRef="minkowski:qz", dataType="xsd:double").text = f"{qz:.4f}"
        ET.SubElement(e_node, "scalar", dictRef="minkowski:delta", dataType="xsd:double").text = f"{delta[idx]:.4f}"
        ET.SubElement(e_node, "scalar", dictRef="minkowski:phase", dataType="xsd:double").text = f"{phase[idx]:.4f}"

    xmlstr = minidom.parseString(ET.tostring(cml)).toprettyxml(indent="  ")
    xmlstr = '\n'.join([line for line in xmlstr.split('\n') if line.strip()])
    with open(filename, "w") as f:
        f.write(xmlstr)

def run_simulation(steps=1000):
    carbon_pos, params = parse_cml("cnt_quaternion_seed.cml")
    
    lr_schedule = optax.exponential_decay(init_value=0.002, transition_steps=200, decay_rate=0.5)
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)
    compiled_step = create_step_fn(optimizer)
    
    print(f"\nStarting 4D JAX Optimization ({steps} steps)...")
    start_time = time.time()
    
    for i in range(steps):
        params, opt_state, loss = compiled_step(params, opt_state, carbon_pos)
        if i % 100 == 0 or i == steps - 1:
            print(f"Step {i:04d} | System Energy: {loss:,.4f} eV")
            
    print(f"\nRelaxation complete in {time.time() - start_time:.2f} seconds.\n")
    export_relaxed_cml(carbon_pos, params)

if __name__ == "__main__":
    run_simulation()
