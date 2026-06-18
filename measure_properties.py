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
import numpy as np

# --- Physical Constants ---
K_E = 14.3996
DELTA_0 = 0.00386
EPS = 1e-6
CORE_RADIUS = 0.45
LAM_MAG = 0.42       
DELTA_MAG = 0.026    
J_MAG = 0.5          
K_ANISO = 0.2        
K_TENSION = 1000.0   
K_COMPRESS = 50.0    

def parse_relaxed_cml(filename="cnt_relaxed.cml"):
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
def compute_energy_components(params, carbon_pos, e_field_z=0.0):
    pos = params['pos']
    quat = normalize_quaternions(params['quat'])
    delta = params['delta']
    phase = params['phase']
    N = pos.shape[0]
    
    diff_eN = pos[:, None, :] - carbon_pos[None, :, :]
    dist_eN_sq = jnp.sum(diff_eN**2, axis=-1)
    energy_lattice = jnp.sum(-K_E / jnp.sqrt(dist_eN_sq + delta[:, None]**2 + CORE_RADIUS**2))

    diff_ee = pos[:, None, :] - pos[None, :, :]
    safe_dist_ee_sq = jnp.sum(diff_ee**2, axis=-1) + jnp.eye(N) * 1e-12
    dist_ee = jnp.sqrt(safe_dist_ee_sq)
    mask = jnp.triu(jnp.ones((N, N)), k=1)
    
    delta_avg = (delta[:, None] + delta[None, :]) / 2.0
    v_ee = 0.5 * K_E * (1.0 / (dist_ee + EPS) + 1.0 / jnp.sqrt(safe_dist_ee_sq + delta_avg**2))
    energy_repulsion = jnp.sum(v_ee * mask)

    normals = quat_to_loop_normal(quat)
    easy_axis = jnp.array([0.0, 0.0, 1.0]) 
    energy_anisotropy = jnp.sum(K_ANISO * (1.0 - jnp.sum(normals * easy_axis, axis=-1)**2))

    spin_alignment = jnp.einsum('id,jd->ij', normals, normals)
    phase_interaction = jnp.cos(phase[:, None] - phase[None, :])
    v_mag_potential = (LAM_MAG * K_E) / jnp.sqrt(safe_dist_ee_sq + DELTA_MAG**2)
    energy_magnetic = jnp.sum(v_mag_potential * (spin_alignment * phase_interaction) * mask)

    strain = delta - DELTA_0
    energy_spring = jnp.sum(jnp.where(strain > 0, 0.5 * K_TENSION * strain**2, 0.5 * K_COMPRESS * strain**2))

    energy_external = jnp.sum(-e_field_z * pos[:, 2])

    total = energy_lattice + energy_repulsion + energy_anisotropy + energy_magnetic + energy_spring + energy_external
    return total, energy_magnetic

def measure_ionization_energy(carbon_pos, params):
    total_E, _ = compute_energy_components(params, carbon_pos)
    
    params_n_minus_1 = {
        'pos': params['pos'][:-1],
        'quat': params['quat'][:-1],
        'delta': params['delta'][:-1],
        'phase': params['phase'][:-1]
    }
    
    total_E_minus_1, _ = compute_energy_components(params_n_minus_1, carbon_pos)
    ionization_E = total_E_minus_1 - total_E
    
    print(f"\n--- Test A: Ionization Energy ---")
    print(f"System Energy (N electrons):   {total_E:,.2f} eV")
    print(f"System Energy (N-1 electrons): {total_E_minus_1:,.2f} eV")
    print(f"First Ionization Energy:       {ionization_E:,.2f} eV")

def measure_exciton_breakdown(carbon_pos, params):
    print(f"\n--- Test B: Z-Axis Phase-Lock Breakdown (Band Gap) ---")
    
    @jax.jit
    def step(p, opt_state, e_field):
        loss, grads = jax.value_and_grad(lambda p: compute_energy_components(p, carbon_pos, e_field)[0])(p)
        updates, new_opt_state = optimizer.update(grads, opt_state, p)
        new_params = optax.apply_updates(p, updates)
        new_params['quat'] = normalize_quaternions(new_params['quat'])
        new_params['phase'] = jnp.mod(new_params['phase'], 2 * jnp.pi)
        return new_params, new_opt_state
    
    optimizer = optax.adam(learning_rate=0.001)
    opt_state = optimizer.init(params)
    current_params = params
    
    e_fields = np.linspace(0.0, 5.0, 20)
    
    for e_z in e_fields:
        for _ in range(50):
            current_params, opt_state = step(current_params, opt_state, e_z)
            
        _, mag_E = compute_energy_components(current_params, carbon_pos, e_z)
        print(f"Applied Field: {e_z:.2f} eV/A | Magnetic Phase-Lock Energy: {mag_E:,.2f} eV")

if __name__ == "__main__":
    carbons, relaxed_params = parse_relaxed_cml("cnt_relaxed.cml")
    measure_ionization_energy(carbons, relaxed_params)
    measure_exciton_breakdown(carbons, relaxed_params)
