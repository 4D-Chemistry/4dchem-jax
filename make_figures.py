#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py  --  Reproduce the two figures in the MinkiEngine paper.

  fig_sweep.png : phase-lock energy vs applied axial field, (12,4) vs (12,12)
  fig_tubes.png : relaxed pi-electron shell over a clean and a dented (12,4) tube

Everything heavy is imported from the existing pipeline -- foundry.py builds the
scaffold, minkowski_engine.py supplies the energy, its gradient (via JAX autodiff),
and the JIT-compiled Adam step. This script only orchestrates and plots, which is
the point: the physics core is written once and reused.

Run from the directory containing foundry.py and minkowski_engine.py:
    python3 make_figures.py
"""

import importlib.util
import xml.etree.ElementTree as ET
import time

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Load the existing pipeline modules by path (no packaging needed).
# ----------------------------------------------------------------------
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

foundry = _load("foundry", "foundry.py")
eng     = _load("eng", "minkowski_engine.py")

NS = {"cml": "http://www.xml-cml.org/schema"}


# ----------------------------------------------------------------------
# CML I/O helper (same parameter layout the engine writes).
# ----------------------------------------------------------------------
def parse_cml(filename):
    root = ET.parse(filename).getroot()
    C, P, Q, D, Ph = [], [], [], [], []
    for atom in root.findall(".//cml:atom", NS):
        x = float(atom.attrib["x3"]); y = float(atom.attrib["y3"]); z = float(atom.attrib["z3"])
        if atom.attrib["elementType"] == "C":
            C.append([x, y, z])
        else:
            P.append([x, y, z])
            s = {k.attrib["dictRef"]: float(k.text) for k in atom.findall("cml:scalar", NS)}
            Q.append([s["minkowski:qw"], s["minkowski:qx"], s["minkowski:qy"], s["minkowski:qz"]])
            D.append(s["minkowski:delta"]); Ph.append(s["minkowski:phase"])
    params = {"pos": jnp.array(P), "quat": jnp.array(Q),
              "delta": jnp.array(D), "phase": jnp.array(Ph)}
    return np.array(C), params


# ----------------------------------------------------------------------
# Relaxation: identical schedule to minkowski_engine.run_simulation.
# ----------------------------------------------------------------------
def relax(carbon_pos, params, steps=1000):
    Cj = jnp.array(carbon_pos)
    sched = optax.exponential_decay(init_value=0.002, transition_steps=200, decay_rate=0.5)
    opt = optax.adam(sched)
    opt_state = opt.init(params)
    step = eng.create_step_fn(opt)
    loss = None
    for _ in range(steps):
        params, opt_state, loss = step(params, opt_state, Cj)
    return params, float(loss)


# ----------------------------------------------------------------------
# The magnetic phase-lock energy, isolated exactly as the engine defines
# it (the order parameter we track under field).
# ----------------------------------------------------------------------
def phase_lock_energy(params, carbon_pos):
    pos = params["pos"]
    quat = eng.normalize_quaternions(params["quat"])
    phase = params["phase"]
    N = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]
    sq = jnp.sum(diff**2, axis=-1) + jnp.eye(N) * 1e-12
    mask = jnp.triu(jnp.ones((N, N)), k=1)
    normals = eng.quat_to_loop_normal(quat)
    spin_align = jnp.einsum("id,jd->ij", normals, normals)
    phase_int = jnp.cos(phase[:, None] - phase[None, :])
    v = (eng.LAM_MAG * eng.K_E) / jnp.sqrt(sq + eng.DELTA_MAG**2)
    return jnp.sum(v * (spin_align * phase_int) * mask)


# ----------------------------------------------------------------------
# Applied-field sweep. The external term -E_z * z is added to the engine
# energy; gradients (forces) still come straight from autodiff.
# ----------------------------------------------------------------------
def field_sweep(carbon_pos, params, fields, inner_steps=50):
    Cj = jnp.array(carbon_pos)

    def energy_with_field(p, ez):
        return eng.compute_total_energy(p, Cj) - jnp.sum(ez * p["pos"][:, 2])

    opt = optax.adam(learning_rate=0.001)
    opt_state = opt.init(params)
    cur = params

    @jax.jit
    def step(p, st, ez):
        loss, grads = jax.value_and_grad(energy_with_field)(p, ez)
        updates, st = opt.update(grads, st, p)
        p = optax.apply_updates(p, updates)
        p["quat"] = eng.normalize_quaternions(p["quat"])
        p["phase"] = jnp.mod(p["phase"], 2 * jnp.pi)
        return p, st

    curve = []
    for ez in fields:
        for _ in range(inner_steps):
            cur, opt_state = step(cur, opt_state, float(ez))
        curve.append(float(phase_lock_energy(cur, carbon_pos)))
    return np.array(curve)


# ----------------------------------------------------------------------
# Make a radial dent: push a localised patch of carbon cores inward.
# ----------------------------------------------------------------------
def dent_scaffold(carbon_pos, push=0.45, half_len=2.5, half_ang=0.9):
    C = carbon_pos.copy()
    z_mid = (C[:, 2].min() + C[:, 2].max()) / 2.0
    ang = np.arctan2(C[:, 1], C[:, 0])
    hit = (np.abs(C[:, 2] - z_mid) < half_len) & (np.abs(ang) < half_ang)
    C[hit, 0] *= (1.0 - push)
    C[hit, 1] *= (1.0 - push)
    return C, hit


# ----------------------------------------------------------------------
# Figure 1: field-driven collapse of the phase-locked state.
# ----------------------------------------------------------------------
def figure_sweep():
    fields = np.linspace(0.0, 6.0, 25)
    curves = {}
    for n, m, label in [(12, 4, "(12,4) chiral"), (12, 12, "(12,12) armchair")]:
        carbons, electrons, _ = foundry.generate_quaternion_cnt(n, m, 240)
        foundry.export_quaternion_cml(carbons, electrons, n, m, filename=f"seed_{n}_{m}.cml")
        C, p = parse_cml(f"seed_{n}_{m}.cml")
        t0 = time.time(); p, E = relax(C, p); t_relax = time.time() - t0
        t0 = time.time(); curve = field_sweep(C, p, fields); t_sweep = time.time() - t0
        curves[label] = curve
        print(f"{label:18s} relax {t_relax:4.1f}s (E={E:,.0f} eV)  sweep {t_sweep:4.1f}s")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fields, curves["(12,4) chiral"],    "o-", ms=4, label="(12,4) chiral")
    ax.plot(fields, curves["(12,12) armchair"], "s-", ms=4, label="(12,12) armchair")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(r"Applied axial field  $E_z$  (eV/$\AA$)")
    ax.set_ylabel("Phase-lock energy (eV)")
    ax.set_title("Field-driven collapse of the phase-locked state")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("fig_sweep.png", dpi=140)
    print("wrote fig_sweep.png")


# ----------------------------------------------------------------------
# Figure 2: clean vs dented relaxed tube, electrons coloured by phase.
# ----------------------------------------------------------------------
def figure_tubes():
    carbons, electrons, _ = foundry.generate_quaternion_cnt(12, 4, 240)
    foundry.export_quaternion_cml(carbons, electrons, 12, 4, filename="seed_12_4.cml")

    C, p = parse_cml("seed_12_4.cml")
    p, E_clean = relax(C, p)
    pos_clean, phase_clean = np.asarray(p["pos"]), np.asarray(p["phase"])

    Cd, hit = dent_scaffold(C, push=0.45)
    _, p2 = parse_cml("seed_12_4.cml")          # fresh electron seed
    p2, E_dent = relax(Cd, p2)
    pos_dent, phase_dent = np.asarray(p2["pos"]), np.asarray(p2["phase"])

    r_dent = np.sqrt(pos_dent[:, 0]**2 + pos_dent[:, 1]**2)
    print(f"clean  E={E_clean:,.0f} eV | dented E={E_dent:,.0f} eV "
          f"| carbons dented={int(hit.sum())} | dented shell std={r_dent.std():.2f} A")

    fig = plt.figure(figsize=(9, 5))
    panels = [(pos_clean, C,  phase_clean, "Clean (12,4)"),
              (pos_dent,  Cd, phase_dent,  "Dented (12,4)")]
    for k, (pp, Cx, ph, ttl) in enumerate(panels):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(Cx[:, 0], Cx[:, 1], Cx[:, 2], c="tab:gray", s=14, alpha=0.55, edgecolors="none")
        ax.scatter(pp[:, 0], pp[:, 1], pp[:, 2], c=ph, cmap="hsv", s=14, alpha=0.9, edgecolors="none")
        ax.view_init(elev=8, azim=0)            # side-on so the dent reads as an inward notch
        ax.set_title(ttl, fontsize=11)
        ax.set_box_aspect((1, 1, 2.6)); ax.set_axis_off()
    fig.suptitle(r"Relaxed $\pi$-electron shell (colour = temporal phase); grey = C scaffold",
                 fontsize=10, y=0.97)
    fig.tight_layout(); fig.savefig("fig_tubes.png", dpi=140, bbox_inches="tight")
    print("wrote fig_tubes.png")


if __name__ == "__main__":
    figure_sweep()
    figure_tubes()
    print("done.")
