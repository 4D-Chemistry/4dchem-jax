# 4D-Chemistry ⚡️

**Fast, scalable, classical 4D electrostatic engines for macroscopic materials design.**

Density Functional Theory (DFT) is the gold standard for electronic structure, but its $O(N^3)$ scaling strictly limits dynamic simulations of mesoscopic structures. The **4D-Chemistry** organization provides open-source, hardware-accelerated computational tools that bypass this quantum bottleneck. 

By modeling unbound electrons as fully parameterized, spatial-temporal kinematic loops interacting within a rigid geometric lattice—and utilizing a temporal displacement parameter ($\delta$) to regularize the $1/r$ Coulomb singularity—we reproduce macroscopic electronic and magnetic phenomena using pure classical kinematics.

No wavefunctions. No supercells. No boundary artifacts. Just clean, exact gradients and lightning-fast optimization.

---

## 📂 Our Repositories

To keep our tooling modular, the 4D-Chemistry suite is divided into specific domain repositories:

### 1. [4dchem-jax](https://github.com/4D-Chemistry/4dchem-jax)
**The Minkowski Engine: A Differentiable 4D Electrostatic Solver for Carbon Lattices**
Built entirely in Google's `JAX` ecosystem, this engine is designed for the rapid geometric screening of mesoscopic carbon nanotubes and graphene sheets. 
* Uses continuous unit quaternions to eliminate Gimbal lock during spin relaxation.
* Automatically derives exact force vectors via `jax.value_and_grad`.
* Differentiates between metallic conduction and semiconducting band gaps mechanically in seconds on standard laptop hardware.
* **Includes:** `foundry.py`, `minkowski_engine.py`, `measure_properties.py`

### 2. [4dchem-mag](https://github.com/4D-Chemistry/4dchem-mag)
**Rare-Earth-Free Magnetic Alloy Prospector & Tumble Classifier**
An $O(N^2)$ Hamiltonian screening tool for discovering and optimizing next-generation permanent magnets. It evaluates crystal structures to predict magnetic performance based on the 4D magnetic phase-locking of half-photon loops.
* **Stage I:** Predicts Curie Temperature ($T_C$) and saturation magnetization ($M_s$).
* **Stage II:** Classifies magnetic hardness (SOFT vs. HARD), calculates FMR frequencies, and evaluates domain wall energy densities.
* **Stage III:** Ranks candidates using a custom Figure of Merit.
* Evaluates the novel phase-locking advantages of Graphene-intercalated transition metals.
* **Includes:** `mag_prospector_v2.py`, `mag_stages.py`, `mag_tumble_v1.py`

### 3. [4dchem-core](https://github.com/4D-Chemistry/4dchem-core)
**The Foundational 4D Chemistry Suite**
The core validation scripts that anchor the 4D physical constants to fundamental experimental data. 
* Solves the Hydrogen ground state to validate the temporal radius ($\delta_0 = 0.00386$ Å).
* Solves Lithium to validate "Exchange Energy" as magnetic phase-locking.
* Derives Nitrogen VSEPR geometry ($sp^3$) purely from classical magnetic packing.
* **Includes:** `4D_Chemistry_Suite.py`

---

## 📖 Publications & Academic Use

The mathematical foundations and physical calibrations of these engines are detailed in the following preprints. If you use our code in your materials research, please cite the relevant papers and Zenodo software DOIs.

* **4D Electrostatics & Carbon Lattices:**
  > Sinclair, D. A. (2026). *A Differentiable Pairwise-Relaxation Engine in JAX for Fast Screening of Carbon-Nanotube Geometries.* ChemRxiv. [DOI]
* **Magnetic Prospecting:**
  > Sinclair, D. A. (2026). *A Predictive 4D Electrostatic Framework for Rare-Earth-Free Permanent Magnet Screening.* ChemRxiv. [DOI]

*(Note: Corresponding Zenodo DOIs for the exact software releases can be found in the README of each respective repository).*

---

## ⚖️ Licensing & Commercial Use

All software in the 4D-Chemistry suite is released under the **GNU General Public License v3.0 (GPLv3)** to ensure that improvements to the physics engines remain open and available to the academic community.

**Commercial Exemption (The Beerware Clause):** If a commercial materials lab or startup wishes to use these tools to design proprietary, closed-source architectures (e.g., battery anodes, proprietary magnetic alloys) without being subject to the GPLv3 copyleft requirements, they must contact the author to negotiate a commercial buyout. (Historically, the price of this buyout is a very good bottle of Scotch or a small cask of ale).

Contact: `david@s-hull.org`
