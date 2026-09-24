# Third-Party Notices

The repository's root `LICENSE` governs original project contributions. It does not replace the licenses, copyrights, or attribution requirements that apply to the material below.

## MyoSuite

This repository contains a vendored copy of [MyoSuite](https://github.com/MyoHub/myosuite), corresponding to upstream version `v2.8.3`, commit `2be010609f63eb1e577eef395871aedbc1c442f2`.

MyoSuite is licensed under the Apache License 2.0. A retained copy of that license is at [`myosuite/LICENSE`](myosuite/LICENSE). The local tree contains project-specific additions and modifications; existing upstream source-file copyright and license headers are preserved, and modified files are marked accordingly. Existing license files in the vendored SimHive component directories remain applicable to those components.

The root project's copyright does not replace MyoSuite or SimHive copyrights.

## Model and Mesh Assets

The [`models/`](models/) directory was inherited from the MyoAssist simulation codebase used as the simulation foundation of this project. Most model and mesh assets were inherited without modification. Existing file-level provenance, copyright, attribution, and license statements are preserved and remain authoritative.

This project added `models/22muscle_2D/myoLeg22_2D_BASELINE_HIP_EXO.xml`, an exoskeleton model variant derived from the inherited 22-muscle simulation model by adding the bilateral hip-exoskeleton simulation elements used by this project. The formal released pipeline uses that model and `models/22muscle_2D/myoLeg22_2D_BASELINE.xml`.

The headers of these two formal models identify their OpenSim/MyoSuite lineage, contributors, modifications, and Creative Commons Attribution 3.0 licensing. The applicable license text is retained at [`models/LICENSES/CC-BY-3.0.txt`](models/LICENSES/CC-BY-3.0.txt). This statement is limited to the model material for which the retained headers establish that license; it does not assert that every file under `models/` is CC BY 3.0.

The anatomical meshes used by the formal models were inherited with the same MyoAssist model asset set and were not modified by this project. The root Apache-2.0 license does not override any third-party terms applying to inherited model or mesh assets. See [`models/PROVENANCE.md`](models/PROVENANCE.md) for details.

## Separately Installed Dependencies

Ordinary dependencies installed through `requirements.txt`, including MuJoCo, NumPy, PyTorch, Gymnasium, Stable-Baselines3, and h5py, are installed separately and are not vendored source code merely because this project depends on them. Their respective distributions provide their license information.
