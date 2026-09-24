# Model and Asset Provenance

## Source

The model and mesh assets in this directory were inherited from the MyoAssist simulation codebase used as the simulation foundation of this project.

Unless explicitly noted below, these assets were not created or modified by this project. Existing copyright, attribution, provenance, and licensing statements in the source files are preserved and remain authoritative.

## Models Used by the Released Pipeline

- `22muscle_2D/myoLeg22_2D_BASELINE.xml`
- `22muscle_2D/myoLeg22_2D_BASELINE_HIP_EXO.xml`

These files retain OpenSim/MyoSuite lineage, contributor attribution, and modification information in their XML headers. Those headers explicitly identify the underlying 22-muscle model material as licensed under Creative Commons Attribution 3.0; the license text is retained at [`LICENSES/CC-BY-3.0.txt`](LICENSES/CC-BY-3.0.txt). That license reference does not automatically apply to other files in this directory whose retained notices do not establish it.

## Project-Specific Model Adaptation

Git history identifies `22muscle_2D/myoLeg22_2D_BASELINE_HIP_EXO.xml` as the model added by this project. It was derived from the inherited 22-muscle simulation model by adding the bilateral hip-exoskeleton simulation elements used by this project (`Exo_R` and `Exo_L`).

The underlying musculoskeletal model and its inherited anatomical meshes were not newly created by this project.

## Meshes

The anatomical mesh assets used by the formal models were inherited together with the MyoAssist model set and were not modified for this project. Existing upstream licensing and attribution terms remain applicable.

The formal models reference these inherited meshes:

```text
mesh/hat_jaw.stl
mesh/hat_ribs.stl
mesh/hat_skull.stl
mesh/hat_spine.stl
mesh/l_bofoot.stl
mesh/l_clavicle.stl
mesh/l_femur.stl
mesh/l_fibula.stl
mesh/l_foot.stl
mesh/l_hand.stl
mesh/l_humerus.stl
mesh/l_pelvis.stl
mesh/l_radius.stl
mesh/l_scapula.stl
mesh/l_talus.stl
mesh/l_tibia.stl
mesh/l_ulna.stl
mesh/r_bofoot.stl
mesh/r_clavicle.stl
mesh/r_femur.stl
mesh/r_fibula.stl
mesh/r_foot.stl
mesh/r_hand.stl
mesh/r_humerus.stl
mesh/r_pelvis.stl
mesh/r_radius.stl
mesh/r_scapula.stl
mesh/r_talus.stl
mesh/r_tibia.stl
mesh/r_ulna.stl
mesh/sacrum.stl
```

The binary STL files do not contain embedded provenance notices. This repository therefore does not assert a new or broader license for them beyond the terms accompanying their inherited source asset set.

## Other Inherited Models

Other model families and device/CAD assets under this directory were inherited from the MyoAssist codebase. Where individual files include source, copyright, attribution, or license information, those notices remain authoritative.

No new license is asserted by this repository for inherited assets whose upstream terms are not otherwise stated.
