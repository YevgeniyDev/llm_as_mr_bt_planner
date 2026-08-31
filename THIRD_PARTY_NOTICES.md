# Third-party notices

## LLM-as-BT-Planner / KIOS comparison source

- Official repository: https://github.com/ProNeverFake/kios
- Pinned commit: `e9f16f5bd110ab647242077d55d5cb0a71e4fcd9`
- Archive SHA-256: `32468fcbb0be4df496c273968af098134bea87fd588e468ec75cabba124e8bde`
- Software license: MIT
- Paper: J. Ao et al., "LLM as BT-Planner: Leveraging LLMs for Behavior Tree Generation in Robot Task Planning," ICRA 2025 / arXiv:2409.10444: https://arxiv.org/abs/2409.10444

The official repository is downloaded only through `lmrbtp compare llm-as-bt-planner prepare` into the selected ignored output directory. The archive, expected license, required architecture files, and all extracted files are verified and hashed. Upstream source is not bundled in this package. The local common-domain runner is a compatibility implementation of the published KIOS interfaces and generation schemes; it replaces the authors' Neo4j, WebSocket skill server, Panda task domains, and hardware execution with this project's declared capability contracts and evaluators.

## LLM-BT comparison source and parser checkpoint

- Official repository: https://github.com/henryhaotian/LLM-BT
- Pinned commit: `c69c18d0cf4b78f166ed352fc0fa8470823b32f6`
- Released parser checkpoint SHA-256: `e77ac3903c0f0fa46f1336e4d8e14de3e17986b61e3c8f3e32663ca3ce264eb8`
- Released parser checkpoint size: 265,510,949 bytes
- Paper: H. Zhou, Y. Lin, L. Yan, J. Zhu, and H. Min, "LLM-BT: Performing Robotic Adaptive Tasks based on Large Language Models and Behavior Trees," ICRA 2024 / arXiv:2404.05134: https://arxiv.org/abs/2404.05134
- Project-wide software license: not declared in the pinned repository
- Parser-model license: not declared in the pinned repository
- Embedded `LLMBT/BTsUpdate/core/LICENSE`: MIT, for the included Michele Colledanchise BT core only

The preparation command downloads only 16 method-defining files and, unless explicitly omitted, the released DistilBERT checkpoint into the selected ignored output directory. It verifies their immutable hashes, checkpoint size, architecture, and label vocabulary; no upstream file is bundled by this package. The embedded core license must not be interpreted as a license for the complete LLM-BT repository or parser model. The local common-domain implementation retains the published reasoning/parser/ATL-expansion boundary while replacing V-REP 3.6.2, the Qt editor, original scenes, and physical perception with declared common contracts and evaluators. It must not be described as an unmodified execution of the authors' application.

## BETR-XP-LLM comparison source

- Official repository: https://github.com/jstyrud/BETR-XP-LLM
- Pinned commit: `bf83bda4b8921eea7fe0b8756daacb7da9fb6133`
- Pinned archive SHA-256: `54bc787eb7ae78901e3d9dee3929dfc1d90bf0412a246428a3e2b4dc7ecb370f`
- Paper: J. Styrud, A. K. Iovino, M. Standar, K. LeBlanc, and C. Smith, "A Framework for Automated Behavior Tree Generation in Collaborative Robotic Applications," ICRA 2025 / arXiv:2409.13356: https://arxiv.org/abs/2409.13356
- IEEE DOI: https://doi.org/10.1109/ICRA55743.2025.11127942
- Software license: BSD-3-Clause; copyright (c) 2024, ABB

The preparation command downloads the complete 120-file repository archive into the selected ignored output directory and verifies its immutable archive hash, every extracted file hash, required method-defining prompts/planner/tests, and license. No upstream source is bundled by this package. The local common-domain runner is informed by the official source but replaces its ABB YuMi, Azure OpenAI, custom PyTrees, RWS, collision-planning, and perception stack with the benchmark's declared capability contracts and evaluators. The dropped-object recovery uses the paper's missing-parameter resolution mechanism for a generic pickup-location binding; it must not be presented as one of the authors' ten original missing-precondition experiments or as an unmodified execution of their application.

## LLM-HBT paper and author project page

- Author project page: https://github.com/baoziweiyuebing/LLM-HBT
- Pinned project commit: `17ff0ad9fc8e0f5ef3534086589cfa812b20cf29`
- Project archive SHA-256: `d7e22fc0ce6ea5c30dfdd4f10da7ccf914e9ef1b230f3db9c8140bc4c7f96002`
- Paper: C. Wang, J. Sun, Y. Zhang, M. Zhang, and C. Wu, “LLM-HBT: Dynamic Behavior Tree Construction for Adaptive Coordination in Heterogeneous Robots,” arXiv:2510.09963v1: https://arxiv.org/abs/2510.09963v1
- arXiv v1 source SHA-256: `bb40eff629f12f7a9ae58e989abe518a1092f73a9a26288ec4f361f17f29ca28`
- Paper license: arXiv non-exclusive distribution license
- Executable author code: not released; the pinned page says `Code Repository: Coming Soon`
- Software license: not declared because the repository contains no software implementation

`lmrbtp compare llm-hbt prepare` downloads the exact project-page and paper-source archives into the selected ignored output directory, verifies their immutable hashes and method-defining files, and records the unavailable-code boundary. No upstream content is bundled by this package. The local runner is a paper-based clean-room common-domain reproduction. Its strict JSON prompts, response grammar, default model, temperature, and seed are disclosed reproduction choices because the paper does not publish those details. It must not be described as official author code or an exact reproduction of unreported inference settings.

## MRBTP comparison source

- Official repository: https://github.com/DIDS-EI/MRBTP
- Pinned commit: `3d6bd240aa2903245b2335711a97ee394f174313`
- Pinned archive SHA-256: `959d3559d10721b45629074ca944d95df92ba73bc44a9f6a57332a28dcd20030`
- Paper: Y. Cai et al., "MRBTP: Efficient Multi-Robot Behavior Tree Planning and Collaboration," AAAI 2025 / arXiv:2502.18072: https://arxiv.org/abs/2502.18072
- AAAI DOI: https://doi.org/10.1609/aaai.v39i14.33594
- Software license: MIT; copyright (c) 2024 MABTPG

`lmrbtp compare mrbtp prepare` downloads the complete pinned repository into the selected ignored output directory and verifies the immutable archive hash, every extracted file, required planner/action-node sources, and MIT license. No upstream source is bundled by this package. The local runner is a source-aligned common-domain port of the FIFO MRBTP/MABTP path with composite actions and the optional LLM subtree plugin disabled. It replaces MiniGrid, VirtualHome, and native robot execution with the benchmark's declared capability contracts and evaluators; it must not be described as an unmodified execution of the authors' application.

## MuJoCo simulation

The simulator downloads robot model assets on first use. These files are cached outside the repository and are not LLM outputs.

## MuJoCo Menagerie models

- Source: https://github.com/google-deepmind/mujoco_menagerie
- Pinned commit: `da76818e269b82289eba39808e2fb91d679d6994`
- Sparse paths: `franka_emika_panda`, `unitree_go2`, and `unitree_z1`
- The exact license file shipped beside each model remains in the cached checkout. At the pinned revision, the Panda model is Apache-2.0 and the Unitree Go2/Z1 models use their included BSD-style licenses.

The loader verifies the revision before constructing the simulation and writes `LMRBTP_ASSET_PROVENANCE.json` into a cache it creates.

## Five-agent inspection robot sources

- Unitree MuJoCo (official B2 MJCF and meshes): https://github.com/unitreerobotics/unitree_mujoco
- Pinned Unitree commit: `4134cb5dc7ff1ba7f484deda48b5274b58694519`
- Unitree repository license: BSD-3-Clause
- Clearpath Husky (official description and meshes): https://github.com/husky/husky/tree/noetic-devel
- Pinned Husky commit: `41e15d283a8d955938204e79554a875264417bb9`
- Husky repository license: BSD-3-Clause

The inspection asset loader sparse-checks out both exact revisions into the external cache, rejects
a dirty checkout, and records their provenance. The scene attaches the official Unitree B2 MJCF
to the task-level base controller. Husky is a local MuJoCo adaptation using the official A200 URDF
dimensions, wheel positions/radius, mass, and inertial values; it is not an official Clearpath
MuJoCo model. Panda and Z1 models come from the pinned MuJoCo Menagerie checkout above. The mobile
controllers are deterministic task-level simulation controllers and are not presented as Unitree
or Clearpath real-robot control software.

## Controller references

- `kevinzakka/mjctrl`, commit `f1c82c257248e2e52b0e6c1beaf23c6a2026e4f2`, Apache-2.0: https://github.com/kevinzakka/mjctrl
- `elijah-waichong-chan/go2-convex-mpc`, commit `1c63c6a762779887ab0431fd60db681dede6cb32`, MIT: https://github.com/elijah-waichong-chan/go2-convex-mpc

The local arm controller adapts the damped least-squares differential-IK and null-space approach demonstrated by `mjctrl` to multiple prefixed robots in one model. The local Go2 controller adapts the alternating-contact gait and actuator-torque structure needed by the combined Go2/Z1 scene. It does not bundle or claim to run the upstream convex-MPC solver unchanged.
