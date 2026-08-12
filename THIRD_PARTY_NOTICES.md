# Third-party notices for MuJoCo simulation

The simulator downloads robot model assets on first use. These files are cached outside the repository and are not LLM outputs.

## MuJoCo Menagerie models

- Source: https://github.com/google-deepmind/mujoco_menagerie
- Pinned commit: `da76818e269b82289eba39808e2fb91d679d6994`
- Sparse paths: `franka_emika_panda`, `unitree_go2`, and `unitree_z1`
- The exact license file shipped beside each model remains in the cached checkout. At the pinned revision, the Panda model is Apache-2.0 and the Unitree Go2/Z1 models use their included BSD-style licenses.

The loader verifies the revision before constructing the simulation and writes `LMRBTP_ASSET_PROVENANCE.json` into a cache it creates.

## Controller references

- `kevinzakka/mjctrl`, commit `f1c82c257248e2e52b0e6c1beaf23c6a2026e4f2`, Apache-2.0: https://github.com/kevinzakka/mjctrl
- `elijah-waichong-chan/go2-convex-mpc`, commit `1c63c6a762779887ab0431fd60db681dede6cb32`, MIT: https://github.com/elijah-waichong-chan/go2-convex-mpc

The local arm controller adapts the damped least-squares differential-IK and null-space approach demonstrated by `mjctrl` to multiple prefixed robots in one model. The local Go2 controller adapts the alternating-contact gait and actuator-torque structure needed by the combined Go2/Z1 scene. It does not bundle or claim to run the upstream convex-MPC solver unchanged.
