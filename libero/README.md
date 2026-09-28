# LIBERO boundary

`libero/` is the simulator-facing layer.

- `src/libero_gateway/`: multi-session HTTP Gateway and GPU worker pool.
- `client/`: small launch and Agent clients.
- `runtime/full/`: trusted lease proxy, artifact recorder, SDK, and workspace
  materializer for ordinary Level-4 observations.
- `runtime/strict/`: the frozen Table 10
  `l4-no-tactile-no-gripper-proprio-v1` projection.
- `patches/`: the camera-only LIBERO patch used by the Gateway.
- `env.yml`: reference simulator environment.

The Gateway owns task allocation, session tokens, accepted Steps, success checks,
and simulator state. The native Agent never receives the remote session token. It
sees a localhost proxy and the generated `libero_sdk.py` inside its isolated
workspace.

The strict profile retains RGB, depth, calibration, initial bboxes, Panda arm joint
position/velocity, and end-effector pose/velocity. It removes all direct gripper
proprioception and recursively rejects keys matching
`gripper|force|torque|wrench|tactile`. The action interface is unchanged.

From the repository root, `make bootstrap` creates the pinned Conda environment,
checks out the patched LIBERO source, installs the Gateway, and writes a
non-interactive LIBERO path config below `.runtime/`. Start the real backend with:

```bash
conda activate agent-for-robot-libero
LIBERO_GATEWAY_GPU_IDS=0,1,2,3 ./scripts/run_libero_gateway.sh
```

The launcher creates the required static-token hash, audit log, and evaluation
database at runtime; none is checked into the release.

For API fields see [docs/AGENT_API.md](docs/AGENT_API.md). For Gateway internals see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tests

```bash
PYTHONPATH=libero/src python -m unittest discover -s libero/tests -p 'test_*.py'
PYTHONPATH=libero/runtime/strict python -m unittest \
  libero/runtime/strict/test_strict_projection.py
```
