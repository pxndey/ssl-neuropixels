Repo-level instructions for OpenCode sessions working on this learned spike-localization project.

---

## Environment: Singularity Only

**Never run Python bare-metal.** The PyTorch/SpikeInterface environment lives inside the Singularity overlay.

```bash
singularity exec --nv --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python script.py"
```

- Always `--nv` for GPU jobs, always `source /ext3/env.sh` first
- `:ro` for read-only; `:rw` only if writing to overlay
- For parallel jobs (multiprocessing, `n_jobs > 1`), **drop `--fakeroot`** — it breaks forked children via FUSE UID remapping
- Use `mp_context="spawn"` for SpikeInterface calls with `n_jobs > 1`

---
