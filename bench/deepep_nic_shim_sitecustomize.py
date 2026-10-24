"""Bench-only shim (#21 W3): DeepEP's get_rdma_gbs() looks for a NIC named mlx5_0 in ibstat; on this host the NICs are
ibp115s0f0/ibp116s0f0, so it returns 0 and BucketBuffer (used by CP) asserts. Wrap it after deep_ep.utils.envs loads
(without importing deep_ep early, which would initialize CUDA before CUDA_VISIBLE_DEVICES is set) and fall back to
DEEPEP_RDMA_GBS_FALLBACK (default 12.5 GB/s = 100 Gb/s, the NIC rate ibstat reports)."""
import importlib.abc
import importlib.util
import os
import sys

TARGET = "deep_ep.utils.envs"


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        exec_orig = spec.loader.exec_module

        def exec_module(module):
            exec_orig(module)
            orig = module.get_rdma_gbs

            def get_rdma_gbs(*a, **k):
                return orig(*a, **k) or float(os.environ.get("DEEPEP_RDMA_GBS_FALLBACK", "12.5"))

            module.get_rdma_gbs = get_rdma_gbs

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Hook())
