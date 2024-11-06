# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

import torch
import utils.dist as dist_utils


class AccmStageINR:
    def __init__(
        self,
        scalar_metric_names,
        vector_metric_names=(),
        vector_metric_lengths=(),
        device="cpu",
    ):
        self.device = device

        assert len(vector_metric_lengths) == len(vector_metric_names)

        self.scalar_metric_names = scalar_metric_names
        self.vector_metric_names = vector_metric_names
        self.metrics_sum = {}

        for n in self.scalar_metric_names:
            self.metrics_sum[n] = torch.zeros(1, device=self.device)

        for n, length in zip(self.vector_metric_names, vector_metric_lengths):
            self.metrics_sum[n] = torch.zeros(length, device=self.device)

        self.counter = 0
        self.Summary.scalar_metric_names = self.scalar_metric_names
        self.Summary.vector_metric_names = self.vector_metric_names

    @torch.no_grad()
    def update(self, metrics_to_add, count=None, sync=False, distenv=None):
        # we assume that value is simultaneously None (or not None) for every process
        metrics_to_add = {
            name: value for (name, value) in metrics_to_add.items() if value is not None
        }

        if sync:
            for name, value in metrics_to_add.items():
                gathered_value = dist_utils.all_gather_cat(distenv, value.unsqueeze(0))
                gathered_value = gathered_value.sum(dim=0).detach()
                metrics_to_add[name] = gathered_value

        for name, value in metrics_to_add.items():
            if name not in self.metrics_sum:
                raise KeyError(f"unexpected metric name: {name}")
            self.metrics_sum[name] += value

        self.counter += count if not sync else count * distenv.world_size

    @torch.no_grad()
    def get_summary(self, n_samples=None):
        n_samples = n_samples if n_samples else self.counter
        return self.Summary({k: v / n_samples for k, v in self.metrics_sum.items()})

    class Summary:
        scalar_metric_names = ()
        vector_metric_names = ()

        def __init__(self, metrics):
            for key, value in metrics.items():
                self[key] = value

        def print_line(self):
            reprs = []
            for k in self.scalar_metric_names:
                v = self[k]
                repr = f"{k}: {v.item():.4f}"
                reprs.append(repr)

            for k in self.vector_metric_names:
                v = self[k]
                array_repr = ",".join([f"{v_i.item():.4f}" for v_i in v])
                repr = f"{k}: [{array_repr}]"
                reprs.append(repr)

            return ", ".join(reprs)

        def tb_like(self):
            tb_summary = {}
            for k in self.scalar_metric_names:
                v = self[k]
                tb_summary[f"loss/{k}"] = v

            for k in self.vector_metric_names:
                v = self[k]
                for i, v_i in enumerate(v):
                    tb_summary[f"loss/{k}_{i}"] = v_i

            return tb_summary

        def __getitem__(self, item):
            return getattr(self, item)

        def __setitem__(self, key, value):
            setattr(self, key, value)
