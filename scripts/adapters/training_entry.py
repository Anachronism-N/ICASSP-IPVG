"""Set a per-rank seed and optionally observe the pinned upstream training loop."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import resource
import runpy
import sys
import time


class TrainingObservations:
    """Observe native optimizer calls, not Accelerate's microstep attempts."""

    def __init__(self, directory):
        import torch
        self.torch = torch
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.parameters = []
        self.initial = None
        self.handles = []
        self.report = {
            "schema_version": 1, "status": "recording",
            "rank": int(os.environ.get("RANK", "0")),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "actual_optimizer_steps": 0, "backward_calls": 0,
            "initial_snapshot_taken": False, "losses": [], "steps": [],
            "errors": [], "optimizer_step_source": "native AdamW step pre/post hooks",
            "resource_start": self.resources(),
        }
        self.write()

    def resources(self):
        result = {"process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "elapsed_wall_s": time.monotonic() - self.started,
                  "cuda_initialized": self.torch.cuda.is_initialized()}
        if result["cuda_initialized"]:
            device = self.torch.cuda.current_device()
            result.update({"cuda_device": device, "gpu_name": self.torch.cuda.get_device_name(device),
                           "allocated_bytes": self.torch.cuda.memory_allocated(device),
                           "reserved_bytes": self.torch.cuda.memory_reserved(device),
                           "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(device),
                           "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(device)})
        return result

    def write(self):
        path = self.directory / ("rank_%s.json" % self.report["rank"])
        path.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def bind_accelerator(self, accelerator):
        self.report.update(rank=accelerator.process_index, world_size=accelerator.num_processes,
                           device=str(accelerator.device))

    def fail(self, error):
        self.report["status"] = "failed"
        text = str(error)
        if text not in self.report["errors"]:
            self.report["errors"].append(text)
        self.report["resource_end"] = self.resources()
        self.write()

    def loss(self, accelerator, loss):
        self.bind_accelerator(accelerator)
        value = loss.detach()
        finite = bool(self.torch.isfinite(value).all().item())
        self.report["backward_calls"] += 1
        self.report["losses"].append({"backward_call": self.report["backward_calls"],
                                      "value": float(value.item()) if finite else None,
                                      "finite": finite, "sync_gradients": bool(accelerator.sync_gradients)})
        self.write()
        if not finite:
            raise RuntimeError("Pilot observed nonfinite loss")

    def attach(self, optimizer):
        if self.parameters:
            raise RuntimeError("Pilot observations support one native AdamW optimizer only")
        self.parameters = [parameter for group in optimizer.param_groups
                           for parameter in group["params"] if parameter.requires_grad]
        self.report["trainable_tensor_count"] = len(self.parameters)
        self.report["trainable_parameter_count"] = sum(parameter.numel() for parameter in self.parameters)
        self.handles.extend([optimizer.register_step_pre_hook(self.before_step),
                             optimizer.register_step_post_hook(self.after_step)])
        self.write()

    def before_step(self, optimizer, args, kwargs):
        if not self.parameters:
            raise RuntimeError("Pilot optimizer has no trainable parameters")
        if self.initial is None:
            # This runs after accelerator.prepare/DDP broadcast and before the first real update.
            self.initial = [parameter.detach().cpu().clone() for parameter in self.parameters]
            self.report["initial_snapshot_taken"] = True
        finite, nonzero, present, maximum = True, 0, 0, 0.0
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            if gradient.is_sparse:
                gradient = gradient.coalesce().values()
            present += 1
            valid = bool(self.torch.isfinite(gradient).all().item())
            finite = finite and valid
            if valid:
                magnitude = float(gradient.abs().max().item()) if gradient.numel() else 0.0
                maximum = max(maximum, magnitude)
                nonzero += int(magnitude > 0)
        self.report["steps"].append({"optimizer_step": self.report["actual_optimizer_steps"] + 1,
                                     "backward_calls": self.report["backward_calls"],
                                     "gradient_tensors": present, "nonzero_gradient_tensors": nonzero,
                                     "gradients_finite": finite, "gradient_max_abs": maximum,
                                     "completed": False})
        self.write()
        if not finite:
            raise RuntimeError("Pilot observed nonfinite trainable gradients")

    def after_step(self, optimizer, args, kwargs):
        self.report["actual_optimizer_steps"] += 1
        self.report["steps"][-1]["completed"] = True
        self.write()

    def parameter_summary(self, model):
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        initial_hash, final_hash = hashlib.sha256(), hashlib.sha256()
        maximum, total, changed, finite = 0.0, 0.0, 0, True
        if self.initial is None:
            return {"initial_trainable_sha256": None, "final_trainable_sha256": None,
                    "parameter_max_abs_change": 0.0, "parameter_total_abs_change": 0.0,
                    "changed_trainable_tensors": 0, "parameters_finite": False}
        for parameter, initial in zip(self.parameters, self.initial):
            current = parameter.detach().cpu().contiguous()
            header = json.dumps([names[id(parameter)], str(current.dtype), list(current.shape)]).encode()
            for digest, tensor in [(initial_hash, initial), (final_hash, current)]:
                digest.update(header)
                digest.update(tensor.contiguous().reshape(-1).view(self.torch.uint8).numpy().tobytes())
            valid = bool(self.torch.isfinite(current).all().item())
            finite = finite and valid and bool(self.torch.isfinite(initial).all().item())
            if valid and bool(self.torch.isfinite(initial).all().item()):
                difference = (current.double() - initial.double()).abs()
                change = float(difference.max().item()) if difference.numel() else 0.0
                maximum = max(maximum, change)
                total += float(difference.sum().item())
                changed += int(change > 0)
        return {"initial_trainable_sha256": initial_hash.hexdigest(),
                "final_trainable_sha256": final_hash.hexdigest(),
                "parameter_max_abs_change": maximum, "parameter_total_abs_change": total,
                "changed_trainable_tensors": changed, "parameters_finite": finite}

    def finish(self, accelerator, model, gather_object):
        self.bind_accelerator(accelerator)
        self.report.update(self.parameter_summary(accelerator.unwrap_model(model)))
        errors = self.report["errors"]
        if not self.report["losses"]:
            errors.append("No backward loss observed")
        if not self.report["actual_optimizer_steps"]:
            errors.append("No completed native optimizer step observed")
        if not any(step["completed"] and step["nonzero_gradient_tensors"] for step in self.report["steps"]):
            errors.append("Trainable gradients were zero or absent at all completed optimizer steps")
        if not self.report["parameters_finite"]:
            errors.append("Initial or final trainable parameters were absent or nonfinite")
        if not self.report["parameter_max_abs_change"]:
            errors.append("No trainable parameter update observed")
        summary = {key: self.report[key] for key in
                   ("rank", "actual_optimizer_steps", "initial_trainable_sha256", "final_trainable_sha256", "errors")}
        # Never enter a collective from the exception handler: a failed rank may be alone there.
        gathered = gather_object([summary]) if accelerator.num_processes > 1 else [summary]
        self.report["rank_summaries"] = gathered
        ranks = sorted(item["rank"] for item in gathered)
        if ranks != list(range(accelerator.num_processes)):
            errors.append("Rank evidence does not cover the declared world size")
        if any(item["errors"] for item in gathered):
            errors.append("One or more ranks failed local training observations")
        synchronized = all(len({item[key] for item in gathered}) == 1 for key in
                           ("actual_optimizer_steps", "initial_trainable_sha256", "final_trainable_sha256"))
        self.report["rank_parameters_synchronized"] = synchronized
        if not synchronized:
            errors.append("Rank optimizer step counts or initial/final trainable hashes differ")
        self.report["status"] = "failed" if errors else "passed"
        self.report["resource_end"] = self.resources()
        self.write()
        if errors:
            raise RuntimeError("Pilot training observations failed: " + "; ".join(errors))

    def close(self):
        for handle in self.handles:
            handle.remove()


@contextmanager
def observe_training(directory):
    observations = TrainingObservations(directory)
    try:
        from accelerate import Accelerator
        from accelerate.utils import gather_object
        from diffsynth.diffusion import runner
        from diffsynth.diffusion.logger import ModelLogger
    except BaseException as error:
        observations.fail(error)
        raise
    original_optimizer = runner.get_optimizer_class
    original_backward = Accelerator.backward
    original_end = ModelLogger.on_training_end

    def optimizer_class(customized_optimizer=None):
        if customized_optimizer is not None:
            raise ValueError("Pilot observations require the pinned default AdamW optimizer")
        base = original_optimizer(customized_optimizer)

        class ObservedAdamW(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                observations.attach(self)

        return ObservedAdamW

    def backward(accelerator, loss, *args, **kwargs):
        observations.loss(accelerator, loss)
        return original_backward(accelerator, loss, *args, **kwargs)

    def training_end(logger, accelerator, model, *args, **kwargs):
        result = original_end(logger, accelerator, model, *args, **kwargs)
        observations.finish(accelerator, model, gather_object)
        return result

    runner.get_optimizer_class = optimizer_class
    Accelerator.backward = backward
    ModelLogger.on_training_end = training_end
    try:
        yield observations
        if observations.report["status"] != "passed":
            raise RuntimeError("Upstream ended without completed training observations")
    except BaseException as error:
        observations.fail(error)
        raise
    finally:
        runner.get_optimizer_class = original_optimizer
        Accelerator.backward = original_backward
        ModelLogger.on_training_end = original_end
        observations.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-train",required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--observations-dir")
    args,remaining = parser.parse_known_args()
    if remaining and remaining[0] == "--":
        remaining = remaining[1:]
    from accelerate.utils import set_seed
    # accelerate launch sets RANK but has not created AcceleratorState yet.
    # device_specific=True here would require an initialized Accelerator.
    set_seed(args.seed + int(os.environ.get("RANK", "0")),device_specific=False)
    sys.argv = [args.upstream_train]+remaining
    if args.observations_dir:
        with observe_training(args.observations_dir):
            runpy.run_path(args.upstream_train,run_name="__main__")
    else:
        runpy.run_path(args.upstream_train,run_name="__main__")


if __name__ == "__main__":
    main()
