#!/usr/bin/env python3
"""
Learning rate schedulers for joint video-action training.
"""

from typing import Optional
import math

try:
    from diffusers.optimization import get_scheduler as hf_get_scheduler
except ImportError:
    hf_get_scheduler = None


class ClampedCosineScheduler:
    """Cosine decay that stays at the floor after the schedule horizon."""

    def __init__(
        self,
        optimizer,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        warmup_steps: int = 0,
        min_lr: Optional[float] = None,
    ):
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.explicit_min_lr = None if min_lr is None else float(min_lr)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.step_count = 0

        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps")
        if self.min_lr_ratio < 0:
            raise ValueError("min_lr_ratio must be non-negative")

    @property
    def min_lrs(self):
        if self.explicit_min_lr is not None:
            return [self.explicit_min_lr for _ in self.base_lrs]
        return [base_lr * self.min_lr_ratio for base_lr in self.base_lrs]

    def _lr_at(self, step: int, base_lr: float, min_lr: float) -> float:
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return base_lr * float(step) / float(self.warmup_steps)

        decay_steps = max(1, self.total_steps - self.warmup_steps)
        decay_step = min(max(0, step - self.warmup_steps), decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_steps))
        return min_lr + (base_lr - min_lr) * cosine

    def step(self):
        self.step_count += 1
        min_lrs = self.min_lrs
        for idx, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[idx]
            param_group["lr"] = self._lr_at(self.step_count, base_lr, min_lrs[idx])

    def get_last_lr(self):
        return [param_group["lr"] for param_group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "step_count": self.step_count,
            "base_lrs": self.base_lrs,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "explicit_min_lr": self.explicit_min_lr,
        }

    def load_state_dict(self, state_dict):
        self.step_count = int(state_dict["step_count"])
        self.base_lrs = [float(lr) for lr in state_dict.get("base_lrs", self.base_lrs)]
        self.total_steps = int(state_dict.get("total_steps", self.total_steps))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.min_lr_ratio = float(state_dict.get("min_lr_ratio", self.min_lr_ratio))
        self.explicit_min_lr = state_dict.get("explicit_min_lr", self.explicit_min_lr)


class CosineRestartScheduler:
    """Cosine decay with fixed-length warm restarts."""

    def __init__(
        self,
        optimizer,
        cycle_steps: int,
        min_lr_ratio: float = 0.0,
        warmup_steps: int = 0,
        min_lr: Optional[float] = None,
        restart_decay: float = 1.0,
    ):
        self.optimizer = optimizer
        self.cycle_steps = int(cycle_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.explicit_min_lr = None if min_lr is None else float(min_lr)
        self.restart_decay = float(restart_decay)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.step_count = 0

        if self.cycle_steps <= 0:
            raise ValueError("cycle_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.warmup_steps >= self.cycle_steps:
            raise ValueError("warmup_steps must be smaller than cycle_steps")
        if self.min_lr_ratio < 0:
            raise ValueError("min_lr_ratio must be non-negative")
        if self.restart_decay <= 0:
            raise ValueError("restart_decay must be positive")

    def _cycle_index_and_step(self, step: int) -> tuple[int, int]:
        if step <= self.cycle_steps:
            return 0, step
        offset = step - self.cycle_steps - 1
        return 1 + offset // self.cycle_steps, 1 + offset % self.cycle_steps

    def _min_lr_for_peak(self, peak_lr: float) -> float:
        if self.explicit_min_lr is not None:
            return min(self.explicit_min_lr, peak_lr)
        return peak_lr * self.min_lr_ratio

    def _lr_at(self, step: int, base_lr: float) -> float:
        if step <= 0:
            return base_lr

        cycle_index, cycle_step = self._cycle_index_and_step(step)
        peak_lr = base_lr * (self.restart_decay ** cycle_index)

        if cycle_index == 0 and self.warmup_steps > 0 and step <= self.warmup_steps:
            return peak_lr * float(step) / float(self.warmup_steps)

        min_lr = self._min_lr_for_peak(peak_lr)
        if cycle_index == 0:
            decay_steps = max(1, self.cycle_steps - self.warmup_steps)
            decay_step = min(max(0, cycle_step - self.warmup_steps), decay_steps)
        else:
            decay_steps = max(1, self.cycle_steps - 1)
            decay_step = min(max(0, cycle_step - 1), decay_steps)

        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_step / decay_steps))
        return min_lr + (peak_lr - min_lr) * cosine

    def step(self):
        self.step_count += 1
        for idx, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[idx]
            param_group["lr"] = self._lr_at(self.step_count, base_lr)

    def get_last_lr(self):
        return [param_group["lr"] for param_group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "step_count": self.step_count,
            "base_lrs": self.base_lrs,
            "cycle_steps": self.cycle_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "explicit_min_lr": self.explicit_min_lr,
            "restart_decay": self.restart_decay,
        }

    def load_state_dict(self, state_dict):
        self.step_count = int(state_dict["step_count"])
        self.base_lrs = [float(lr) for lr in state_dict.get("base_lrs", self.base_lrs)]
        self.cycle_steps = int(state_dict.get("cycle_steps", self.cycle_steps))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self.min_lr_ratio = float(state_dict.get("min_lr_ratio", self.min_lr_ratio))
        self.explicit_min_lr = state_dict.get("explicit_min_lr", self.explicit_min_lr)
        self.restart_decay = float(state_dict.get("restart_decay", self.restart_decay))


class LambdaLinearScheduler:
    """
    Linear learning rate scheduler with warmup.
    
    This scheduler implements:
    1. Warmup phase: Linear increase from f_start to f_max over warm_up_steps
    2. Decay phase: Linear decrease from f_max to f_min over remaining steps
    
    Args:
        optimizer: PyTorch optimizer
        warm_up_steps: Number of warmup steps
        cycle_length: Total number of steps in the cycle
        f_max: Maximum learning rate multiplier (reached after warmup)
        f_min: Minimum learning rate multiplier (at end of cycle)
        f_start: Starting learning rate multiplier (at step 0)
    """
    
    def __init__(self, optimizer, warm_up_steps: int, cycle_length: int, 
                 f_max: float = 1.0, f_min: float = 0.1, f_start: float = 1e-6):
        self.optimizer = optimizer
        self.warm_up_steps = warm_up_steps
        self.cycle_length = cycle_length
        self.f_max = f_max
        self.f_min = f_min
        self.f_start = f_start
        # Support per-parameter-group base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        # Backward-compat single base lr (first group's lr)
        self.base_lr = self.base_lrs[0]
        self.step_count = 0
        
        # Validate parameters
        assert warm_up_steps >= 0, "warm_up_steps must be non-negative"
        assert cycle_length > warm_up_steps, "cycle_length must be greater than warm_up_steps"
        assert f_max >= f_min, "f_max must be >= f_min"
        assert f_start >= 0, "f_start must be non-negative"
        
    def step(self):
        """Update learning rates for next step (supports multiple param groups)"""
        self.step_count += 1
        lr_multiplier = self.get_lr_multiplier(self.step_count)

        # Apply per-group base lr scaling
        for idx, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[idx] if idx < len(self.base_lrs) else self.base_lr
            param_group['lr'] = base_lr * lr_multiplier
    
    def get_lr_multiplier(self, step: int) -> float:
        """Calculate learning rate multiplier for given step"""
        if step <= 0:
            return self.f_start
        elif step <= self.warm_up_steps:
            # Warmup: linear increase from f_start to f_max
            return self.f_start + (self.f_max - self.f_start) * step / self.warm_up_steps
        elif step < self.cycle_length:
            # Main phase: linear decay from f_max to f_min
            remaining_steps = self.cycle_length - step
            decay_steps = self.cycle_length - self.warm_up_steps
            return self.f_min + (self.f_max - self.f_min) * remaining_steps / decay_steps
        else:
            # After cycle ends, maintain minimum learning rate
            return self.f_min
    
    def get_last_lr(self):
        """Return current learning rates for all parameter groups"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Return scheduler state for checkpointing"""
        return {
            'step_count': self.step_count,
            'base_lr': self.base_lr,
            'base_lrs': self.base_lrs,
            'warm_up_steps': self.warm_up_steps,
            'cycle_length': self.cycle_length,
            'f_max': self.f_max,
            'f_min': self.f_min,
            'f_start': self.f_start,
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint"""
        self.step_count = state_dict['step_count']
        self.base_lr = state_dict.get('base_lr', self.base_lr)
        self.base_lrs = state_dict.get('base_lrs', self.base_lrs)
        self.warm_up_steps = state_dict['warm_up_steps']
        self.cycle_length = state_dict['cycle_length']
        self.f_max = state_dict['f_max']
        self.f_min = state_dict['f_min']
        self.f_start = state_dict['f_start']


def create_scheduler(optimizer, config):
    """
    Create learning rate scheduler based on config.
    
    Args:
        optimizer: PyTorch optimizer
        config: Training configuration object
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = getattr(config.training, 'scheduler_type', 'cosine')
    
    if scheduler_type == "linear":
        return LambdaLinearScheduler(
            optimizer,
            warm_up_steps=config.training.warmup_steps,
            cycle_length=config.training.cycle_length,
            f_max=config.training.f_max,
            f_min=config.training.f_min,
            f_start=getattr(config.training, 'f_start', 1e-6)
        )
    elif scheduler_type == "cosine":
        total_steps = int(getattr(config.training, 'lr_schedule_steps', getattr(config.training, 'max_steps', 10000)))
        min_lr = getattr(config.training, 'min_lr', None)
        min_lr = None if min_lr is None else float(min_lr)
        return ClampedCosineScheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=int(getattr(config.training, 'warmup_steps', 0)),
            min_lr_ratio=float(getattr(config.training, 'min_lr_ratio', 0.1)),
            min_lr=min_lr,
        )
    elif scheduler_type in {"cosine_restart", "cosine_warm_restart", "cosine_restarts"}:
        cycle_steps = int(getattr(
            config.training,
            'lr_cycle_steps',
            getattr(config.training, 'lr_schedule_steps', getattr(config.training, 'max_steps', 10000)),
        ))
        min_lr = getattr(config.training, 'min_lr', None)
        min_lr = None if min_lr is None else float(min_lr)
        return CosineRestartScheduler(
            optimizer,
            cycle_steps=cycle_steps,
            warmup_steps=int(getattr(config.training, 'warmup_steps', 0)),
            min_lr_ratio=float(getattr(config.training, 'min_lr_ratio', 0.1)),
            min_lr=min_lr,
            restart_decay=float(getattr(config.training, 'lr_restart_decay', 1.0)),
        )
    elif scheduler_type == "diffusers_cosine":
        # 使用 diffusers.optimization.get_scheduler 的余弦调度
        if hf_get_scheduler is None:
            raise ImportError("diffusers is required when scheduler_type='diffusers_cosine'")
        total_steps = int(getattr(config.training, 'lr_schedule_steps', getattr(config.training, 'max_steps', 10000)))
        warmup_steps = int(getattr(config.training, 'warmup_steps', 0))
        base = float(config.training.learning_rate)
        min_lr = float(getattr(config.training, 'min_lr', base * getattr(config.training, 'min_lr_ratio', 0.0)))

        inner = hf_get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        class _MinLRWrapper:
            def __init__(self, inner_sched, opt, min_lr_value: float):
                self.inner = inner_sched
                self.optimizer = opt
                self.min_lr = float(min_lr_value)

            def step(self):
                self.inner.step()
                if self.min_lr > 0.0:
                    for g in self.optimizer.param_groups:
                        if g.get('lr', 0.0) < self.min_lr:
                            g['lr'] = self.min_lr

            def get_last_lr(self):
                try:
                    return self.inner.get_last_lr()
                except Exception:
                    return [g['lr'] for g in self.optimizer.param_groups]

            # 兼容 accelerate.save_state / load_state
            def state_dict(self):
                sd = getattr(self.inner, 'state_dict', None)
                return sd() if callable(sd) else {}

            def load_state_dict(self, state):
                ld = getattr(self.inner, 'load_state_dict', None)
                if callable(ld):
                    ld(state)

        return _MinLRWrapper(inner, optimizer, min_lr)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
