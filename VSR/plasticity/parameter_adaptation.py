import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=1, alpha=1.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRA 只能包装 nn.Linear")
        rank = int(rank)
        if rank < 1:
            raise ValueError("LoRA rank 必须大于 0")
        alpha = float(alpha)
        if alpha <= 0:
            raise ValueError("LoRA alpha 必须大于 0")

        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs):
        base_output = self.base(inputs)
        delta = F.linear(F.linear(inputs, self.lora_a), self.lora_b)
        return base_output + self.scaling * delta


def inject_attention_lora(model, rank=1, alpha=1.0, projections=None):
    projections = tuple(
        projections or ("linear_q", "linear_k", "linear_v", "linear_out")
    )
    if not projections or len(set(projections)) != len(projections):
        raise ValueError("LoRA attention projections 必须非空且互不重复")
    allowed = {"linear_q", "linear_k", "linear_v", "linear_out"}
    if not set(projections).issubset(allowed):
        raise ValueError("LoRA 仅支持 attention 的 Q/K/V/O 线性层")

    replacements = []
    for name, module in list(model.named_modules()):
        if (
            not isinstance(module, nn.Linear)
            or not name.startswith("encoder.encoders.")
            or ".self_attn." not in name
        ):
            continue
        if name.rsplit(".", 1)[-1] not in projections:
            continue
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
        replacements.append(name)
    if not replacements:
        raise ValueError("模型 encoder 中没有找到可注入的 attention Q/K/V/O 线性层")
    return tuple(replacements)


def configure_batch_norm_adaptation(model):
    selected = []
    model.eval()
    model.requires_grad_(False)
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        module.train()
        module.track_running_stats = False
        module.running_mean = None
        module.running_var = None
        for parameter_name in ("weight", "bias"):
            parameter = getattr(module, parameter_name, None)
            if parameter is not None:
                parameter.requires_grad_(True)
                selected.append((f"{module_name}.{parameter_name}", parameter))
    if not selected:
        raise ValueError("模型中没有可更新的 BatchNorm2d affine 参数")
    return tuple(selected)


def configure_attention_lora_adaptation(
    model, rank=1, alpha=1.0, projections=None
):
    model.eval()
    model.requires_grad_(False)
    replacements = inject_attention_lora(
        model,
        rank=rank,
        alpha=alpha,
        projections=projections,
    )
    selected = named_lora_parameters(model)
    for _, parameter in selected:
        parameter.requires_grad_(True)
    if not selected:
        raise ValueError("模型中没有可更新的 LoRA 参数")
    return replacements, selected


def named_lora_parameters(model):
    selected = []
    for module_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        selected.extend(
            (
                (f"{module_name}.lora_a", module.lora_a),
                (f"{module_name}.lora_b", module.lora_b),
            )
        )
    return tuple(selected)


def validate_named_parameters(named_parameters):
    named_parameters = tuple(named_parameters)
    names = [name for name, _ in named_parameters]
    if len(names) != len(set(names)):
        raise ValueError("适应参数名称必须互不重复")
    if any(not isinstance(parameter, nn.Parameter) for _, parameter in named_parameters):
        raise TypeError("适应参数必须为 nn.Parameter")
    return named_parameters


def parameter_state(named_parameters):
    named_parameters = validate_named_parameters(named_parameters)
    return OrderedDict(
        (name, parameter.detach().cpu().clone())
        for name, parameter in named_parameters
    )


@torch.no_grad()
def load_parameter_state(named_parameters, state):
    parameters = OrderedDict(validate_named_parameters(named_parameters))
    if set(parameters) != set(state):
        missing = sorted(set(parameters) - set(state))
        unexpected = sorted(set(state) - set(parameters))
        raise ValueError(
            f"适应参数状态不匹配：missing={missing}, unexpected={unexpected}"
        )
    for name, parameter in parameters.items():
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"适应参数 {name} 的状态必须为 tensor")
        if value.shape != parameter.shape:
            raise ValueError(
                f"适应参数 {name} 形状不匹配：{tuple(value.shape)} != "
                f"{tuple(parameter.shape)}"
            )
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
