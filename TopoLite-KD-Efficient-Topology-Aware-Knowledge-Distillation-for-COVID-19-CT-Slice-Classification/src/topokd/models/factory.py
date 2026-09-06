from __future__ import annotations

from torch import nn

from .msf import TopoLiteMSFKD
from .teacher import EfficientNetTeacher
from .topofm_slice import TopoFMSliceV1
from .topolite import TopoLiteModel, TopologyOnlyModel, VisualOnlyModel


def build_student(config: dict) -> nn.Module:
    name = config["model"]["name"]
    if name == "baseline_cnn_tda":
        return TopoLiteModel(config, baseline=True)
    if name in {"topolite_kd", "topolite_fkd_sam"}:
        return TopoLiteModel(config, baseline=False)
    if name == "visual_only":
        return VisualOnlyModel(config)
    if name == "tda_only":
        return TopologyOnlyModel(config)
    if name == "topolite_msf_kd":
        return TopoLiteMSFKD(config)
    if name == "topofm_slice_v1":
        return TopoFMSliceV1(config)
    raise ValueError(f"Unknown student model: {name}")


def build_teacher(config: dict, require_checkpoint: bool = True) -> EfficientNetTeacher | None:
    kd_weight = float(config["loss"].get("response_kd_weight", 0.0))
    feature_weight = float(config["loss"].get("feature_kd_weight", 0.0))
    if kd_weight <= 0 and feature_weight <= 0:
        return None
    teacher_cfg = config["model"]["teacher"]
    checkpoint = teacher_cfg.get("checkpoint")
    if require_checkpoint and not checkpoint:
        raise ValueError(
            "Knowledge distillation is enabled but model.teacher.checkpoint is null. Train the teacher first or point "
            "the config to the archived teacher_best.pt."
        )
    return EfficientNetTeacher(
        pretrained=bool(teacher_cfg.get("pretrained", True)),
        checkpoint=checkpoint,
        freeze=bool(teacher_cfg.get("freeze", True)),
        backbone_checkpoint=teacher_cfg.get("backbone_checkpoint"),
    )
