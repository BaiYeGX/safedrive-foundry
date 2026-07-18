"""Real SimLingo DrivingModel load + bf16 forward (G3 neural V0).

Does NOT use Leaderboard/ScenarioRunner as tick master.
Requires PYTHONPATH to include simlingo-main and sdf venv with peft/transformers/hydra.
"""

from __future__ import annotations

import importlib.util
import io
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

SIMLINGO_CAMERA_XYZ = (-1.5, 0.0, 2.0)
SIMLINGO_CAMERA_FOV_DEG = 110.0
SIMLINGO_CAMERA_NATIVE_SIZE = (1024, 512)

# Repo roots
_REPO = Path(__file__).resolve().parents[3]
_SIMLINGO_CODE = _REPO / "simlingo-main"
_DEFAULT_CKPT = (
    _REPO / "models/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
)
_DEFAULT_HYDRA = _REPO / "models/simlingo/simlingo/.hydra/config.yaml"
_DEFAULT_INTERNVL = _REPO / "models/InternVL2-1B"

SPECIAL_TOKENS = [
    "<WAYPOINTS>",
    "<WAYPOINTS_DIFF>",
    "<ORG_WAYPOINTS_DIFF>",
    "<ORG_WAYPOINTS>",
    "<WAYPOINT_LAST>",
    "<ROUTE>",
    "<ROUTE_DIFF>",
    "<TARGET_POINT>",
]


def _ensure_simlingo_on_path() -> None:
    root = str(_SIMLINGO_CODE)
    if root not in sys.path:
        sys.path.insert(0, root)


def apply_transformers_internvl_compat_patch() -> None:
    """transformers>=5 expects all_tied_weights_keys; InternVL2 remote code may omit it."""
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return
    if getattr(PreTrainedModel, "_sdf_internvl_compat_patched", False):
        return

    orig = PreTrainedModel._move_missing_keys_from_meta_to_device

    def _patched(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return orig(self, *args, **kwargs)

    PreTrainedModel._move_missing_keys_from_meta_to_device = _patched  # type: ignore[method-assign]
    PreTrainedModel._sdf_internvl_compat_patched = True  # type: ignore[attr-defined]

    # Also ensure post_init children aggregation never crashes on missing attr.
    if hasattr(PreTrainedModel, "get_expanded_tied_weights_keys"):
        _orig_get = PreTrainedModel.get_expanded_tied_weights_keys

        def _get_safe(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            try:
                return _orig_get(self, *args, **kwargs)
            except Exception:
                return {}

        PreTrainedModel.get_expanded_tied_weights_keys = _get_safe  # type: ignore[method-assign]


@dataclass
class NeuralLoadReport:
    ok: bool
    source: str = "neural_simlingo"
    n_params: int = 0
    n_state_keys: int = 0
    missing_critical: list[str] = field(default_factory=list)
    missing_count: int = 0
    unexpected_count: int = 0
    matched_lora: int = 0
    matched_heads: int = 0
    load_s: float = 0.0
    construct_s: float = 0.0
    device: str = "cpu"
    error: str = ""
    head_key_match_ok: bool = False


@dataclass
class NeuralForwardResult:
    route_xy: np.ndarray  # (20, 2) ego frame meters
    speed_wps_xy: np.ndarray  # (10, 2) ego frame
    speed_mps: tuple[float, ...]  # (10,)
    latency_s: float
    source: str = "neural_simlingo"
    peak_vram_mb: float = 0.0


class SimLingoNeuralRuntime:
    """Holds a live DrivingModel for inference."""

    def __init__(
        self,
        *,
        ckpt_path: Path | str | None = None,
        hydra_config: Path | str | None = None,
        internvl_root: Path | str | None = None,
        device: str | None = None,
        predict_language: bool = False,
    ) -> None:
        self.ckpt_path = Path(ckpt_path or os.environ.get("SDF_SIMLINGO_CKPT", _DEFAULT_CKPT))
        self.hydra_config = Path(hydra_config or _DEFAULT_HYDRA)
        self.internvl_root = Path(internvl_root or os.environ.get("SDF_INTERNVL2_1B_ROOT", _DEFAULT_INTERNVL))
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self.predict_language = predict_language
        self.model: Any = None
        self.tokenizer: Any = None
        self.processor: Any = None
        self.cfg: Any = None
        self.num_image_token: int = 256
        self.load_report = NeuralLoadReport(ok=False)

    def load(self) -> NeuralLoadReport:
        t_all = time.perf_counter()
        try:
            _ensure_simlingo_on_path()
            apply_transformers_internvl_compat_patch()
            import torch
            from omegaconf import OmegaConf
            from hydra.utils import instantiate
            from transformers import AutoConfig, AutoProcessor

            if not self.ckpt_path.is_file():
                raise FileNotFoundError(self.ckpt_path)
            if not self.hydra_config.is_file():
                raise FileNotFoundError(self.hydra_config)
            if not self.internvl_root.is_dir():
                raise FileNotFoundError(self.internvl_root)

            cfg = OmegaConf.load(str(self.hydra_config))
            local = str(self.internvl_root)
            cfg.model.vision_model.variant = local
            cfg.model.language_model.variant = local
            if hasattr(cfg, "data_module") and hasattr(cfg.data_module, "use_global_img"):
                cfg.model.vision_model.use_global_img = cfg.data_module.use_global_img
            else:
                cfg.model.vision_model.use_global_img = False
            self.cfg = cfg

            t0 = time.perf_counter()
            processor = AutoProcessor.from_pretrained(local, trust_remote_code=True)
            if "tokenizer" in processor.__dict__:
                tokenizer = processor.tokenizer
            else:
                tokenizer = processor
            # transformers 5.x Qwen2Tokenizer may not expose additional_special_tokens_ids
            try:
                tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
            except Exception:
                for tok in SPECIAL_TOKENS:
                    try:
                        tokenizer.add_tokens(tok, special_tokens=True)
                    except Exception:
                        pass
            tok_ids = []
            for tok in SPECIAL_TOKENS:
                tid = tokenizer.convert_tokens_to_ids(tok)
                if tid is not None and tid != getattr(tokenizer, "unk_token_id", None):
                    tok_ids.append(int(tid))
            # SimLingo internvl2_model expects this attribute (legacy HF API)
            object.__setattr__(tokenizer, "additional_special_tokens_ids", tok_ids or [tokenizer.eos_token_id])
            try:
                object.__setattr__(tokenizer, "additional_special_tokens", list(SPECIAL_TOKENS))
            except Exception:
                pass
            tokenizer.padding_side = "left"
            self.processor = processor
            self.tokenizer = tokenizer

            tmp_config = AutoConfig.from_pretrained(local, trust_remote_code=True)
            image_size = tmp_config.force_image_size or tmp_config.vision_config.image_size
            patch_size = tmp_config.vision_config.patch_size
            self.num_image_token = int((image_size // patch_size) ** 2 * (tmp_config.downsample_ratio**2))

            default_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            model = instantiate(
                cfg.model,
                cfg_data_module=cfg.data_module,
                processor=processor,
                cache_dir=None,
                _recursive_=False,
            )
            torch.set_default_dtype(default_dtype)
            construct_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            sd = torch.load(str(self.ckpt_path), map_location="cpu", weights_only=False)
            if not isinstance(sd, dict):
                raise TypeError(f"unexpected ckpt type {type(sd)}")
            # unwrap common wrappers
            if "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]
            # strip module.
            if any(k.startswith("module.") for k in sd):
                sd = {k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()}

            missing, unexpected = model.load_state_dict(sd, strict=False)
            missing = list(missing)
            unexpected = list(unexpected)
            matched_lora = sum(1 for k in model.state_dict() if "lora_" in k and k in sd)
            head_keys = [k for k in model.state_dict() if "route_head" in k or "speed_wps_head" in k or "query_embeds" in k]
            matched_heads = sum(1 for k in head_keys if k in sd)
            critical_missing = [
                m
                for m in missing
                if any(x in m for x in ("route_head", "speed_wps_head", "query_embeds_wps", "query_embeds_speed"))
            ]
            head_ok = matched_heads >= max(1, len(head_keys) // 2) and len(critical_missing) == 0
            # Free CPU-side checkpoint tensors ASAP (otherwise RAM stays multi-GB while VRAM looks idle)
            n_sd_keys = len(sd)
            del sd
            import gc

            gc.collect()

            model.predict_language = bool(self.predict_language)
            model.eval()
            if self.device.startswith("cuda"):
                model = model.to(self.device)
                self._resident_device = "cuda"
            else:
                self._resident_device = "cpu"
            # Patch upstream bug: predict_language=False path passes (features,logits)
            # tuple into split_outputs_by_adaptor. Use explicit driving head path.
            _orig_fwd = model.forward

            def _forward_fixed(example, return_language=None, prompt_ids=None):  # type: ignore[no-untyped-def]
                try:
                    driving_input = example.driving_input
                except AttributeError:
                    driving_input = example
                model.speed_wps, model.route, model.language = None, None, []
                if bool(model.predict_language):
                    return _orig_fwd(example, return_language=return_language, prompt_ids=prompt_ids)
                adaptor_dict = model.adaptors(example, inference=True)
                features, logits = model.forward_model(driving_input, adaptor_dict)
                # features already adaptor slice; reverse-perm split by adaptor sizes
                outputs_by_adaptor = model.adaptors.split_outputs_by_adaptor(adaptor_dict, features)
                logits_by = model.adaptors.split_outputs_by_adaptor(adaptor_dict, logits)
                predictions = model.adaptors.driving.get_predictions(
                    outputs_by_adaptor["driving"], logits_by.get("driving")
                )
                for k, v in predictions.items():
                    if v is not None:
                        setattr(model, k, v)
                model.language = []
                return model.speed_wps, model.route, model.language

            model.forward = _forward_fixed  # type: ignore[method-assign]
            self.model = model
            n_params = sum(p.numel() for p in model.parameters())

            self.load_report = NeuralLoadReport(
                ok=head_ok and n_params > 1_000_000,
                source="neural_simlingo",
                n_params=int(n_params),
                n_state_keys=int(n_sd_keys),
                missing_critical=critical_missing[:20],
                missing_count=len(missing),
                unexpected_count=len(unexpected),
                matched_lora=matched_lora,
                matched_heads=matched_heads,
                load_s=time.perf_counter() - t1,
                construct_s=construct_s,
                device=getattr(self, "_resident_device", self.device),
                head_key_match_ok=head_ok,
            )
            if not self.load_report.ok:
                self.load_report.error = (
                    f"head_key_match_ok={head_ok} matched_heads={matched_heads}/{len(head_keys)} "
                    f"critical_missing={critical_missing[:5]}"
                )
        except Exception as exc:  # noqa: BLE001
            self.load_report = NeuralLoadReport(
                ok=False,
                load_s=time.perf_counter() - t_all,
                error=f"{type(exc).__name__}: {exc}",
                device=self.device,
            )
        return self.load_report

    def _load_conv_template(self):
        conv_path = self.internvl_root / "conversation.py"
        if not conv_path.is_file():
            raise FileNotFoundError(conv_path)
        spec = importlib.util.spec_from_file_location("internvl_conversation", str(conv_path))
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.get_conv_template

    def build_driving_input(
        self,
        rgb: np.ndarray,
        *,
        speed_mps: float,
        target_point_xy: tuple[float, float] = (10.0, 0.0),
        target_point2_xy: tuple[float, float] | None = None,
        command_text: str = "Command: follow the road.",
        camera_mount_xyz: tuple[float, float, float] = SIMLINGO_CAMERA_XYZ,
        jpeg_roundtrip: bool = True,
    ) -> Any:
        """rgb: HxWx3 uint8 RGB."""
        import torch
        from PIL import Image
        from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
        from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess

        if rgb is None:
            raise ValueError("neural backend requires front RGB image")
        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"rgb must be HxWx3, got {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Upstream evaluation JPEG-encodes every camera frame before the
        # training crop.  Matching it avoids a silent train/deploy mismatch.
        if jpeg_roundtrip:
            encoded = io.BytesIO()
            Image.fromarray(arr).save(encoded, format="JPEG", quality=95)
            encoded.seek(0)
            arr = np.asarray(Image.open(encoded).convert("RGB"), dtype=np.uint8)

        # crop bottom quarter (training)
        h, w = arr.shape[:2]
        cut = h - (h * 48) // 160  # ~4.8/16
        arr = arr[:cut]
        image = Image.fromarray(arr)
        use_global = bool(getattr(self.cfg.model.vision_model, "use_global_img", False))
        images = dynamic_preprocess(
            image, image_size=448, use_thumbnail=use_global, max_num=2
        )
        transform = build_transform(input_size=448)
        pixel_values = torch.stack([transform(im) for im in images])  # [N,3,448,448] CPU float
        # pad/truncate to 2 patches
        if pixel_values.shape[0] < 2:
            pad = pixel_values[-1:].repeat(2 - pixel_values.shape[0], 1, 1, 1)
            pixel_values = torch.cat([pixel_values, pad], dim=0)
        pixel_values = pixel_values[:2]
        # Host tensor only; _to_device moves camera_images to CUDA bfloat16 in one shot
        camera_images = pixel_values.unsqueeze(0).unsqueeze(0).contiguous()  # [1,1,2,3,448,448]

        speed = float(round(speed_mps, 1))
        tp2 = target_point2_xy or (target_point_xy[0] + 5.0, target_point_xy[1])
        prompt = f"Current speed: {speed} m/s. {command_text} Target waypoint: <TARGET_POINT><TARGET_POINT>. What should the ego do next?"

        get_conv_template = self._load_conv_template()
        if "<image>" not in prompt:
            question = "<image>\n" + prompt
        else:
            question = prompt
        template = get_conv_template("internlm2-chat")
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        system_prompt = template.system_template.replace("{system_message}", template.system_message) + template.sep
        query = query.replace(system_prompt, "")
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * 2 + IMG_END_TOKEN
        query = query.replace("<image>", image_tokens, 1)

        tok = self.tokenizer([query], padding=True, return_tensors="pt", add_special_tokens=False)
        phrase_ids = tok["input_ids"]
        phrase_valid = phrase_ids != self.tokenizer.pad_token_id
        tp = np.array([[target_point_xy[0], target_point_xy[1]], [tp2[0], tp2[1]]], dtype=np.float32)
        # placeholder map: token id -> coords
        tid = self.tokenizer.convert_tokens_to_ids("<TARGET_POINT>")
        placeholder_values = [{tid: tp}]

        ll = LanguageLabel(
            phrase_ids=phrase_ids,
            phrase_valid=phrase_valid,
            phrase_mask=phrase_valid.clone(),
            placeholder_values=placeholder_values,
            language_string=[query],
            loss_masking=torch.ones_like(phrase_ids, dtype=torch.bool),
        )

        # image sizes after crop; CARLA-like FOV 110 (inline — avoid team_code/cv2 import)
        ih, iw = arr.shape[0], arr.shape[1]
        fov = SIMLINGO_CAMERA_FOV_DEG
        focal = iw / (2.0 * math.tan(fov * math.pi / 360.0))
        K = torch.tensor(
            [[focal, 0.0, iw / 2.0], [0.0, focal, ih / 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 3, 3)
        E = torch.eye(4, dtype=torch.float32)
        E[0, 3] = float(camera_mount_xyz[0])
        E[1, 3] = float(camera_mount_xyz[1])
        E[2, 3] = float(camera_mount_xyz[2])
        E = E.view(1, 4, 4)

        di = DrivingInput(
            camera_images=camera_images,
            image_sizes=torch.tensor([[ih, iw]], dtype=torch.long),
            camera_intrinsics=K,
            camera_extrinsics=E,
            vehicle_speed=torch.tensor([[speed]], dtype=torch.float32),
            target_point=torch.tensor([[target_point_xy[0], target_point_xy[1]]], dtype=torch.float32),
            prompt=ll,
            prompt_inference=ll,
        )
        return di

    def _to_device(self, di: Any, device: str | None = None) -> Any:
        import torch
        from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel

        dev = device or self.device

        def move(x):
            if torch.is_tensor(x):
                # non_blocking helps when pinned; safe no-op on CPU tensors
                return x.to(dev, non_blocking=True)
            return x

        ll = di.prompt
        # placeholder coords: small numpy → torch on GPU for target tokens if used later
        ph_vals = ll.placeholder_values
        if ph_vals and dev.startswith("cuda"):
            ph_gpu = []
            for d in ph_vals:
                nd = {}
                for k, v in d.items():
                    if isinstance(v, np.ndarray):
                        nd[k] = torch.as_tensor(v, device=dev, dtype=torch.float32)
                    elif torch.is_tensor(v):
                        nd[k] = v.to(dev, non_blocking=True)
                    else:
                        nd[k] = v
                ph_gpu.append(nd)
            ph_vals = ph_gpu
        ll = LanguageLabel(
            phrase_ids=move(ll.phrase_ids),
            phrase_valid=move(ll.phrase_valid),
            phrase_mask=move(ll.phrase_mask),
            placeholder_values=ph_vals,
            language_string=ll.language_string,
            loss_masking=move(ll.loss_masking) if ll.loss_masking is not None else None,
        )
        # Vision activations live on GPU in bf16 (matches construct-time default_dtype)
        cam = di.camera_images.to(device=dev, dtype=torch.bfloat16, non_blocking=True)
        return DrivingInput(
            camera_images=cam,
            image_sizes=move(di.image_sizes),
            camera_intrinsics=move(di.camera_intrinsics).float(),
            camera_extrinsics=move(di.camera_extrinsics).float(),
            vehicle_speed=move(di.vehicle_speed).float(),
            target_point=move(di.target_point).float(),
            prompt=ll,
            prompt_inference=ll,
        )

    def release_gpu_for_carla(self) -> None:
        """Move model to CPU and free CUDA cache so Windows CARLA can use the 4080."""
        if self.model is None:
            return
        import torch

        try:
            self.model.cpu()
            self._resident_device = "cpu"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print("model moved to CPU, CUDA cache cleared (for CARLA)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print("release_gpu_for_carla warn", exc, flush=True)

    def keep_model_on_gpu(self) -> None:
        """Demo/live mode: pin DrivingModel on CUDA for the whole session (no GPU↔CPU bounce)."""
        if self.model is None:
            return
        import torch

        if not torch.cuda.is_available():
            self._resident_device = "cpu"
            print("keep_model_on_gpu: CUDA unavailable, staying on CPU", flush=True)
            return
        self.device = "cuda"
        self.model.to("cuda")
        self.model.eval()
        # Prefer inference kernels; leave weights on GPU permanently
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._resident_device = "cuda"
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
        reserv = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
        free_b, total_b = torch.cuda.mem_get_info()
        print(
            f"model kept resident on CUDA alloc={alloc:.0f}MB reserved={reserv:.0f}MB "
            f"free={free_b/1024**2:.0f}/{total_b/1024**2:.0f}MB",
            flush=True,
        )

    def forward_numpy(
        self,
        rgb: np.ndarray,
        *,
        speed_mps: float,
        target_point_xy: tuple[float, float] = (12.0, 0.0),
        target_point2_xy: tuple[float, float] | None = None,
        borrow_gpu: bool = True,
        keep_on_gpu: bool | None = None,
        camera_mount_xyz: tuple[float, float, float] = SIMLINGO_CAMERA_XYZ,
        jpeg_roundtrip: bool = True,
    ) -> NeuralForwardResult:
        """Neural forward.

        keep_on_gpu=True (demo): model stays on CUDA across calls — no GPU→CPU→GPU.
        borrow_gpu=True (legacy): one-shot to CUDA then bounce back to CPU for CARLA VRAM.
        """
        if self.model is None or not self.load_report.ok:
            raise RuntimeError(f"model not loaded: {self.load_report.error}")
        import torch
        from driving_vla.model.speed_convert import speed_wps_2d_to_mps

        resident = getattr(self, "_resident_device", None)
        if keep_on_gpu is None:
            keep_on_gpu = resident == "cuda"
        use_cuda = bool(torch.cuda.is_available()) and (keep_on_gpu or borrow_gpu or resident == "cuda")
        run_dev = "cuda" if use_cuda else "cpu"

        # Ensure weights on run device without bouncing away when keep_on_gpu.
        if use_cuda:
            if next(self.model.parameters()).device.type != "cuda":
                self.model.to("cuda")
            if keep_on_gpu:
                self._resident_device = "cuda"
                self.device = "cuda"
            torch.cuda.reset_peak_memory_stats()
        else:
            if next(self.model.parameters()).device.type != "cpu":
                self.model.to("cpu")
            self._resident_device = "cpu"

        try:
            di = self._to_device(
                self.build_driving_input(
                    rgb,
                    speed_mps=speed_mps,
                    target_point_xy=target_point_xy,
                    target_point2_xy=target_point2_xy,
                    camera_mount_xyz=camera_mount_xyz,
                    jpeg_roundtrip=jpeg_roundtrip,
                ),
                device=run_dev,
            )
            t0 = time.perf_counter()
            with torch.inference_mode():
                speed_wps, route, _lang = self.model(di)
            if use_cuda:
                torch.cuda.synchronize()
            latency = time.perf_counter() - t0
            if route is None or speed_wps is None:
                raise RuntimeError("model returned None route/speed_wps")
            # Pull only small trajectory tensors back to host (not the whole model)
            route_np = route[0].detach().float().cpu().numpy()
            speed_np = speed_wps[0].detach().float().cpu().numpy()
            if route_np.shape[0] != 20:
                if route_np.shape[0] > 20:
                    route_np = route_np[:20]
                else:
                    pad = np.repeat(route_np[-1:], 20 - route_np.shape[0], axis=0)
                    route_np = np.concatenate([route_np, pad], axis=0)
            speeds = speed_wps_2d_to_mps(speed_np, n_out=10)
            peak = 0.0
            if use_cuda:
                peak = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            return NeuralForwardResult(
                route_xy=route_np.astype(np.float64),
                speed_wps_xy=speed_np.astype(np.float64),
                speed_mps=speeds,
                latency_s=latency,
                source="neural_simlingo",
                peak_vram_mb=peak,
            )
        finally:
            # Only bounce to CPU in legacy borrow mode (not keep-on-GPU / resident).
            # Demo keep_on_gpu: never empty_cache here (would thrash allocator / look like "no VRAM").
            if use_cuda and not keep_on_gpu and resident != "cuda":
                try:
                    self.model.cpu()
                    torch.cuda.empty_cache()
                    self._resident_device = "cpu"
                except Exception:
                    pass


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False
