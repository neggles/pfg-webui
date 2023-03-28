from pathlib import Path

import modules.scripts as scripts  # type: ignore
import gradio as gr
import numpy as np
import torch
from PIL import Image

from modules.processing import StableDiffusionProcessing  # type: ignore
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser  # type: ignore

from scripts.dbimutils import make_square, smart_24bit, smart_imread_pil, smart_resize
from scripts.download_model import TAGGER_DIR, ONNX_FILE, download

# extensions/pfg-webui直下のパス
EXTN_ROOT = Path(scripts.basedir())
MODELS_PATH = EXTN_ROOT.joinpath("models")

ONNX_PATH = MODELS_PATH.joinpath(ONNX_FILE)
TAGGER_PATH = MODELS_PATH.joinpath(TAGGER_DIR)

# Check for TensorFlow/Keras and import if available
HAS_TF = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model

    HAS_TF = True
except ImportError as e:
    print(f"Failed to import TensorFlow and Keras: {e}")

# Check for ONNX runtime and import it if available
HAS_ONNX = False
try:
    import onnxruntime

    HAS_ONNX = True
except ImportError as e:
    print(f"Failed to import ONNX Runtime: {e}")

if not (HAS_TF or HAS_ONNX):
    raise ImportError("Neither TensorFlow/Keras nor ONNX Runtime are available, PFG cannot be used.")


def is_model(file: Path):
    if not file.is_file():
        return False
    if file.name.startswith("."):
        return False
    if any(file.suffix.endswith(x) for x in ["onnx", "pt", "safetensors", ".pb"]):
        return True
    return False


class Script(scripts.Script):
    def __init__(self):
        # Get/update needed model files
        download(models_path=MODELS_PATH)

        # Save list of available models
        print(f"PFG loading models from {MODELS_PATH}")
        _ = self.get_model_list()
        print(f"Loaded models: {self.model_list}")
        self.callbacks_added = False

        # Initial values for processing
        self.use_onnx = False
        self.image: Image = None
        self.sub_image: Image = None
        self.pfg_scale: float = 1.0
        self.pfg_num_tokens: float = 10.0
        self.batch_size: int = 1

        # Initial values for model
        self.weight = None
        self.bias = None

    def get_model_list(self):
        model_list = [x.name for x in MODELS_PATH.iterdir() if is_model(x)]
        if not model_list:
            raise FileNotFoundError("No models found in the models directory.")
        else:
            self.model_list = model_list
        return self.model_list

    def title(self):
        return "Prompt-Free Generation"

    # どうやらこれをやるとタブに常に表示されるらしい。
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Prompt-Free Generation", open=False):
                with gr.Row(variant="compact"):
                    enabled = gr.Checkbox(value=False, label="Enabled")
                    use_onnx = gr.Checkbox(value=False, label="Use ONNX instead of TensorFlow")
                with gr.Row():
                    image = gr.Image(type="pil", label="Guide Image")
                with gr.Row(variant="compact"):
                    with gr.Column(scale=2, min_width=120):
                        pfg_model = gr.Dropdown(self.get_model_list(), label="Model", value=None)
                    with gr.Column(scale=3, min_width=240):
                        pfg_scale = gr.Slider(minimum=0, maximum=3, step=0.05, label="PFG scale", value=1.0)
                    with gr.Column(scale=3, min_width=240):
                        pfg_num_tokens = gr.Slider(
                            minimum=0, maximum=20, step=1.0, value=10.0, label="Tokens"
                        )
                with gr.Row():
                    sub_image = gr.Image(type="pil", label="sub image for latent couple")

        return enabled, image, pfg_scale, pfg_model, pfg_num_tokens, use_onnx, sub_image

    # wd-14-taggerの推論関数
    def infer(self, img: Image):
        img = smart_imread_pil(img)
        img = smart_24bit(img)
        img = make_square(img, 448)
        img = smart_resize(img, 448)
        img = img.astype(np.float32)
        if self.use_onnx:
            print("inferencing by onnx model.")
            probs = self.tagger.run(
                [self.tagger.get_outputs()[0].name], {self.tagger.get_inputs()[0].name: np.array([img])}
            )[0]
        else:
            print("inferencing by tensorflow model.")
            probs = self.tagger(np.array([img]), training=False).numpy()
        return torch.tensor(probs).squeeze(0).cpu()

    # CFGのdenoising step前に起動してくれるらしい。
    def denoiser_callback(self, params: CFGDenoiserParams):
        if self.enabled:
            # (batch_size*num_prompts, cond_tokens, dim)
            cond = params.text_cond
            couple = self.batch_size * 3 == cond.shape[0]
            # (batch_size*num_prompts, uncond_tokens, dim)
            uncond = params.text_uncond

            # (1, num_tokens, dim)
            pfg_cond = self.pfg_cond.to(cond.device, dtype=cond.dtype)
            if couple:
                pfg_cond_sub = self.pfg_cond_sub.to(cond.device, dtype=cond.dtype)
                pfg_cond_zero = torch.zeros_like(pfg_cond_sub)
                # (3, num_tokens, dim) - >  (batch size * 3, num_tokens, dim)
                pfg_cond = torch.cat([pfg_cond, pfg_cond_sub, pfg_cond_zero]).repeat(self.batch_size, 1, 1)
            else:
                pfg_cond = pfg_cond.repeat(cond.shape[0], 1, 1)
            # concatenate
            params.text_cond = torch.cat([cond, pfg_cond], dim=1)

            # copy zero
            pfg_uncond_zero = torch.zeros(uncond.shape[0], self.pfg_num_tokens, uncond.shape[2]).to(
                uncond.device, dtype=uncond.dtype
            )
            params.text_uncond = torch.cat([uncond, pfg_uncond_zero], dim=1)

            if params.sampling_step == 0:
                print(f"Apply pfg num_tokens:{self.pfg_num_tokens}(this message will be duplicated)")

    def load_tagger(self, use_onnx: bool):
        if use_onnx:
            if not HAS_ONNX:
                raise ImportError("ONNX Runtime is not available, must use TensorFlow/Keras.")
            self.tagger = onnxruntime.InferenceSession(
                ONNX_PATH,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        else:
            if not HAS_TF:
                raise ImportError("TensorFlow is not available, must use ONNX.")
            # なんもいみわかっとらんけどこれしないとVRAMくわれる。対応するバージョンもよくわからない
            physical_devices = tf.config.list_physical_devices("GPU")
            if len(physical_devices) > 0:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                    print(
                        f"{device} memory growth enabled: {tf.config.experimental.get_memory_growth(device)}"
                    )
            else:
                print(f"No GPU devices available, cannot use TensorFlow/Keras model")
                raise RuntimeError("No GPU devices available for TensorFlow/Keras model")

            self.tagger = load_model(TAGGER_PATH)
            # 最終層手前のプーリング層の出力を使う
            self.tagger = Model(self.tagger.layers[0].input, self.tagger.layers[-3].output)

    def process(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        image: Image,
        pfg_scale: float,
        pfg_model: str,
        pfg_num_tokens: int,
        use_onnx: bool,
        sub_image: Image = None,
    ):
        self.enabled = enabled
        if not self.enabled:
            return

        self.image = image
        self.sub_image = sub_image
        self.pfg_scale = pfg_scale
        self.pfg_num_tokens = pfg_num_tokens
        self.batch_size = p.batch_size

        pfg_weight = torch.load(MODELS_PATH.joinpath(pfg_model))
        self.weight = pfg_weight["pfg_linear.weight"].cpu()  # 大した計算じゃないのでcpuでいいでしょう
        self.bias = pfg_weight["pfg_linear.bias"].cpu()

        if self.tagger is None or use_onnx != self.use_onnx:
            self.load_tagger(use_onnx=use_onnx)
            self.use_onnx = use_onnx

        pfg_feature = self.infer(self.image) * self.pfg_scale
        # (768,) -> (dim * num_tokens, )
        self.pfg_cond = self.weight @ pfg_feature + self.bias

        # (dim * num_tokens, ) -> (1, num_tokens, dim)
        self.pfg_cond = self.pfg_cond.reshape(1, self.pfg_num_tokens, -1)

        if self.sub_image is not None:
            pfg_feature_sub = self.infer(self.sub_image) * self.pfg_scale
            self.pfg_cond_sub = self.weight @ pfg_feature_sub + self.bias
            self.pfg_cond_sub = self.pfg_cond_sub.reshape(1, self.pfg_num_tokens, -1)
        else:
            self.pfg_feature_sub = None

        if self.callbacks_added is not True:
            on_cfg_denoiser(self.denoiser_callback)
            self.callbacks_added = True

        return

    def postprocess(self, *args):
        return
