import torch


class PromptableSAM:
    """
    Thin wrapper around the original SAM predictor API.

    The repository keeps this optional so the core benchmark does not depend
    on the external SAM package unless you explicitly run the promptable track.
    """

    def __init__(self, checkpoint, model_type="vit_b", device=None):
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "Install the optional SAM dependency first: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)

    def predict_with_box(self, image_rgb, box):
        self.predictor.set_image(image_rgb)
        masks, scores, _ = self.predictor.predict(
            box=box[None, :],
            multimask_output=True,
        )
        best = int(scores.argmax())
        return masks[best], float(scores[best])


class MedSAMPromptable(PromptableSAM):
    """
    MedSAM uses the same SAM-style predictor interface but with medical weights.
    """

    pass
