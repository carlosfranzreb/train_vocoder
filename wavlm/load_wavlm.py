import torch
from wavlm.WavLM import WavLM, WavLMConfig


def init_wavlm_large(pretrained=True, progress=True, device="cuda") -> WavLM:
    """
    Load the WavLM large checkpoint from the original paper.
    See https://github.com/microsoft/unilm/tree/master/wavlm for details.
    """
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt",
        map_location=device,
        progress=progress,
    )

    cfg = WavLMConfig(checkpoint["cfg"])
    device = torch.device(device)
    model = WavLM(cfg)
    if pretrained:
        model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    print(
        f"WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters."
    )
    return model
