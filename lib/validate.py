import torch
from utils import calc_mean_IOU
import numpy as np
from tqdm import tqdm


def validate(val_loader, encoder, convrnn, decoder, device=None):
    if device is None:
        device = torch.device("cpu")
    NLL = torch.nn.NLLLoss()
    # initialize the hidden state:
    losses = []
    ious = []
    hidden = (
        torch.zeros((val_loader.batch_size, 128, 4, 4, 4)).to(device),
        torch.zeros((val_loader.batch_size, 128, 4, 4, 4)).to(device),
    )
    val_loader_pbar = tqdm(val_loader, desc="Validating", unit="batch")
    for batch, data in enumerate(val_loader_pbar):
        hidden[0].detach_()
        hidden[1].detach_()
        num_views = data["imgs"].shape[1]
        # loop over the image views and update the 3D-LSTM hidden state
        for v in range(num_views):
            cur_view = torch.squeeze(data["imgs"][:, v, :, :, :]).to(device)
            encoded_vec = encoder(cur_view)
            hidden = convrnn(encoded_vec, hidden)
        # finally decode the final hidden state and calculate the loss
        output = decoder(hidden[0])
        # torch.exp(output) will return the softmax scores before the log
        loss = NLL(output, data["label"].to(device)).item()
        iou = calc_mean_IOU(
            torch.exp(output).detach().cpu().numpy(), data["label"].numpy(), 0.4
        )[5]
        val_loader_pbar.set_postfix({"Loss": f"{loss:.4f}", "IOU": f"{iou:.4f}"})
        losses.append(loss)
        ious.append(iou)
    return np.mean(losses), np.mean(ious)
