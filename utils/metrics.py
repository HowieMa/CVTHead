import lpips
from pytorch_msssim import SSIM, MS_SSIM

import torch
from torch import nn
import lpips


class PSNR(object):
    def __call__(self, y_pred, y_true):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return PSNR, larger the better
        """
        mse = ((y_pred - y_true) ** 2).mean()
        return 10 * torch.log10(1 / mse)


class LPIPS(nn.Module):
    def __init__(self, device="cuda"):
        super(LPIPS, self).__init__()
        self.metric = lpips.LPIPS(net='alex', pretrained=True, )
        self.metric.eval()
        self.metric.to(device)

        for m in self.metric.modules():
            names = [name for name, _ in m.named_parameters()]
            for name in names:
                if hasattr(m, name):
                    data = getattr(m, name).data
                    delattr(m, name)
                    m.register_buffer(name, data, persistent=False)

            names = [name for name, _ in m.named_buffers()]
            for name in names:
                if hasattr(m, name):
                    data = getattr(m, name).data
                    delattr(m, name)
                    m.register_buffer(name, data, persistent=False)

    @torch.no_grad()
    def __call__(self, inputs, targets):
        return self.metric(inputs, targets, normalize=True).mean()

    def train(self, mode: bool = True):
        return self


