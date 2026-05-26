import torch

from omnisafe.common.normalizer import Normalizer

class NormalizerH(Normalizer):

    # def __init__(self, shape: tuple[int, ...], clip: float = 1e6) -> None:
    #     super().__init__(shape, clip)
    #     if shape == ():
    #         self._clip = clip * torch.tensor(1.0)
    #     else:
    #         self._clip = clip * torch.ones(*shape)

    def normalize_only(self, data: float) -> torch.Tensor:

        output = (data - self._mean) / self._std
        return torch.clamp(output, -self._clip, self._clip)

class MeanNormalizer(Normalizer):

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        data = data.to(self._mean.device)
        self._push(data)
        if self._count <= 1:
            return data
        output = data - self._mean
        return output
