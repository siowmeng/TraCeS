from abc import ABC, abstractmethod

from ubcrl.models.base import ActorH


class GaussianActorH(ActorH, ABC):
    """An abstract class for normal distribution actor.

    A NormalActor inherits from Actor and use Normal distribution to approximate the policy function.

    .. note::
        You can use this class to implement your own actor by inheriting it.
    """

    @property
    @abstractmethod
    def std(self) -> float:
        """Get the standard deviation of the normal distribution."""

    @std.setter
    @abstractmethod
    def std(self, std: float) -> None:
        """Set the standard deviation of the normal distribution."""
