from __future__ import annotations
from dataclasses import dataclass

from ..__abstract import AbstractTriplet


class MultifurcatingTriplet(AbstractTriplet):
    """
    A class representing a multifurcating triplet, which can be used to represent the relationships between three nodes.
    It can only represent fanned and resolved triplets, not general triplets.
    """

    def __init__(self, triplet: str | "MultifurcatingTriplet"):
        """
        Initializes a MultifurcatingTriplet instance.
        :param triplet: A string representation of the triplet or an instance of MultifurcatingTriplet.
        """
        if isinstance(triplet, MultifurcatingTriplet):
            super().__init__(triplet)
            self.parts = triplet.parts
            self.labels = triplet.labels
            self._type = triplet._type
        else:
            super().__init__(triplet)
            self.parts = {tuple(part.split(",")) if "," in part else part for part in self._string.split("|")}
            if len(self.parts) == 3:
                self._type = r"1|2|3"
            else:
                self._type = r"1|2,3"
            self.labels = set(self._string.replace("|", ",").split(","))

    def apart(self, label: str):
        """
        Checks if a label is in its own branch in the triplet.
        :param label: The label to check.
        :return: True if the label is in its own branch, False otherwise.
        """
        if label not in self:
            return True
        elif self._string.endswith(f"|{label}"):
            return True
        elif self._string.startswith(f"{label}|"):
            return True
        elif f"|{label}|" in self._string:
            return True
        return False
