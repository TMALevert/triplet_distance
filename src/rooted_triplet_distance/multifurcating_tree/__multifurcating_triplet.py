from dataclasses import dataclass

from ..__abstract import AbstractTriplet


class MultifurcatingTriplet(AbstractTriplet):
    def __init__(self, triplet: str | AbstractTriplet):
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
        if label not in self:
            return True
        elif self._string.endswith(f"|{label}"):
            return True
        elif self._string.startswith(f"{label}|"):
            return True
        elif f"|{label}|" in self._string:
            return True
        return False
