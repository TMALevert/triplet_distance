import re

from ..__abstract import AbstractTriplet

_re_patern_to_triplet_types = {
    re.compile(r"(.*)\|(.*),(.*)"): r"1|2,3",
    re.compile(r"(.*)\|(.*)\|(.*)"): r"1|2|3",
    re.compile(r"(.*)/(.*)\|(.*)"): r"1/2|3",
    re.compile(r"(.*)/(.*)/(.*)"): r"1/2/3",
    re.compile(r"(.*)/(.*)\\(.*)"): r"1/2\3",
    re.compile(r"(.*)\|(.*)\\(.*)"): r"1|2\3",
    re.compile(r"(.*),(.*)\|(.*)"): r"1,2|3",
    re.compile(r"(.*)\\(.*)\\(.*)"): r"1\2\3",
}

_triplet_types_to_re_pattern = {
    triplet_type: re_pattern for re_pattern, triplet_type in _re_patern_to_triplet_types.items()
}

_triplet_to_tuples = {
    r"1|2,3": lambda x, y, z: (None, (x, (None, tuple(sorted((y, z)))))),
    r"1|2|3": lambda x, y, z: (None, tuple(sorted((x, y, z)))),
    r"1/2|3": lambda x, y, z: (None, (z, (y, tuple({x})))),
    r"1/2/3": lambda x, y, z: (z, (y, tuple({x}))),
    r"1/2\3": lambda x, y, z: (y, tuple(sorted((x, z)))),
    r"1|2\3": lambda x, y, z: (None, (x, (y, tuple({z})))),
    r"1,2|3": lambda x, y, z: (None, (z, (None, tuple(sorted((x, y)))))),
    r"1\2\3": lambda x, y, z: (x, (y, tuple({z}))),
}


class GeneralTriplet(AbstractTriplet):
    def __init__(self, triplet: str | AbstractTriplet):
        if isinstance(triplet, GeneralTriplet):
            super().__init__(triplet)
            self._parts = triplet._parts
            self._labels = triplet._labels
            self._tree_relation = triplet._tree_relation
            self._branches = triplet._branches
            self._possible_root = triplet._possible_root
            self._descendants = triplet._descendants
            self._separations = triplet._separations
            self.__type = triplet.__type
        else:
            super().__init__(triplet)
            self._parts = {
                tuple(part.split(",")) if "," in part else part for part in re.split(r"[/\\|]", self._string)
            }
            self._labels = set()
            for label in self._parts:
                self._labels = self._labels.union({label} if isinstance(label, str) else set(label))
            self._tree_relation = self.__define_relations()
            self._branches = self.__get_branches()
            self._possible_root = self.__get_possible_root()
            self._descendants = self.__get_descendants()
            self._separations = self.__get_separations()

    @property
    def root(self) -> set[str]:
        return self._possible_root

    @property
    def branches(self) -> list[set]:
        return [branch.copy() for branch in self._branches]

    @property
    def type(self) -> str:
        return self.__type

    def __get_descendants(self) -> dict[str, set]:
        nodes = _triplet_types_to_re_pattern[self.__type].fullmatch(self._string).groups()
        if self.__type == "1|2,3":
            return {}
        elif self.__type == "1|2|3":
            return {}
        elif self.__type == "1/2|3":
            return {nodes[1]: {nodes[0]}}
        elif self.__type == "1/2/3":
            return {nodes[2]: {nodes[0], nodes[1]}, nodes[1]: {nodes[0]}}
        elif self.__type == r"1/2\3":
            return {nodes[1]: {nodes[0], nodes[2]}}
        elif self.__type == r"1|2\3":
            return {nodes[1]: {nodes[2]}}
        elif self.__type == r"1,2|3":
            return {}
        elif self.__type == r"1\2\3":
            return {nodes[0]: {nodes[1], nodes[2]}, nodes[1]: {nodes[2]}}

    def __get_separations(self) -> dict[str, set]:
        nodes = _triplet_types_to_re_pattern[self.__type].fullmatch(self._string).groups()
        if self.__type == "1|2,3":
            return {nodes[0]: {nodes[1], nodes[2]}, nodes[1]: {nodes[0], nodes[2]}, nodes[2]: {nodes[0], nodes[1]}}
        elif self.__type == "1|2|3":
            return {nodes[0]: {nodes[1], nodes[2]}, nodes[1]: {nodes[0], nodes[2]}, nodes[2]: {nodes[0], nodes[1]}}
        elif self.__type == "1/2|3":
            return {nodes[0]: {nodes[2]}, nodes[1]: {nodes[2]}, nodes[2]: {nodes[0], nodes[1]}}
        elif self.__type == "1/2/3":
            return {}
        elif self.__type == r"1/2\3":
            return {nodes[0]: {nodes[2]}, nodes[2]: {nodes[0]}}
        elif self.__type == r"1|2\3":
            return {nodes[0]: {nodes[1], nodes[2]}, nodes[1]: {nodes[0]}, nodes[2]: {nodes[0]}}
        elif self.__type == r"1,2|3":
            return {nodes[0]: {nodes[1], nodes[2]}, nodes[1]: {nodes[0], nodes[2]}, nodes[2]: {nodes[0], nodes[1]}}
        elif self.__type == r"1\2\3":
            return {}

    def __get_possible_root(self) -> set:
        if self.__type == "1|2,3":
            return set()
        elif self.__type == "1|2|3":
            return set()
        elif self.__type == "1/2|3":
            return set()
        elif self.__type == "1/2/3":
            return {self._tree_relation[0]}
        elif self.__type == r"1/2\3":
            return {self._tree_relation[0]}
        elif self.__type == r"1|2\3":
            return set()
        elif self.__type == r"1,2|3":
            return set()
        elif self.__type == r"1\2\3":
            return {self._tree_relation[0]}

    def __get_branches(self) -> list[set]:
        if self.__type == "1|2,3":
            return [set(self._tree_relation[1][1][1]), {self._tree_relation[1][0]}]
        elif self.__type == "1|2|3":
            return [{label} for label in self._labels]
        elif self.__type == "1/2|3":
            return [{self._tree_relation[1][1][0], self._tree_relation[1][1][1][0]}, {self._tree_relation[1][0]}]
        elif self.__type == "1/2/3":
            return [self._labels]
        elif self.__type == r"1/2\3":
            return [{self._tree_relation[1][0]}, {self._tree_relation[1][1]}]
        elif self.__type == r"1|2\3":
            return [{self._tree_relation[1][1][0], self._tree_relation[1][1][1][0]}, {self._tree_relation[1][0]}]
        elif self.__type == r"1,2|3":
            return [set(self._tree_relation[1][1][1]), {self._tree_relation[1][0]}]
        elif self.__type == r"1\2\3":
            return [self._labels]

    def __define_relations(self) -> dict:
        for template in _re_patern_to_triplet_types.keys():
            if template.fullmatch(self._string):
                self.__type = _re_patern_to_triplet_types[template]
                nodes = template.fullmatch(self._string).groups()
                relation_function = _triplet_to_tuples[self.__type]
                return relation_function(*nodes)
        else:
            raise ValueError(f"Invalid triplet: {self._string}")

    def __eq__(self, other):
        if not isinstance(other, (AbstractTriplet, str)):
            raise TypeError(
                f"unsupported operand type(s) for ==: '{self.__class__.__name__}' and '{other.__class__.__name__}'"
            )
        other = GeneralTriplet(other)
        return self._tree_relation == other._tree_relation
