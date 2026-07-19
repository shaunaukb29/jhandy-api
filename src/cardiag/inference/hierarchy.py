from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    name: str
    parent: Node | None = None
    children: list[Node] = field(default_factory=list)

    def add_child(self, child: Node):
        child.parent = self
        self.children.append(child)

    def __hash__(self):
        return hash(self.name)

# Define hierarchy
ROOT = Node("Vehicle")

ENGINE = Node("Engine", parent=ROOT)
ROOT.add_child(ENGINE)

BOTTOM_END = Node("Bottom End", parent=ENGINE)
ENGINE.add_child(BOTTOM_END)

VALVETRAIN = Node("Valvetrain", parent=ENGINE)
ENGINE.add_child(VALVETRAIN)

TIMING = Node("Timing", parent=ENGINE)
ENGINE.add_child(TIMING)

COOLING = Node("Cooling", parent=ROOT)
ROOT.add_child(COOLING)

DRIVETRAIN = Node("Drivetrain", parent=ROOT)
ROOT.add_child(DRIVETRAIN)

TRANSMISSION = Node("Transmission", parent=ROOT)
ROOT.add_child(TRANSMISSION)

BRAKES = Node("Brakes", parent=ROOT)
ROOT.add_child(BRAKES)

SUSPENSION = Node("Suspension", parent=ROOT)
ROOT.add_child(SUSPENSION)

STEERING = Node("Steering", parent=ROOT)
ROOT.add_child(STEERING)

FUEL_IGNITION = Node("Fuel & Ignition", parent=ENGINE)
ENGINE.add_child(FUEL_IGNITION)

EXHAUST = Node("Exhaust", parent=ENGINE)
ENGINE.add_child(EXHAUST)

BELT = Node("Belt", parent=ENGINE)
ENGINE.add_child(BELT)

LOW_OIL = Node("Low Oil", parent=ENGINE)
ENGINE.add_child(LOW_OIL)

ACCESSORIES = Node("Accessories", parent=ENGINE)
ENGINE.add_child(ACCESSORIES)

def build_hierarchy():
    nodes_by_name = {}
    def _traverse(n):
        nodes_by_name[n.name.lower()] = n
        for c in n.children:
            _traverse(c)
    _traverse(ROOT)
    return nodes_by_name

HIERARCHY_MAP = build_hierarchy()

def get_ancestors(node_name: str) -> set[str]:
    node = HIERARCHY_MAP.get(node_name.lower())
    if not node:
        return set()
    ancestors = set()
    curr = node.parent
    while curr:
        ancestors.add(curr.name.lower())
        curr = curr.parent
    return ancestors

def get_descendants(node_name: str) -> set[str]:
    node = HIERARCHY_MAP.get(node_name.lower())
    if not node:
        return set()
    desc = set()
    def _traverse(n):
        desc.add(n.name.lower())
        for c in n.children:
            _traverse(c)
    _traverse(node)
    desc.remove(node.name.lower())
    return desc
