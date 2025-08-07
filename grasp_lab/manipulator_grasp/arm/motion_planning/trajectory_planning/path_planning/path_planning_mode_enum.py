from enum import unique
from ..\.interface import ModeEnum


@unique
class PathPlanningModeEnum(ModeEnum):
    JOINT = 'joint'
    CARTESIAN = 'cartesian'
