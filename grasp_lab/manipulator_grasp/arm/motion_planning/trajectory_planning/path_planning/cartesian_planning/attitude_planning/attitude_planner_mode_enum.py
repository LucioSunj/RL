from enum import unique
from ../../../../../interface import ModeEnum


@unique
class AttitudePlannerModeEnum(ModeEnum):
    ONE = 'one'
    TWO = 'two'
    THREE = 'three'
