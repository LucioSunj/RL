from ../../../geometry.simplex.factory.simplex_factory import SimplexFactory
from ../../../geometry.simplex.line_segment import LineSegment
from ../../../geometry.simplex.factory.simplex_parameter import SimplexParameter


class LineSegmentFactory(SimplexFactory):

    @property
    def key(self):
        return '2'

    def create_product(self, simplex_parameter: SimplexParameter):
        return LineSegment(simplex_parameter.parameter())


line_segment_factory = LineSegmentFactory()
line_segment_factory.register()
