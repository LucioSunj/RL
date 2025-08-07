from ....factory.simplex_factory import SimplexFactory
from ....line_segment import LineSegment
from ....factory.simplex_parameter import SimplexParameter


class LineSegmentFactory(SimplexFactory):

    @property
    def key(self):
        return '2'

    def create_product(self, simplex_parameter: SimplexParameter):
        return LineSegment(simplex_parameter.parameter())


line_segment_factory = LineSegmentFactory()
line_segment_factory.register()
