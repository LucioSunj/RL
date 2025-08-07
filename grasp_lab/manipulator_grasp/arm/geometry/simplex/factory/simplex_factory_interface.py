from .simplex_parameter import SimplexParameter
from ..\.geometry.simplex.simplex import Simplex


class SimplexFactoryInterface:
    def create_product(self, simplex_parameter: SimplexParameter) -> Simplex:
        pass
