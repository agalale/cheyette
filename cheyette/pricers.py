from cheyette.models import CheyetteModel
from cheyette.curves import Curve
from cheyette.processes import CheyetteProcess
from cheyette.products import CheyetteProduct


class CheyettePricer:
    """
    Aggregation pattern
    """
    def __init__(self, model: CheyetteModel, curve: Curve, process: CheyetteProcess, product: CheyetteProduct,
                 valuation_time: float):
        self.model = model
        self.curve = curve
        self.process = process
        self.product = product
        self.valuation_time = valuation_time

    def price(self):
        self.product.initialize(self.curve, self.process)
        return self.model.price(self.curve, self.process, self.product, self.valuation_time)

    def set_curve(self, curve: Curve):
        self.curve = curve
        return self

    def __repr__(self):
        return str(vars(self))

