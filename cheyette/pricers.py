from cheyette.models import CheyetteModel


class CheyettePricer(dict):
    """
    Aggregation pattern
    """
    def __init__(self, model: CheyetteModel, curve, process, product, valuation_time):
        dict.__init__(self)
        self['model'] = model
        self['curve'] = curve
        self['process'] = process
        self['product'] = product
        self['valuation_time'] = valuation_time

    def price(self):
        return self['model'].price(self['curve'], self['process'], self['product'], self['valuation_time'])

    def set(self, key, value):
        if key in self:
            self[key] = value
            return self
        raise Exception(f'Wrong attribute {key} of {self.__class__}')
