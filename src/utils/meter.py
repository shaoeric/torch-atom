from typing import List

__all__ = ['AverageMeter']

class AverageScalarMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            raise KeyError("name should be specified")
        name = kwargs['name']
        self.__sum = 0
        self.__count = 0
        self.name = name

    def update(self, val, batchsize=1):
        self.__sum += val * batchsize
        self.__count += batchsize

    def get_value(self):
        if self.__count == 0:
            return 0
        return self.__sum / self.__count

    def get_value_by_name(self, name):
        return self.get_value()

    def __repr__(self) -> str:
        return "{}: {}".format(self.name, round(self.get_value(), 4))


class AverageTupleMeter(object):
    def __init__(self, num_scalar: int, *args, **kwargs) -> None:
        assert isinstance(num_scalar, int)
        if 'names' not in kwargs:
            raise KeyError('names should be specified')
        names = kwargs['names']
        self.names = names
        self.__num_scalar = num_scalar
        self.__meter_list = [AverageScalarMeter(name=self.names[i]) for i in range(self.__num_scalar)]
        self.__count = 0
    
    def update(self, vals, batchsize=1):
        assert isinstance(vals, tuple), "vals is not tuple"
        assert len(vals) == self.__num_scalar
        for i in range(len(vals)):    
            self.__meter_list[i].update(vals[i], batchsize)
        self.__count += 1

    def get_value(self):
        return tuple([meter.get_value() for meter in self.__meter_list])

    def get_value_by_name(self, name):
        idx = self.names.index(name)
        return self.__meter_list[idx].get_value()

    def __repr__(self) -> str:
        string = ""
        values = self.get_value()
        for name, value in zip(self.names, values):
            string += "{}: {} ".format(name, round(value, 4))
        return string


class AverageMeter(object):
    def __init__(self, type: str="scalar", num_scalar: int=1, *args, **kwargs) -> None:
        assert type in ('scalar', 'tuple')
        assert isinstance(num_scalar, int)
        self.type = type
        self.num_scalar = num_scalar
        if self.type == 'scalar':
            self.meter = AverageScalarMeter(*args, **kwargs)
        else:
            self.meter = AverageTupleMeter(self.num_scalar, *args, **kwargs)
    
    def update(self, val, batchsize):
        self.meter.update(val, batchsize)

    def get_value(self):
        return self.meter.get_value()

    def get_value_by_name(self, name):
        return self.meter.get_value_by_name(name)

    def __repr__(self) -> str:
        return self.meter.__repr__()

if __name__ == '__main__':
    meter_scalar = AverageMeter(type='scalar')
    meter_scalar.update(3.0, 2)
    meter_scalar.update(1.0, 3)
    meter_scalar.update(2.0, 2)
    meter_scalar.update(4.0, 1)
    print(meter_scalar.get_value())

    meter_tuple = AverageMeter(type='tuple', num_scalar=2)
    meter_tuple.update((3.0,2), 2)
    meter_tuple.update((1.0,1), 3)
    meter_tuple.update((2.0,2), 2)
    meter_tuple.update((4.0,3), 1)
    print(meter_tuple.get_value())