import torch
import numpy as np
import functools


class Metrics:
    def __init__(self, epsilon=1e-10):
        self.values = []
        self.accumulate_value = 0
        self.epsilon = epsilon


    def reset(self):
        self.values = []

    def __call__(self, y_pred, y_true):
        pass

    @property
    def value(self):
        return self.values[-1]

    @functools.lru_cache
    @property
    def mean(self):
        nb_value = len(self.values)
        accumulate = sum(self.values)
        return accumulate / nb_value

    @functools.lru_cache
    @property
    def std(self):
        return np.std(self.values)


class FuncContinueAverage(Metrics):
    def __init__(self, func, epsilon=1e-10):
        super().__init__(epsilon)
        self.func = func

    def __call__(self, *args, **kwargs):
        super().__call__(None, None)
        self.values.append(self.func(*args, **kwargs))

        return self


class ContinueAverage(Metrics):
    def __init__(self, epsilon=1e-10):
        super().__init__(epsilon)

    def __call__(self, value):
        super().__call__(None, None)
        self.values.append(value)

        return self


class BinaryAccuracy(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)

    def __call__(self, y_pred, y_true):
        super().__call__(y_pred, y_true)

        with torch.set_grad_enabled(False):
            y_pred = (y_pred > 0.5).float()
            correct = (y_pred == y_true).float().sum()
            self.values.append(correct / np.prod(y_true.shape))

        return self


class CategoricalAccuracy(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)

    def __call__(self, y_pred, y_true):
        super().__call__(y_pred, y_true)

        with torch.set_grad_enabled(False):
            self.values.append(torch.mean((y_true == y_pred).float()))

        return self

    @property
    def accuracy(self):
        return self.value_


class Ratio(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)

    def __call__(self, y_pred, y_adv_pred):
        super().__call__(y_pred, y_adv_pred)

        results = zip(y_pred, y_adv_pred)
        results_bool = [int(r[0] != r[1]) for r in results]
        self.values.append(sum(results_bool) / len(results_bool))

        return self


class Precision(Metrics):
    def __init__(self, dim = None, epsilon=1e-10):
        Metrics.__init__(self, epsilon)
        self.dim = dim

    def __call__(self, y_pred, y_true):
        super().__call__(y_pred, y_true)

        with torch.set_grad_enabled(False):
            dim = () if self.dim is None else self.dim
            y_true = y_true.float()
            y_pred = y_pred.float()

            true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)), dim=dim)
            predicted_positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)), dim=dim)

            if self.dim is None and predicted_positives == 0:
                self.values.append(torch.as_tensor(0.0))
            else:
                self.values.append(true_positives / (predicted_positives + self.epsilon))
                
        return self
        

class Recall(Metrics):
    def __init__(self, dim = None, epsilon=1e-10):
        Metrics.__init__(self, epsilon)
        self.dim = dim

    def __call__(self, y_pred, y_true):
        super().__call__(y_pred, y_true)

        with torch.set_grad_enabled(False):
            dim = () if self.dim is None else self.dim            
            y_true = y_true.float()
            y_pred = y_pred.float()
            
            true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0.0, 1.0)), dim=dim)
            possible_positives = torch.sum(torch.clamp(y_true, 0.0, 1.0), dim=dim)
            
            if self.dim is None and possible_positives == 0:
                self.values.append(torch.as_tensor(0.0))
            else:
                self.values.append(true_positives / (possible_positives + self.epsilon))
                
            return self

        
class FScore(Metrics):
    def __init__(self, dim = None, epsilon=np.spacing(1)):
        Metrics.__init__(self, epsilon)
        self.dim = dim
        
        self.precision_func = Precision(dim, epsilon)
        self.recall_func = Recall(dim, epsilon)

    def __call__(self, y_pred, y_true):
        super().__call__(y_pred, y_true)

        with torch.set_grad_enabled(False):
            dim = () if self.dim is None else self.dim
            
            self.precision = self.precision_func(y_pred, y_true)
            self.recall = self.recall_func(y_pred, y_true)

            if self.dim is None and (self.precision == 0 and self.recall == 0):
                self.values.append(torch.as_tensor(0.0))
            else:
                self.values.append(2 * ((self.precision_func.value * self.recall_func.value) / (self.precision_func.value + self.recall_func.value + self.epsilon)))
                
            return self
