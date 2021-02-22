from __future__ import annotations
import torch
from torch import Tensor
import numpy as np
from pytorch_metrics.utils import is_binary
from typing import Union


class Metrics:
    def __init__(self, epsilon=1e-10):
        self.values = []
        self.accumulate_value = 0
        self.epsilon = epsilon

    def reset(self):
        self.values = []

    def __call__(self, y_pred: Tensor, y_true: Tensor, **kwargs) -> Metrics:
        pass

    @property
    def value(self) -> Union[int, float]:
        return self.values[-1]

    def mean(self, dim: int = 0) -> float:
        return torch.mean(torch.tensor(self.values), dim=dim)

    def var(self, dim: int = 0) -> float:
        return torch.var(torch.tensor(self.values), dim=dim)

    def std(self, dim: int = 0) -> float:
        return torch.std(torch.tensor(self.values), dim=dim)


class FuncContinueAverage(Metrics):
    def __init__(self, func, epsilon=1e-10):
        super().__init__(epsilon)
        self.func = func

    def __call__(self, *args, **kwargs) -> FuncContinueAverage:
        super().__call__(None, None)
        self.values.append(self.func(*args, **kwargs))

        return self


class ContinueAverage(Metrics):
    def __init__(self, epsilon=1e-10):
        super().__init__(epsilon)

    def __call__(self, value) -> ContinueAverage:
        super().__call__(None, None)
        self.values.append(value)

        return self


class BinaryAccuracy(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)

    def __call__(self, y_pred: Tensor, y_true: Tensor,
                 threshold: Union[float, list[float]]) -> BinaryAccuracy:
        super().__call__(y_pred, y_true)

        # Compute the accuracy
        with torch.set_grad_enabled(False):
            
            # Test if the y_pred given is not binary
            if not is_binary(y_pred):

                # If no threshold provided raise error
                if threshold is None:
                    raise ValueError("To calc binary accuracy you need to provide binary y_pred (or use threshold param)")

                y_pred = (y_pred > threshold).float()

            correct = (y_pred == y_true).float().sum()
            self.values.append(correct / np.prod(y_true.shape))

        return self


class CategoricalAccuracy(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> CategoricalAccuracy:
        super().__call__(y_pred, y_true)

        # Check if y_pred is of type long and contain class index
        if y_pred.dtype != torch.long:
            raise ValueError("To calc Categorical accurcy you need to provide long y_pred, containing only class index")

        with torch.set_grad_enabled(False):
            self.values.append(torch.mean((y_true == y_pred).float()))

        return self


class Ratio(Metrics):
    def __init__(self, epsilon=1e-10):
        Metrics.__init__(self, epsilon)

    def __call__(self, y1: Tensor, y2: Tensor) -> Ratio:
        super().__call__(y1, y2)

        results = zip(y1, y2)
        results_bool = [int(r[0] != r[1]) for r in results]
        self.values.append(sum(results_bool) / len(results_bool))

        return self


class Precision(Metrics):
    def __init__(self, dim = None, epsilon=1e-10):
        Metrics.__init__(self, epsilon)
        self.dim = dim

    def __call__(self, y_pred: Tensor, y_true: Tensor,
                 threshold: Union[float, list[float]] = None) -> Precision:
        super().__call__(y_pred, y_true)

        with torch.set_grad_enabled(False):

            # Test if the y_pred given is not binary
            if not is_binary(y_pred):

                # If no threshold provided raise error
                if threshold is None:
                    raise ValueError("To calc binary accuracy you need to provide binary y_pred (or use threshold param)")

                y_pred = (y_pred > threshold).float()

            dim = () if self.dim is None else self.dim
            y_true = y_true.float()
            y_pred = y_pred.float()

            true_positives = torch.sum(y_true * y_pred, dim=dim)
            predicted_positives = torch.sum(y_pred, dim=dim)

            if self.dim is None and predicted_positives == 0:
                self.values.append(torch.as_tensor(0.0))

            else:
                self.values.append(true_positives / (predicted_positives + self.epsilon))
                
        return self
        

class Recall(Metrics):
    def __init__(self, dim = None, epsilon=1e-10):
        Metrics.__init__(self, epsilon)
        self.dim = dim

    def __call__(self, y_pred: Tensor, y_true: Tensor,
                 threshold: Union[float, list[float]] = None) -> Recall:
        super().__call__(y_pred, y_true)

        with torch.set_grad_enabled(False):

            # Test if the y_pred given is not binary
            if not is_binary(y_pred):

                # If no threshold provided raise error
                if threshold is None:
                    raise ValueError("To calc binary accuracy you need to provide binary y_pred (or use threshold param)")

                y_pred = (y_pred > threshold).float()

            dim = () if self.dim is None else self.dim            
            y_true = y_true.float()
            y_pred = y_pred.float()
            
            true_positives = torch.sum(y_true * y_pred, dim=dim)
            possible_positives = torch.sum(y_true, dim=dim)
            
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

    def __call__(self, y_pred: Tensor, y_true: Tensor,
                 threshold: Union[float, list[float]] = None) -> FScore:
        super().__call__(y_pred, y_true)

        with torch.set_grad_enabled(False):
            dim = () if self.dim is None else self.dim
            
            self.precision = self.precision_func(y_pred, y_true, threshold)
            self.recall = self.recall_func(y_pred, y_true, threshold)

            if self.dim is None and (self.precision == 0 and self.recall == 0):
                self.values.append(torch.as_tensor(0.0))
            else:
                self.values.append(2 * ((self.precision_func.value * self.recall_func.value) / (self.precision_func.value + self.recall_func.value + self.epsilon)))
                
            return self
