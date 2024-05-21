import numpy as np


def scale_gradients(model, gradient_logger):
    scaling_factors = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = gradient_logger.gradient_distributions.get(name, [])
            if len(grads) > 0:
                mean = np.mean(grads)
                std = np.std(grads)
                scaling_factors[name] = (mean, std)

    for name, param in model.named_parameters():
        if param.grad is not None and name in scaling_factors:
            mean, std = scaling_factors[name]
            if std > 0:
                param.grad.data = (param.grad.data - mean) / (std + 1e-6)