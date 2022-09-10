import sys
import numpy

backends_list = ["tensorflow", "pytorch", "jax", "numpy"]



def set_backend(backends_list: list = None, backend_instance: str = None):
    if not backend_instance == None:
        backend = backend_instance
    else:
        for backend_item in backends_list:
            if backend_item in sys.modules:
                backend = backend_item
                break
    return backend

print(set_backend(backends_list))
