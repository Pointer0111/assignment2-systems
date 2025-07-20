from util import *
from mlp import run_mlp

def benchmarking():
    benchmark("sleep", lambda : time.sleep(50 / 1000))

    if torch.cuda.is_available():
        dims = (1024, 2048, 4096, 8192, 16384)  # @inspect dims
    else:
        dims = (1024, 2048)  # @inspect dims
    
    matmul_results = [] 
    for dim in dims:
        # @ inspect dim
        result = benchmark(f"matmul(dim={dim})", run_operation2(dim=dim, operation=lambda a, b: a @ b))
        matmul_results.append((dim, result))  # @inspect matmul_results
        
    dim = 256  # @inspect dim
    num_layers = 4  # @inspect num_layers 
    batch_size = 256  # @inspect batch_size
    num_steps = 2  # @inspect num_steps
    mlp_base = benchmark("run_mlp", run_mlp(dim=dim, num_layers=num_layers, batch_size=batch_size, num_steps=num_steps)) # @inspect mlp_base
    # Scale the number of steps.
    step_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x num_steps)", 
                         run_mlp(dim=dim, num_layers=num_layers, 
                                batch_size=batch_size, num_steps=scale * num_steps)) # @inspect result, @inspect scale, @inspect num_steps
        step_results.append((scale, result))  # @inspect step_results
    # Scale the number of layers.
    layer_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x num_layers)", 
                         run_mlp(dim=dim, num_layers=scale * num_layers, 
                                batch_size=batch_size, num_steps=num_steps)) # @inspect result, @inspect scale, @inspect num_layers, @inspect num_steps
        layer_results.append((scale, result))  # @inspect layer_results
    # Scale the batch size.
    batch_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x batch_size)", 
                         run_mlp(dim=dim, num_layers=num_layers, 
                                batch_size=scale * batch_size, num_steps=num_steps)) # @inspect result, @inspect scale, @inspect num_layers, @inspect num_steps
        batch_results.append((scale, result))  # @inspect batch_results
    # Scale the dimension.
    dim_results = []
    for scale in (2, 3, 4, 5):
        result = benchmark(f"run_mlp({scale}x dim)", 
                         run_mlp(dim=scale * dim, num_layers=num_layers, 
                                batch_size=batch_size, num_steps=num_steps)) # @inspect result, @inspect scale, @inspect num_layers, @inspect num_steps
        dim_results.append((scale, result))  # @inspect dim_results

    print("matmul_results:")
    print(matmul_results)
    print("--------------------------------")
    print("step_results:")
    print(step_results)
    print("--------------------------------")
    print("layer_results:")
    print(layer_results)
    print("--------------------------------")
    print("batch_results:")
    print(batch_results)
    print("--------------------------------")
    print("dim_results:")
    print(dim_results)


if __name__ == "__main__":
    benchmarking()