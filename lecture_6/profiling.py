from util import *
from mlp import run_mlp

def profiling(is_composite: bool):
    if not is_composite:
        sleep_function = lambda: time.sleep(50 / 1000)
        sleep_profile = profile("sleep", sleep_function)
        print(sleep_profile)

        add_function = lambda a, b: a + b
        add_profile = profile("add", run_operation2(2048, add_function))
        print(add_profile)

        matmul_function = lambda a, b: a @ b
        matmul_profile = profile("matmul", run_operation2(2048, matmul_function))
        print(matmul_profile)

        matmul_profile_128 = profile("matmul(dim=128)", run_operation2(dim=128, operation=matmul_function))
        print(matmul_profile_128)

        cdist_function = lambda a, b: torch.cdist(a, b)
        cdist_profile = profile("cdist", run_operation2(dim=2048, operation=cdist_function))

    else:
        cdist_function = lambda a, b: torch.cdist(a, b)
        cdist_profile = profile("cdist", run_operation2(dim=2048, operation=cdist_function))
        print(cdist_profile)

        gelu_function = lambda a, b: torch.nn.functional.gelu(a + b)
        gelu_profile = profile("gelu", run_operation2(dim=2048, operation=gelu_function))
        print(gelu_profile)

        softmax_function = lambda a, b: torch.nn.functional.softmax(a + b, dim=-1)
        softmax_profile = profile("softmax", run_operation2(dim=2048, operation=softmax_function))
        print(softmax_profile)

        if torch.cuda.is_available():
            mlp_profile = profile("mlp", run_mlp(dim=2048, num_layers=64, batch_size=1024, num_steps=2))
            print(mlp_profile)

if __name__ == "__main__":
    profiling(is_composite=True)