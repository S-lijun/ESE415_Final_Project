import numpy as np
import matplotlib.pyplot as plt
from objective_function import (
    f_quadratic, grad_quadratic,
    f_rastrigin, grad_rastrigin,
    f_saddle, grad_saddle,
    f_rosenbrock, grad_rosenbrock
)
from generate_parameters import (
    generate_quadratic_params,
    generate_rastrigin_params,
    generate_saddle_params,
    generate_rosenbrock_params
)
from Gradient_Methods import (
    gradient_descent,
    exact_line_search,
    backtracking_line_search,
    accelerated_gradient
)
from scipy.optimize import minimize
from inspect import signature

# Fixed experiment config
x0 = np.array([1.0, 1.0])
step_size = 0.01
max_iters = 1000
tol = 1e-6
num_trials = 100

# Method execution logic
def run_method(method_fn, f, grad_f, x0, args):
    sig = signature(method_fn)
    if 'step_size' in sig.parameters:
        x_final, traj = method_fn(f, grad_f, x0, args=args, step_size=step_size, max_iters=max_iters, tol=tol)
    else:
        x_final, traj = method_fn(f, grad_f, x0, args=args, max_iters=max_iters, tol=tol)
    return f(x_final, *args), len(traj)

# Global optimizer for true min
def find_global_min(f, args):
    res = minimize(lambda x: f(x, *args), x0=np.zeros(2), method='BFGS')
    return res.fun

# Function evaluation per function type
#def evaluate_function_type(name, gen_params, f, grad_f, methods):
    results = {m.__name__: [] for m in methods}
    true_vals = []
    for _ in range(num_trials):
        params = gen_params()
        true_min = find_global_min(f, params)
        true_vals.append(true_min)
        for method in methods:
            try:
                val, iters = run_method(method, f, grad_f, x0, params)
            except Exception as e:
                val, iters = float('nan'), float('nan')
            results[method.__name__].append((val, iters))

    print(f"\n===== {name.upper()} =====")
    print(f"Average True Minimum: {np.mean(true_vals):.4f}")
    for method in methods:
        vals = [v for v, _ in results[method.__name__]]
        iters = [i for _, i in results[method.__name__]]
        print(f"{method.__name__}: Avg f(x) = {np.nanmean(vals):.4f}, Avg iters = {np.nanmean(iters):.1f}")


def evaluate_function_type(name, gen_params, f, grad_f, methods):
    results = {m.__name__: [] for m in methods}
    true_vals = []
    all_params = []

    # Collect all params and true min for reproducibility
    for _ in range(num_trials):
        params = gen_params()
        true_min = find_global_min(f, params)
        true_vals.append(true_min)
        all_params.append(params)

    print(f"\n===== {name.upper()} =====")
    print("True Minimums per trial:")
    print(np.round(true_vals, 4))


    all_iters = {}  # save all methods' iterations for global plotting
    for method in methods:
        method_vals = []
        method_iters = []
        for i in range(num_trials):
            try:
                val, iters = run_method(method, f, grad_f, x0, all_params[i])
            except:
                val, iters = float('nan'), float('nan')
            method_vals.append(val)
            method_iters.append(iters)

        mse = np.nanmean((np.array(method_vals) - np.array(true_vals)) ** 2)
        print(f"{method.__name__}: MSE = {mse:.4f}, Avg iters = {np.nanmean(method_iters):.1f}")
        
        all_iters[method.__name__] = method_iters

    # --- Plot histogram for all methods ---
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, max([max(iters) for iters in all_iters.values() if len(iters) > 0]), 60)

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']  # 定义颜色
    for idx, (method_name, iters) in enumerate(all_iters.items()):
        plt.hist(iters, bins=bins, alpha=0.5, label=method_name, edgecolor='black', color=colors[idx % len(colors)])
    
        mean_iter = np.nanmean(iters)
        # 画竖直的mean线
        plt.axvline(mean_iter, linestyle='--', color=colors[idx % len(colors)], label=f'{method_name} mean')
        # 在mean线的上方加数值
        plt.text(mean_iter + 5, plt.ylim()[1]*0.9 + 10, f'{mean_iter:.1f}', 
             rotation=0, color=colors[idx % len(colors)], verticalalignment='center', fontsize=12)

    plt.xlabel('Iterations')
    plt.ylabel('Frequency')
    plt.title(f'{name} - Iterations Distribution Across Methods')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
       



if __name__ == "__main__":
    methods_all = [gradient_descent, exact_line_search, backtracking_line_search, accelerated_gradient]

    #evaluate_function_type("Quadratic", generate_quadratic_params, f_quadratic, grad_quadratic, methods_all)
    #evaluate_function_type("Rastrigin", generate_rastrigin_params, f_rastrigin, grad_rastrigin, methods_all)
    evaluate_function_type("Saddle", generate_saddle_params, f_saddle, grad_saddle, methods_all)
    evaluate_function_type("Rosenbrock", generate_rosenbrock_params, f_rosenbrock, grad_rosenbrock, methods_all)
