def visualize_jeffreys_prior(params, model, time_data, flux_data, error_data, param_type='quadratic'):
    import numpy as np
    import matplotlib.pyplot as plt
    from model_functions import compute_fisher_information, kipping_to_quad

    if param_type == 'quadratic':
        param_ranges = {
            'u1': np.linspace(-3, 3, 1000),
            'u2': np.linspace(-3, 3, 1000),
        }
        param_label = [r"$u_1$", r"$u_2$"]
    elif param_type == 'kipping':
        param_ranges = {
            'q1': np.linspace(0, 1, 1000),
            'q2': np.linspace(0, 1, 1000),
        }
        param_label = [r"$q_1$", r"$q_2$"]
    else:
        raise ValueError("Unknown parameter type. Use 'quadratic' or 'kipping'.")

    # Loop through each parameter and compute Jeffrey's prior
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for i, param in enumerate(param_ranges.keys()):
        prior_values = []
        for value in param_ranges[param]:
            if param_type == 'quadratic':
                params.u = [value, 0] if param == 'u1' else [0, value]
            elif param_type == 'kipping':
                theta = kipping_to_quad(value, 0) if param == 'q1' else kipping_to_quad(0, value)
                params.u = theta

            fisher_matrix = compute_fisher_information(params, model, time_data, flux_data, error_data)
            determinant = np.linalg.det(fisher_matrix)

            # Handle small determinants
            if determinant <= 1e-10:
                print(f"Small determinant for {param}={value}: {determinant}")
                prior_value = -1e6  # Large negative value for visualization
            else:
                prior_value = 0.5 * np.log(determinant)

            prior_values.append(prior_value)

        axs[i].plot(param_ranges[param], prior_values, label=f"Jeffrey's Prior for {param_label[i]}")
        axs[i].set_xlabel(param_label[i])
        axs[i].set_ylabel("Log Prior")
        axs[i].grid()
        axs[i].legend()

    plt.tight_layout()
    plt.show()
