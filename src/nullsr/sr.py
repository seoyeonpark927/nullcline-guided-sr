"""PySR settings shared by every symbolic-regression run."""

from pysr import PySRRegressor

SR_SETTINGS = dict(
    binary_operators=["+", "-", "*"],
    unary_operators=[],
    maxsize=20,
    model_selection="best",
    parallelism="serial",  # needed for exact reproducibility with a fixed seed
)


def make_regressor(seed, niterations, output_dir, run_id, variable_names=None, loss_function=None):
    """hall_of_fame.csv is written to  output_dir / run_id /."""
    extra = {}
    if variable_names is not None:
        extra["variable_names"] = variable_names
    if loss_function is not None:
        extra["loss_function"] = loss_function
    return PySRRegressor(
        niterations=niterations,
        random_state=seed,
        output_directory=str(output_dir),
        run_id=run_id,
        **SR_SETTINGS,
        **extra,
    )


def hybrid_loss(lam):
    """Julia loss:  L = L_data + lam * L_nullcline.

    Inputs are (u, v, marker). marker = 0 marks trajectory rows with target
    du/dt; marker = 1 marks nullcline rows, where the model must output 0.
    The marker column is zeroed before evaluation, so it cannot enter the
    equation itself.
    """
    return f"""
    function hybrid_loss(tree, dataset::Dataset{{T,L}}, options)::L where {{T,L}}
        X = dataset.X
        y = dataset.y
        X_expr = copy(X)
        X_expr[3, :] .= 0
        prediction, completed = eval_tree_array(tree, X_expr, options)
        if !completed
            return L(Inf)
        end
        data_mask = X[3, :] .== 0
        null_mask = X[3, :] .== 1
        data_loss = sum((prediction[data_mask] .- y[data_mask]).^2) / sum(data_mask)
        null_loss = sum(prediction[null_mask].^2) / sum(null_mask)
        return L(data_loss + {lam} * null_loss)
    end
    """
