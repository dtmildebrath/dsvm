""" Implementation of an "exact" linear SVM (called DSVM).

    Finds a maximum-margin linear SVM with provably minimal
    classification error on the training data. 

    Also performs an (optional) sensitivity analysis, to explore the range of
    possible values across all separating hyperplanes with minimal
    classification error.

    Main function is dsvmf(...)

    Other functions could also be called independently. This code uses Gurobi
    to solve mixed-integer programs, and is thus not suitable for medium or
    large datasets. It is intended for small scaled experimentation.
"""
import numpy as np
import gurobipy as grb


def dsvmf(X, y, eps=1e-5, sensitivity_analysis=False, verbose=False):
    """ Compute a minimum-error, maximum-margin (in that order) separating
    hyperplane for the data. Optionally perform sensitivity analysis on the
    resulting hyperplane.

    Sensitivity analysis consists of finding, for each feature, the largest and
    smallest possible values of w[i] while still maintaining optimal
    classification error.

    Args:
        X (nxm ndarray):
            Data matrix whose rows correspond to data points, and columns
            correspond to features.
        y (nx1 ndarray):
            Vectory of binary class labels in {0,1}.
        eps (float, optional):
            Margin to use in place of strict inequalities. Default is 1e-5.
        sensitivity_analysis (bool, optional):
            If True, performs sensitivity analysis. Default is False.
        verbose (bool, optional):
            If True, displays Gurobi solver output. Default is False.

    Returns:
        1xm ndarray:
            Minimum-error, maximum-margin separating hyperplane (normalized).
        float:
            Corresponding offset of the hyperplane.
        int:
            Minimum possible classification error.
        float:
            Width of the resulting margin.
        list of tuple of float (is sensitivity_analysis=True):
            List of pairs (w_min, w_max), one for each feature, containing the
            smallest and largest possible values for the separating hyperplane
            coefficient for that feature.
    """
    _, _, err = min_error(X, y, verbose=verbose)
    w_opt, b_opt, width = max_margin(X, y, err, return_width=True, verbose=verbose)
    if sensitivity_analysis:
        sens = sensitivity(X, y, err, eps=eps)
        return w_opt, b_opt, err, width, sens
    return w_opt, b_opt, err, width


def min_error(X, y, verbose=False):
    """ Compute a minimum-classification-error separating hyperplane.
        
    Args:
        X (nxm ndarray):
            Data matrix whose rows correspond to data points, and columns
            correspond to features.
        y (nx1 ndarray):
            Vectory of binary class labels in {0,1}.
        verbose (bool, optional):
            If True, display Gurobi solver output. Default is False.

    Returns:
        mx1 ndarray:
            Normalized normal vector to a minimum-error separating hyperplane.
        float:
            Scalar offset of the hyperplane.
        int:
            Provably minimal classification error for any separating
            hyperplane.

    Raises:
        RuntimeError:
            If the MIP is not solved to optimality.
    """
    model = grb.Model("heuristic")
    C1 = np.where(y == 0)[0]
    C2 = np.where(y == 1)[0]

    n = X.shape[0]
    p = X.shape[1]
    z = model.addVars(n, vtype=grb.GRB.BINARY)
    b = model.addVar(lb=-grb.GRB.INFINITY)
    w = model.addVars(p, lb=-grb.GRB.INFINITY)

    model.setObjective(grb.quicksum(z))

    for i in C1:
        model.addConstr(
            (z[i] == 0) >> (b >= 1 + grb.quicksum(X[i, j] * w[j] for j in range(p)))
        )

    for i in C2:
        model.addConstr(
            (z[i] == 0) >> (b + 1 <= grb.quicksum(X[i, j] * w[j] for j in range(p)))
        )

    model.setParam("OutputFlag", verbose)

    model.optimize()
    if model.status != grb.GRB.OPTIMAL:
        raise RuntimeError("Heuristic MIP not solved to optimality")

    w0 = np.array([w[i].X for i in range(len(w))])
    b0 = b.X
    scale = np.linalg.norm(w0, ord=2)
    return w0 / scale, b0 / scale, model.objval


def sensitivity(X, y, err, eps=1e-5, verbose=False):
    """ Find the largest and smallest possible values of the separating
    hyperplane coefficients among all normalized separating hyperplanes with
    classification error <= err.

    Args:
        X (nxm ndarray):
            Data matrix whose rows correspond to data points, and columns
            correspond to features.
        y (nx1 ndarray):
            Vectory of binary class labels in {0,1}.
        err (int):
            Maximum allowable classification error.
        eps (float, optional):
            Margin to use in place of strict inequalities. Default is 1e-5.
        verbose (bool, optional):
            If True, displays Gurobi solver output. Default is False.

    Returns:
        list of tuple of float:
            List of pairs (w_min, w_max), one for each feature, containing the
            smallest and largest possible values for the separating hyperplane
            coefficient for that feature.

    Raises:
        RuntimeError:
            If the MIP is not solved to optimality.

    """
    return [
        sensitivity_one_index(X, y, index, err, eps=eps, verbose=verbose)
        for index in range(X.shape[1]) 
    ]


def sensitivity_one_index(X, y, index, err, eps=1e-5, verbose=False):
    """ Explore the largest and smallest possible values for the specified
    index of a separating hyperplane vector, among all hyperplanes with
    classification error <= err.

    Args:
        X (nxm ndarray):
            Data matrix whose rows correspond to data points, and columns
            correspond to features.
        y (nx1 ndarray):
            Vectory of binary class labels in {0,1}.
        index (int):
            Index of the w vector to check (i.e. which feature).
        err (int):
            Maximum allowable classification error.
        eps (float, optional):
            Margin to use in place of strict inequalities. Default is 1e-5.
        verbose (bool, optional):
            If True, displays Gurobi solver output. Default is False.
        
    Returns:
        float:
            Smallest possible value of w[index] among all solutions with
            classification error <= err and ||w||_2=1.
        float:
            Largest possible value of w[index] among all solutions with
            classification error <= err and ||w||_2=1.

    Raises:
        RuntimeError:
            If the MIP is not solved to optimality.
    """
    model = grb.Model("heuristic")
    C1 = np.where(y == 0)[0]
    C2 = np.where(y == 1)[0]

    n = X.shape[0]
    p = X.shape[1]
    b = model.addVar(lb=-grb.GRB.INFINITY, name="b")
    z = model.addVars(n, name="z", vtype=grb.GRB.BINARY)
    w = model.addVars(p, lb=-grb.GRB.INFINITY, name="w")
    t = model.addVars(p, name="t")

    model.addConstr(grb.quicksum(z) == err)

    for i in range(p):
        model.addConstr(t[i] == grb.abs_(w[i]))
    model.addConstr(grb.quicksum(t) == 1)

    w[index].obj = 1

    for i in C1:
        model.addConstr(
            (z[i] == 0) >> (b >= eps + grb.quicksum(X[i, j] * w[j] for j in range(p)))
        )

    for i in C2:
        model.addConstr(
            (z[i] == 0) >> (b + eps <= grb.quicksum(X[i, j] * w[j] for j in range(p)))
        )

    model.setParam("OutputFlag", verbose)

    model.optimize()
    if model.status != grb.GRB.OPTIMAL:
        raise RuntimeError("Heuristic MIP not solved to optimality")
    w0 = np.array([w[i].X for i in range(p)])
    b0 = b.X
    b0 /= np.linalg.norm(w0, ord=2)
    w0 /= np.linalg.norm(w0, ord=2)
    w_min = w0[index]

    model.setAttr("ModelSense", grb.GRB.MAXIMIZE)
    model.optimize()
    if model.status != grb.GRB.OPTIMAL:
        raise RuntimeError("Heuristic MIP not solved to optimality")
    w0 = np.array([w[i].X for i in range(p)])
    b0 = b.X
    b0 /= np.linalg.norm(w0, ord=2)
    w0 /= np.linalg.norm(w0, ord=2)
    w_max = w0[index]

    return w_min, w_max


def max_margin(X, y, err, return_width=False, verbose=False):
    """ Compute a maximum-margin separating hyperplane with classification
    error <= err.
    
    Args:
        X (nxm ndarray):
            Data matrix whose rows correspond to data points, and columns
            correspond to features.
        y (nx1 ndarray):
            Vectory of binary class labels in {0,1}.
        err (int):
            Maximum allowable classification error.
        return_width (bool, optional):
            If True, returns the width of the resulting margin. Default is
            False.
        verbose (bool, optional):
            If True, displays Gurobi solver output. Default is False.
        
    Returns:
        mx1 ndarray:
            Normalized normal vector to a maximum-margin separating hyperplane.
        float:
            Scalar offset of the hyperplane.
        float (if return_width=True):
            Width of the resulting maximum-width margin.

    Raises:
        RuntimeError:
            If the MIP is not solved to optimality.
    """
    model = grb.Model("heuristic")
    C1 = np.where(y == 0)[0]
    C2 = np.where(y == 1)[0]

    n = X.shape[0]
    p = X.shape[1]
    b = model.addVar(lb=-grb.GRB.INFINITY, name="b")
    z = model.addVars(n, name="z", vtype=grb.GRB.BINARY)
    w = model.addVars(p, lb=-grb.GRB.INFINITY, name="w")

    model.addConstr(grb.quicksum(z) == err)

    obj = grb.QuadExpr(grb.quicksum(w[i] * w[i] for i in range(p)))
    model.setObjective(obj, sense=grb.GRB.MINIMIZE)

    for i in C1:
        model.addConstr(
            (z[i] == 0) >> (b >= 1 + grb.quicksum(X[i, j] * w[j] for j in range(p)))
        )

    for i in C2:
        model.addConstr(
            (z[i] == 0) >> (b + 1 <= grb.quicksum(X[i, j] * w[j] for j in range(p)))
        )

    model.setParam("OutputFlag", verbose)

    model.optimize()
    if model.status != grb.GRB.OPTIMAL:
        raise RuntimeError("Heuristic MIP not solved to optimality")

    w0 = np.array([w[i].X for i in range(len(w))])
    b0 = b.X
    scale = np.linalg.norm(w0, ord=2)
    if return_width:
        width = 2 / np.sqrt(model.objval)
        return w0 / scale, b0 / scale, width
    return w0 / scale, b0 / scale


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.random.seed(0)
    X = 0.3 * np.random.randn(100, 2)
    X[50:, 0] += 1
    y = np.zeros(100)
    y[50:] = 1
    y[5] = 1

    res = dsvmf(X, y, sens=True)
    print(f'Classification error: {res["err"]}')
    print(f'Margin width: {res["width"]:.6f}')
    print("Sensitivity analysis")
    for index in range(X.shape[1]):
        print(f'\t{index}: [{res["sens"][index][0]:.4f}, {res["sens"][index][1]:.4f}]')

    w, b, width = res["w"], res["b"], res["width"]

    C1 = np.where(y == 0)[0]
    C2 = np.where(y == 1)[0]

    x = np.arange(np.min(X[:, 0]), np.max(X[:, 0]), 0.01)
    plane = (b - w[0] * x) / w[1]
    plane_max = (b + 0.5 * width - w[0] * x) / w[1]
    plane_min = (b - 0.5 * width - w[0] * x) / w[1]

    ax = plt.gca()

    ax.plot(X[C1, 0], X[C1, 1], "or")
    ax.plot(X[C2, 0], X[C2, 1], "ob")
    ax.plot(x, plane, "k")
    ax.plot(x, plane_min, "--k")
    ax.plot(x, plane_max, "--k")

    ax.set_xlim([np.min(X[:, 0]), np.max(X[:, 0])])
    ax.set_ylim([np.min(X[:, 1]), np.max(X[:, 1])])

    plt.show()
