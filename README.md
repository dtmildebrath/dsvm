# dsvm
A mixed-integer programming based tool for computing (linear) separating
hyperplanes.  The code uses the [Gurobi solver](https://www.gurobi.com/) (free
license for academics) to compute a minimum-classification error,
maximum-margin (in that order) separating hyperplane for binary classification
data. It also optionally performs a sensitivity analysis, by computing the
minimum and maximum values of the (normalized) separating hyperplane
coefficients for each feature among the set of all minimum-error separating
hyperplanes.

Because the code relies on mixed-integer programming, it is typically only
suitable for small datasets.

One useful application is to verify whether a given dataset is or is not
linearly separable, via the `min_error` function.
