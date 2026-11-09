"""Ordinal XGBoost classifier with an all-threshold logistic objective."""

import warnings
from numbers import Integral, Real

import numpy as np
import scipy.optimize
import scipy.special
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted, validate_data

from skordinal.utils._optimize import _check_optimize_result
from skordinal.utils.extmath import cumproba_to_proba, repair_cumproba
from skordinal.utils.validation import check_ordinal_targets


class OrdinalXGBClassifier(ClassifierMixin, BaseEstimator):
    """Ordinal XGBoost classifier with an all-threshold logistic objective.

    The estimator trains one XGBoost [3]_ booster whose output ``f(x)`` is a
    latent ordinal score, as in thresholded ensembles [2]_, with the
    all-threshold logistic loss [1]_ as custom objective. The ``n_classes - 1``
    thresholds stay at the logits of the cumulative class frequencies while
    the trees grow, and ``threshold_update`` then sets ``thresholds_``. The
    booster parameters have the names and defaults of
    ``xgboost.XGBClassifier``.

    Requires the ``xgboost`` package, installed with
    ``pip install "skordinal[xgboost]"``.

    Parameters
    ----------
    max_depth : int, default=6
        Maximum tree depth for base learners.

    max_leaves : int, default=0
        Maximum number of leaves, 0 indicates no limit.

    max_bin : int, default=256
        If using histogram-based algorithm, maximum number of bins per feature.

    grow_policy : {"depthwise", "lossguide"}, default="depthwise"
        Tree growing policy.

        - ``depthwise`` favors splitting at nodes closest to the root.
        - ``lossguide`` favors splitting at nodes with highest loss change.

    learning_rate : float, default=0.3
        Boosting learning rate (xgb's "eta").

    n_estimators : int, default=100
        Number of gradient boosted trees. Equivalent to number of boosting rounds.

    verbosity : int, default=1
        The degree of verbosity. Valid values are 0 (silent) - 3 (debug).

    tree_method : {"auto", "exact", "approx", "hist"}, default="auto"
        Specify which tree method to use. ``"auto"`` is the same as ``"hist"``.
        It's recommended to study this option from the parameters document
        :doc:`tree method <xgboost:treemethod>`.

    n_jobs : int or None, default=None
        Number of parallel threads used to run xgboost. When used with other
        Scikit-Learn algorithms like grid search, you may choose which algorithm
        to parallelize and balance the threads. Creating thread contention will
        significantly slow down both algorithms.

    gamma : float, default=0.0
        (min_split_loss) Minimum loss reduction required to make a further
        partition on a leaf node of the tree.

    min_child_weight : float, default=1.0
        Minimum sum of instance weight(hessian) needed in a child.

    max_delta_step : float, default=0.0
        Maximum delta step we allow each tree's weight estimation to be.

    subsample : float, default=1.0
        Subsample ratio of the training instance.

    sampling_method : {"uniform", "gradient_based"}, default="uniform"
        Sampling method. Used only by the GPU version of ``hist`` tree method.

        - ``uniform`` selects random training instances uniformly.
        - ``gradient_based`` selects random training instances with higher
          probability when the gradient and hessian are larger. (cf. CatBoost)

    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.

    colsample_bylevel : float, default=1.0
        Subsample ratio of columns for each level.

    colsample_bynode : float, default=1.0
        Subsample ratio of columns for each split.

    reg_alpha : float, default=0.0
        L1 regularization term on weights (xgb's alpha).

    reg_lambda : float, default=1.0
        L2 regularization term on weights (xgb's lambda).

    random_state : int, RandomState instance, Generator instance or None, \
default=None
        Random number seed.

    missing : float, default=np.nan
        Value in the data which needs to be present as a missing value. Default
        to :py:data:`numpy.nan`.

    num_parallel_tree : int, default=1
        Used for boosting random forest.

    monotone_constraints : dict, str, tuple or None, default=None
        Constraint of variable monotonicity. See
        :doc:`tutorial <xgboost:tutorials/monotonic>` for more information.

    interaction_constraints : str, list or None, default=None
        Constraints for interaction representing permitted interactions. The
        constraints must be specified in the form of a nested list, e.g.
        ``"[[0, 1], [2, 3, 4]]"``, where each inner list is a group of indices
        of features that are allowed to interact with each other. Pass the
        indices as a string, as in the example, since a nested list takes
        feature names and needs ``X`` with feature names. See
        :doc:`tutorial <xgboost:tutorials/feature_interaction_constraint>` for
        more information.

    importance_type : {"gain", "weight", "cover", "total_gain", "total_cover"}, \
default="gain"
        The feature importance type for the ``feature_importances_`` property.

    device : str, default="cpu"
        Device ordinal, available options are ``cpu``, ``cuda``, and ``gpu``.

    feature_weights : array-like of shape (n_features,) or None, default=None
        Weight for each feature, defines the probability of each feature being
        selected when colsample is being used. All values must be non-negative,
        otherwise a ``ValueError`` is thrown.

    threshold_update : {"fixed", "refit", "input_median", "input_mean", \
"latent_median"}, default="refit"
        How ``thresholds_`` is obtained. The booster trains against the prior
        thresholds, the logits of the unweighted cumulative class frequencies.

        - ``"fixed"`` keeps those prior thresholds.
        - ``"refit"`` minimises the weighted loss over the thresholds with f
          fixed, with ``threshold_solver``.
        - ``"input_median"`` takes the midpoint between the projections of the
          per-class median input patterns of consecutive classes (INP-MED),
          skipping missing values.
        - ``"input_mean"`` does the same with mean patterns (INP-MEAN).
        - ``"latent_median"`` takes the midpoint between the median
          projections of consecutive classes (LAT-MED).

    threshold_solver : {"lbfgs", "sgd"}, default="lbfgs"
        Solver of ``threshold_update="refit"``, ignored otherwise. ``"sgd"``
        takes one weighted gradient step per training row, in row order, of
        size ``threshold_learning_rate``, so the result depends on the order of
        the rows. ``"lbfgs"`` runs L-BFGS-B on the weighted mean loss with
        ``maxiter=100``, ``maxls=50``, ``ftol=64 * np.finfo(float).eps`` and
        ``gtol=1e-8``.

    threshold_learning_rate : float or None, default=None
        Step of ``threshold_solver="sgd"``, ignored otherwise. ``None`` reuses
        ``learning_rate``.

    class_weight : dict, "balanced" or None, default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. The
        ``"balanced"`` mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``. The weights apply to
        the booster objective and to ``threshold_update="refit"``.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.

    n_features_in_ : int
        Number of features seen during ``fit``.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during ``fit``. Defined only when ``X`` has
        feature names that are all strings.

    thresholds_ : ndarray of shape (n_classes - 1,)
        Thresholds on the projection that separate consecutive classes, in
        non-decreasing order.

    booster_ : xgboost.Booster
        The trained booster.

    feature_importances_ : ndarray of shape (`n_features_in_`,)
        Feature importances of type ``importance_type``, normalised to sum 1
        unless they are all 0. Unused features get 0.

    Notes
    -----
    For each sample ``i`` and threshold ``k`` the objective is the logistic
    loss

    .. math::

        \\log\\bigl(1 + \\exp(-s_{ik}(\\theta_k - f(x_i)))\\bigr)

    with ``s_ik = +1`` if ``y_i <= k`` and ``-1`` otherwise. It depends only
    on the differences between ``f`` and the thresholds, which absorb
    XGBoost's base score, so ``base_score`` is not a parameter.

    ``predict`` returns the first class whose cumulative probability reaches
    0.5, in general not ``classes_[argmax(predict_proba(X), axis=1)]``.

    References
    ----------
    .. [1] J. D. M. Rennie and N. Srebro, "Loss functions for preference
       levels: regression with discrete ordered labels," in Proc. IJCAI
       Multidisciplinary Workshop on Advances in Preference Handling, 2005.

    .. [2] H.-T. Lin and L. Li, "Large-margin thresholded ensembles for
       ordinal regression: theory and practice," in Proc. Algorithmic
       Learning Theory (ALT), Lecture Notes in Computer Science,
       vol. 4264, pp. 319-333, Springer, 2006.

    .. [3] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting
       system," in Proc. 22nd ACM SIGKDD International Conference on
       Knowledge Discovery and Data Mining (KDD), pp. 785-794, 2016.
       https://doi.org/10.1145/2939672.2939785

    Examples
    --------
    >>> from skordinal.ensemble import OrdinalXGBClassifier
    >>> from skordinal.datasets import make_ordinal_classification
    >>> X, y = make_ordinal_classification(
    ...     n_samples=120, n_classes=3, random_state=0
    ... )
    >>> clf = OrdinalXGBClassifier(n_estimators=20).fit(X, y)
    >>> clf.predict_proba(X[:3]).shape
    (3, 3)
    """

    _parameter_constraints: dict = {
        "max_depth": [Interval(Integral, 0, None, closed="left")],
        "max_leaves": [Interval(Integral, 0, None, closed="left")],
        "max_bin": [Interval(Integral, 2, None, closed="left")],
        "grow_policy": [StrOptions({"depthwise", "lossguide"})],
        "learning_rate": [Interval(Real, 0.0, None, closed="neither")],
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "verbosity": [Interval(Integral, 0, 3, closed="both")],
        "tree_method": [StrOptions({"auto", "exact", "approx", "hist"})],
        "n_jobs": [None, Integral],
        "gamma": [Interval(Real, 0.0, None, closed="left")],
        "min_child_weight": [Interval(Real, 0.0, None, closed="left")],
        "max_delta_step": [Interval(Real, 0.0, None, closed="left")],
        "subsample": [Interval(Real, 0.0, 1.0, closed="right")],
        "sampling_method": [StrOptions({"uniform", "gradient_based"})],
        "colsample_bytree": [Interval(Real, 0.0, 1.0, closed="right")],
        "colsample_bylevel": [Interval(Real, 0.0, 1.0, closed="right")],
        "colsample_bynode": [Interval(Real, 0.0, 1.0, closed="right")],
        "reg_alpha": [Interval(Real, 0.0, None, closed="left")],
        "reg_lambda": [Interval(Real, 0.0, None, closed="left")],
        "random_state": ["random_state", np.random.Generator],
        "missing": [Real],
        "num_parallel_tree": [Interval(Integral, 1, None, closed="left")],
        "monotone_constraints": [dict, str, tuple, None],
        "interaction_constraints": [str, list, None],
        "importance_type": [
            StrOptions({"gain", "weight", "cover", "total_gain", "total_cover"})
        ],
        "device": [str],
        "feature_weights": ["array-like", None],
        "threshold_update": [
            StrOptions(
                {"fixed", "refit", "input_median", "input_mean", "latent_median"}
            )
        ],
        "threshold_solver": [StrOptions({"lbfgs", "sgd"})],
        "threshold_learning_rate": [Interval(Real, 0.0, None, closed="neither"), None],
        "class_weight": [StrOptions({"balanced"}), dict, None],
    }

    def __init__(
        self,
        *,
        max_depth=6,
        max_leaves=0,
        max_bin=256,
        grow_policy="depthwise",
        learning_rate=0.3,
        n_estimators=100,
        verbosity=1,
        tree_method="auto",
        n_jobs=None,
        gamma=0.0,
        min_child_weight=1.0,
        max_delta_step=0.0,
        subsample=1.0,
        sampling_method="uniform",
        colsample_bytree=1.0,
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=None,
        missing=np.nan,
        num_parallel_tree=1,
        monotone_constraints=None,
        interaction_constraints=None,
        importance_type="gain",
        device="cpu",
        feature_weights=None,
        threshold_update="refit",
        threshold_solver="lbfgs",
        threshold_learning_rate=None,
        class_weight=None,
    ):
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_bin = max_bin
        self.grow_policy = grow_policy
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.verbosity = verbosity
        self.tree_method = tree_method
        self.n_jobs = n_jobs
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.sampling_method = sampling_method
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.missing = missing
        self.num_parallel_tree = num_parallel_tree
        self.monotone_constraints = monotone_constraints
        self.interaction_constraints = interaction_constraints
        self.importance_type = importance_type
        self.device = device
        self.feature_weights = feature_weights
        self.threshold_update = threshold_update
        self.threshold_solver = threshold_solver
        self.threshold_learning_rate = threshold_learning_rate
        self.class_weight = class_weight

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training patterns. Missing values are allowed.

        y : array-like of shape (n_samples,)
            Ordinal target labels. Must contain at least 2 unique classes.

        Returns
        -------
        self : OrdinalXGBClassifier
            Fitted estimator.

        Raises
        ------
        ImportError
            If ``xgboost`` is not installed.

        ValueError
            If ``y`` contains fewer than 2 unique classes.
        """
        # Import xgboost, an optional dependency
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "OrdinalXGBClassifier requires xgboost. Install it with "
                "'pip install \"skordinal[xgboost]\"'."
            ) from exc

        # Validate the data, encode the labels and weight each sample by its class
        X, y = validate_data(
            self, X, y, dtype=np.float64, ensure_all_finite="allow-nan"
        )
        self.classes_, y_enc = check_ordinal_targets(y)
        sample_weight = compute_sample_weight(self.class_weight, y)

        # Start the thresholds at the logits of the cumulative class frequencies
        n_classes = len(self.classes_)
        prior = np.bincount(y_enc, minlength=n_classes)[:-1].cumsum() / len(y_enc)
        thresholds = scipy.special.logit(prior)

        # Sign each sample and threshold pair, +1 if y_i <= k and -1 otherwise
        s = np.where(y_enc[:, None] <= np.arange(n_classes - 1), 1.0, -1.0)

        # Define the objective as the weighted gradient and Hessian of the loss in f
        def objective(preds, dtrain):
            sigma = scipy.special.expit(s * (preds[:, None] - thresholds))
            grad = (s * sigma).sum(axis=1) * sample_weight
            hess = (sigma * (1.0 - sigma)).sum(axis=1) * sample_weight
            return grad, hess

        # Draw the seed as XGBModel.get_params does
        seed = self.random_state
        if isinstance(seed, np.random.RandomState):
            seed = seed.randint(np.iinfo(np.int32).max)
        elif isinstance(seed, np.random.Generator):
            seed = int(seed.integers(np.iinfo(np.int32).max))

        # Collect the booster parameters, xgboost ignoring those that are None
        params = {
            "tree_method": self.tree_method,
            "disable_default_eval_metric": True,
            "max_depth": self.max_depth,
            "max_leaves": self.max_leaves,
            "max_bin": self.max_bin,
            "grow_policy": self.grow_policy,
            "learning_rate": self.learning_rate,
            "verbosity": self.verbosity,
            "gamma": self.gamma,
            "min_child_weight": self.min_child_weight,
            "max_delta_step": self.max_delta_step,
            "subsample": self.subsample,
            "sampling_method": self.sampling_method,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "colsample_bynode": self.colsample_bynode,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "num_parallel_tree": self.num_parallel_tree,
            "monotone_constraints": self.monotone_constraints,
            "interaction_constraints": self.interaction_constraints,
            "device": self.device,
            "nthread": self.n_jobs,
            "seed": seed,
        }

        # Train the booster, scoping the verbosity to this fit as XGBModel.fit does
        feature_names = getattr(self, "feature_names_in_", None)
        with xgb.config_context(verbosity=self.verbosity):
            dtrain = xgb.DMatrix(
                X,
                missing=self.missing,
                feature_names=None if feature_names is None else list(feature_names),
                feature_weights=self.feature_weights,
                nthread=self.n_jobs,
            )
            self.booster_ = xgb.train(
                params, dtrain, num_boost_round=self.n_estimators, obj=objective
            )

        # Refit the thresholds to the weighted loss with the booster fixed
        if self.threshold_update == "refit":
            projections = self._project(X)
            if self.threshold_solver == "lbfgs":
                # Average the loss over the total weight, as LogisticRegression does
                weights = sample_weight[:, None] / sample_weight.sum()

                def objective_and_grad(thresholds):
                    # Each pair costs log(1 + exp(u)), u = s * (f - theta_k)
                    u = s * (projections[:, None] - thresholds)
                    loss = (np.logaddexp(0.0, u) * weights).sum()
                    grad = -(s * scipy.special.expit(u) * weights).sum(axis=0)
                    return float(loss), grad

                result = scipy.optimize.minimize(
                    objective_and_grad,
                    x0=thresholds,
                    method="L-BFGS-B",
                    jac=True,
                    options={
                        "maxiter": 100,
                        "maxls": 50,
                        "ftol": 64 * np.finfo(float).eps,
                        "gtol": 1e-8,
                    },
                )
                _check_optimize_result("lbfgs", result)
                thresholds = result.x
            else:
                lr = self.threshold_learning_rate or self.learning_rate
                # Step theta_k down for k < y_i and up for k >= y_i, row by row
                steps = lr * sample_weight[:, None] * s
                for projection, s_i, step_i in zip(projections, s, steps):
                    sig = scipy.special.expit(s_i * (projection - thresholds))
                    thresholds += step_i * sig

        # Bisect the projected input centers of consecutive classes
        if self.threshold_update in ("input_median", "input_mean"):
            center = (
                np.nanmedian if self.threshold_update == "input_median" else np.nanmean
            )
            X_nan = np.where(X == self.missing, np.nan, X)
            # Leave a column missing in a whole class as NaN, without a warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                centers = np.array(
                    [center(X_nan[y_enc == k], axis=0) for k in range(n_classes)]
                )
            class_projections = self._project(centers)
            thresholds = (class_projections[:-1] + class_projections[1:]) / 2.0

        # Bisect the median projections of consecutive classes
        if self.threshold_update == "latent_median":
            projections = self._project(X)
            class_projections = np.array(
                [np.median(projections[y_enc == k]) for k in range(n_classes)]
            )
            thresholds = (class_projections[:-1] + class_projections[1:]) / 2.0

        # Sort the thresholds, which the midpoint rules and SGD may leave unordered
        self.thresholds_ = np.sort(thresholds)
        return self

    @property
    def feature_importances_(self):
        """Return the importance of each feature according to ``importance_type``.

        Computed as ``XGBModel.feature_importances_`` does, from the score of
        ``booster_``, 0 for the unused features, divided by its sum unless it is 0.

        Returns
        -------
        feature_importances : ndarray of shape (`n_features_in_`,), dtype float32
            Normalised importance of each feature.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        check_is_fitted(self)
        score = self.booster_.get_score(importance_type=self.importance_type)
        feature_names = self.booster_.feature_names
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(self.n_features_in_)]
        importances = np.array(
            [score.get(name, 0.0) for name in feature_names], dtype=np.float32
        )
        total = importances.sum()
        if total == 0:
            return importances
        return importances / total

    def get_booster(self):
        """Get the underlying xgboost Booster of this model.

        This will raise an exception when fit was not called.

        Returns
        -------
        booster : xgboost.Booster
            The trained booster, ``booster_``.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        check_is_fitted(self)
        return self.booster_

    def predict_cumproba(self, X):
        """Predict cumulative class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input patterns. Missing values are allowed.

        Returns
        -------
        F : ndarray of shape (n_samples, n_classes - 1)
            ``F[:, k]`` equals ``P(Y <= k+1 | x)`` for ``k = 0, ..., K-2``,
            non-decreasing along the class axis.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        check_is_fitted(self)
        X = validate_data(
            self, X, reset=False, dtype=np.float64, ensure_all_finite="allow-nan"
        )
        return repair_cumproba(self._cumproba(X))

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input patterns. Missing values are allowed.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Non-negative class probabilities, each row summing to 1.0.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        check_is_fitted(self)
        X = validate_data(
            self, X, reset=False, dtype=np.float64, ensure_all_finite="allow-nan"
        )
        return cumproba_to_proba(self._cumproba(X), repair=True)

    def predict(self, X):
        """Predict class labels for samples in X.

        The predicted class is the number of fitted thresholds that the
        projection ``f(x)`` exceeds, a tie going to the lower class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input patterns. Missing values are allowed.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels drawn from ``self.classes_``.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        check_is_fitted(self)
        X = validate_data(
            self, X, reset=False, dtype=np.float64, ensure_all_finite="allow-nan"
        )
        f = self._project(X)
        labels_enc = np.searchsorted(self.thresholds_, f, side="left")
        return self.classes_[labels_enc]

    def predict_projection(self, X):
        """Predict the scalar projection for X.

        The booster output ``f(x)``, base score included, is the raw latent
        projection that ``thresholds_`` partitions into class regions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input patterns. Missing values are allowed.

        Returns
        -------
        projection : ndarray of shape (n_samples,)
            Raw booster output for each sample.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        check_is_fitted(self)
        X = validate_data(
            self, X, reset=False, dtype=np.float64, ensure_all_finite="allow-nan"
        )
        return self._project(X)

    def _project(self, X):
        """Compute the raw latent projection for pre-validated X."""
        import xgboost as xgb

        # Scope the verbosity to this prediction, as XGBModel.predict does
        with xgb.config_context(verbosity=self.verbosity):
            projection = self.booster_.inplace_predict(X, missing=self.missing)
        return projection.astype(np.float64)

    def _cumproba(self, X):
        """Compute raw cumulative probabilities on pre-validated X."""
        f = self._project(X)
        return scipy.special.expit(self.thresholds_[None, :] - f[:, None])

    def __sklearn_tags__(self):
        """Declare that missing values are accepted."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
