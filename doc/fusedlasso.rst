Fused Lasso models
=============================================

.. function:: fit(FusedLasso, y, λ)

    Fits the fused Lasso model:

    .. math:: \underset{\beta}{\operatorname{argmin}} \frac{1}{2} \sum_{k=1}^N(y_k - \beta_k)^2 + \lambda \sum_{k=2}^N |\beta_k - \beta_{k-1}|

    The model coefficients can be obtained by calling ``coef`` on the
    returned model object.

    For details of the algorithm, see Johnson, N. A. (2013). A dynamic
    programming algorithm for the fused lasso and L0-segmentation.
    Journal of Computational and Graphical Statistics, 22(2), 246–260.
    doi:10.1080/10618600.2012.681238
