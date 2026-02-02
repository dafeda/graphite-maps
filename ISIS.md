- ISIS assumes a regularization step like LASSO/SCAD. Forward stagewise is a different estimator with different selection behavior. Any ISIS guarantees in Fan & Lv (2008) do not transfer. The method is implemented as a heuristic.

- The conditioning step uses OLS residuals on the currently selected step, but the selected set itself comes from stagewise, which is biased/shrunk. That bias leaks into the residuals and can systematically supress true signals (especially colinear ones).
