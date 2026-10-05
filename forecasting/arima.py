import warnings
import itertools
import numpy as np

from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def forecast_arima(y, count):
    """
    Forecast using ARIMA.

    ARIMA order is selected using AIC.
    d = 0 and d = 1 are evaluated separately.
    """

    y = np.asarray(y, dtype=float)

    if len(y) < 10:
        raise ValueError(
            "Not enough data for ARIMA forecasting."
        )

    best_by_d = {}

    # --------------------------------------------------------
    # Search ARIMA orders
    # --------------------------------------------------------

    for d in [0, 1]:

        best_aic = np.inf
        best_order = None

        for p, q in itertools.product(
            range(0, 4),
            range(0, 4)
        ):

            try:

                model = ARIMA(
                    y,
                    order=(p, d, q)
                )

                fitted = model.fit()

                if fitted.aic < best_aic:

                    best_aic = fitted.aic
                    best_order = (p, d, q)

            except Exception:
                continue

        if best_order is not None:

            best_by_d[d] = (
                best_order,
                best_aic
            )

    if not best_by_d:

        raise ValueError(
            "Unable to find a valid ARIMA model."
        )

    # --------------------------------------------------------
    # Select the best order
    # --------------------------------------------------------

    best_order, best_aic = min(
        best_by_d.values(),
        key=lambda x: x[1]
    )

    # --------------------------------------------------------
    # Final ARIMA model
    # --------------------------------------------------------

    model = ARIMA(
        y,
        order=best_order
    )

    fitted = model.fit()

    forecast = fitted.forecast(
        steps=count
    )

    return np.asarray(forecast), best_order