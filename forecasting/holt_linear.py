from statsmodels.tsa.holtwinters import Holt


def forecast_holt_linear(y, count):
    model = Holt(
        y,
        exponential=False,
        damped_trend=True
    )

    fitted = model.fit(optimized=True)

    return fitted.forecast(steps=count)