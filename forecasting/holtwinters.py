from statsmodels.tsa.holtwinters import ExponentialSmoothing


def forecast_holt_winters(y, seasonal_period, count):

    model = ExponentialSmoothing(
        y,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_period,
        initialization_method="estimated"
    )

    fitted = model.fit(
        optimized=True,
        use_brute=True
    )

    return fitted.forecast(steps=count)