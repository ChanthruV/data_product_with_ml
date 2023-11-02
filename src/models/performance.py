def print_mse(y_actuals, y_preds, set_name=None):
    """Print the Mean Squared Error (MSE) for the provided data

    Parameters
    ----------
    y_actuals : Numpy Array
        Actual target
    
    y_preds : Numpy Array
        Predicted target values

    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import mean_squared_error
    
    mse = mean_squared_error(y_actuals, y_preds)
    
    print(f'MSE {set_name}: {mse}')
