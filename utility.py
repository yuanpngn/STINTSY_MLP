import numpy as np


def compute_RMSE(y_true, y_pred):
    """Computes the Root Mean Squared Error (RMSE) given the ground truth
    values and the predicted values.

    Arguments:
        y_true {np.ndarray} -- A numpy array of shape (N, 1) containing
        the ground truth values.
        y_pred {np.ndarray} -- A numpy array of shape (N, 1) containing
        the predicted values.

    Returns:
        float -- Root Mean Squared Error (RMSE)
    """

    # TODO: Compute the Root Mean Squared Error
    rmse = (y_pred - y_true) ** 2
    rmse = np.mean(rmse)
    rmse = np.sqrt(rmse)

    return rmse


def poly_feature_transform(X, poly_order=1):
    """Transforms the input data X to match the specified polynomial order.

    Arguments:
        X {np.ndarray} -- A numpy array of shape (N, D) containing N instances
        with D features.
        poly_order {int} -- Order of polynomial of the hypothesis function

    Returns:
        np.ndarray -- A numpy array of shape (N, (D * order) + 1) representing
        the transformed features following the specified `poly_order`
    """
    f_transform = X
    if poly_order > 1:  
        for order in range (2, poly_order+1):
            transformed_feature = X ** order
            f_transform = np.column_stack((f_transform, transformed_feature))
            
    # TODO: Add features to X until poly_order
    ones_column = np.ones((f_transform.shape[0], 1))
    f_transform = np.column_stack((f_transform, ones_column))
    return f_transform