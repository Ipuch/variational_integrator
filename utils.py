from typing import Union, List
import numpy as np


def piecewise_constant_u(t: float, t_eval: Union[np.ndarray, List[float]], u: np.ndarray) -> float:
    """
    This function computes the values of u at any time t as piecewise constant function.
    As the last element is an open end point, we need to use the previous element.

    Parameters
    ----------
    t : float
        time to evaluate the piecewise constant function
    t_eval : Union[np.ndarray, List[float]]
        array of times t the controls u are evaluated at
    u : np.ndarray
        arrays of controls u over the tspans of t_eval

    Returns
    -------
    u_t: float
        value of u at time t
    """

    def previous_t(t: float, t_eval: Union[np.ndarray, List[float]]) -> int:
        """
        find the closest time in t_eval to t

        Parameters
        ----------
        t : float
            time to compare to t_eval
        t_eval : Union[np.ndarray, List[float]]
            array of times to compare to t

        Returns
        -------
        idx: int
            index of the closest previous time in t_eval to t
        """
        diff = t_eval - t
        # keep only positive values
        diff = diff[diff <= 0]
        return int(np.argmin(np.abs(diff)))

    def previous_t_except_the_last_one(t: float, t_eval: Union[np.ndarray, List[float]]) -> int:
        """
        find the closest time in t_eval to t

        Parameters
        ----------
        t : float
            time to compare to t_eval
        t_eval : Union[np.ndarray, List[float]]
            array of times to compare to t

        Returns
        -------
        int
            index of the closest previous time in t_eval to t
        """
        out = previous_t(t, t_eval)
        if out == len(t_eval) - 1:
            return out - 1
        else:
            return out

    if t_eval.shape[0] != u.shape[1]:
        raise ValueError("t_eval and u must have the same length, please report the bug to the developers")

    return u[:, previous_t_except_the_last_one(t, t_eval)]
