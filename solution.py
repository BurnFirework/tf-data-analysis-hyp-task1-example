import pandas as pd
import numpy as np

from scipy.stats import norm

chat_id = 860138765

def solution(x_success: int, 
             x_cnt: int, 
             y_success: int, 
             y_cnt: int) -> bool:
    imp_crit = 0.09
    p_control = x_success / x_cnt
    p_test = y_success / y_cnt
    p_combined = (x_success + y_success) / (x_cnt + y_cnt)

    se = np.sqrt(p_combined * (1 - p_combined) * (1 / x_cnt + 1 / y_cnt))
    z_score = (p_test - p_control) / se
    z_critical = abs(norm.ppf(imp_crit / 2))
    return (abs(z_score) > z_critical)
