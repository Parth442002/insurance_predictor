import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error


def performanceOutPut(y_test, y_pred, model_name=None):
    errors = list()
    print(f"Performance Analysis for {model_name}")
    for i in range(len(y_test)):
        err = (y_test[i] - y_pred[i])**2
        errors.append(err)
        #print('>%.1f, %.1f = %.3f' % (y_test[i], y_pred[i], err))
        plt.title(f"{model_name}")
        plt.plot(errors)
        #plt.xticks(ticks=[i for i in range(len(errors))], labels=y_pred)
        plt.xlabel('Predicted Value')
        plt.ylabel('Mean Squared Error')
        plt.show()
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Squared Error MSE :-{mse}")
    print(f"Root Mean Squared Error MSE :-{rmse}")
    print(f"Mean absolute Error MAE :-{mae}")
