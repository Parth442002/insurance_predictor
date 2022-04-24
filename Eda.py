from pandas_profiling import ProfileReport
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat
import seaborn as sns
dataset = pd.read_csv('insurance.csv')

report = ProfileReport(dataset)
report.to_file('EDA.html')

