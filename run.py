from ARIMA_Experiments import *
from GRU_Experiments import *
from utils.time_utils import get_TimeStamp_str

print('start runing at:', get_TimeStamp_str())
for i in range(1,11):
    GRU_exp1(ds=i)
    GRU_exp2(ds=i)
    GRU_exp3(ds=i)
    GRU_exp4(ds=i)
    ARIMA_exp2(ds=i)
    ARIMA_exp3(ds=i)
    ARIMA_exp4(ds=i)
print('End runing at:', get_TimeStamp_str())  