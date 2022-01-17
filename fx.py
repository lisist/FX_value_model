import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from fredapi import Fred

fred_api_key = 'a2524d0d1124080eb016f900416e4df2'

def twin_deficit():
    usd_nominal_effective = pd.read_excel('total.xlsx', sheet_name='USDEffective', index_col='Date')
    us_twin_defic = -pd.read_excel('total.xlsx', sheet_name='US_TWIN_DEFI', index_col='Date') 

    merged_df = usd_nominal_effective.merge(us_twin_defic,left_index=True,right_index=True)


    ## 참고 : Python - 2축 그래프 그리기
    fig = plt.figure(figsize=(7,4))

    ax1 = fig.subplots()
    ax2 = ax1.twinx()

    # ax1.set_title('dd') 

    line1 = ax1.plot(merged_df['BISNUSN index'].shift(-10))
    line2 = ax2.plot(merged_df['.TWIN G index'],linestyle="--",color='r')

    lines = line1 + line2
    plt.legend(lines,['Nominal Effective US dollar (ls, 10M lagged)','US : Sum of Current Account and Fiscal Balance (rs)'],loc=2)

    ax1.set_ylim(60,160)
    ax2.set_ylim(-20,4)
    ax2.set_ylabel("%GDP")

    plt.tight_layout()
    plt.savefig("./images/US_TwinD_USD.png")


def z_score(data,time_line):
   
    rolling_average = data.rolling(window=time_line).mean()
    rolling_st = data.rolling(window=time_line).std()
    rolling_df = pd.DataFrame({"data":data.values,"rolling_av":rolling_average.values,"rolling_st":rolling_st.values},index=data.index)[time_line-1:]
    return (rolling_df.data-rolling_df.rolling_av)/(rolling_df.rolling_st)


def basic_model():
    time_line = 36
    criteria_line = 0

    cycle_data = pd.read_excel('total.xlsx',sheet_name='markit_pmis',index_col='Date')
    momentum_data = pd.read_excel('total.xlsx',sheet_name='fx_monthly',index_col='Date')
    yield_diff_data = pd.read_excel('total.xlsx',sheet_name='st_yield_diff',index_col='Date')

    k = 0

    for i in cycle_data.columns[1:]:
        k += 1

        cycle = cycle_data[i] - cycle_data['USD']
        cycle_z = z_score(cycle, time_line)

        momentum = (momentum_data[i] - momentum_data[i].rolling(window=8).mean()).dropna()
        momentum_z = z_score(momentum,time_line)
        
        momentum_z = momentum_z[0:len(cycle_z.index)]
        
        
        yield_diff = yield_diff_data[yield_diff_data.columns[k]]-yield_diff_data['usgg2yr index']
        yield_diff_z = z_score(yield_diff,time_line)
        yield_diff_z = yield_diff_z[0:len(cycle_z.index)]

        

        df_z = pd.DataFrame({"cycle":cycle_z.values,"momentum":momentum_z.values,"yield_diff":yield_diff_z.values},index=cycle_z.index)
        df_z['basic criteria'] = df_z.sum(axis=1)

        start_date_id = df_z['basic criteria'][df_z['basic criteria'] > criteria_line].index
        if start_date_id[-1] == df_z.index[-1]:
            start_date_id = start_date_id[0:-1]
    
        end_date_id = []
        for j in range(0,len(start_date_id)):
            end_date_id = end_date_id + [df_z[df_z.index>start_date_id[j]].index[0]]



        ## plot

        plt.figure(figsize=(7,9))
        plt.tight_layout()


        plt.subplot(5,1,1)
        # plt.title(i+'USD')
        plt.subplots_adjust(top=0.99,bottom=0.01)
        plt.plot(momentum_data[i][8+time_line-2:],'r')

        for j in range(0,len(start_date_id)):
            plt.axvspan(start_date_id[j],end_date_id[j],color="grey",alpha=0.3)
        plt.ylabel(i+"USD")


        plt.subplot(5,1,2)
        plt.plot(df_z['basic criteria'])
        plt.axhline(y=criteria_line,color="grey",linestyle=":")
        
        plt.ylabel("Basic Criteria")

        plt.subplot(5,1,3)
        plt.plot(df_z['cycle'])
        plt.ylabel("Cycle")

        plt.subplot(5,1,4)
        plt.plot(df_z['momentum'])
        plt.ylabel("Momentum")

        plt.subplot(5,1,5)
        plt.plot(df_z['yield_diff'])
        plt.ylabel("ST Yield Diff")
        

        
        file_name = './images/'+i+' basic_criteria.png'
        plt.savefig(file_name, dpi=70)




# def real_yield_differentials(api_key=fred_api_key):
#     fred = Fred(api_key)
#     rates_tickers = ['IRLTLT01SEM156N','IRLTLT01AUM156N','IRLTLT01NOM156N','IRLTLT01CAM156N','IRLTLT01EZM156N',
#             'IRLTLT01GBM156N','IRLTLT01JPM156N','IRLTLT01NZM156N','IRLTLT01CHM156N']
#     cpi_tickers = ['SWECPIALLMINMEI','AUSCPIALLQINMEI','NORCPIALLMINMEI','CPALCY01CAM661N','CP0000EZ19M086NEST',
#             'GBRCPIALLMINMEI','JPNCPIALLMINMEI','NZLCPIALLQINMEI','CHECPIALLMINMEI']

#     rates_names = ['SW','AU','NO','CA','EU','GB','JP','NZ','CF']

#     rates_data_df = pd.read_csv('./data/10yearyield.csv',index_col='Unnamed: 0',parse_dates=True)
    

#     df_rates = pd.DataFrame(fred.get_series(rates_tickers[0])['1994-01-01':])
    
#     if rates_data_df.index[-1] != df_rates.index[-1]:
#         for i in rates_tickers[1:]:
#             df_rates = df_rates.merge(pd.DataFrame(fred.get_series(i)['1994-01-01':]),left_index=True, right_index=True, how='outer')
        
#         df_rates.to_csv('./data/10yearyield.csv')
#         rates_data_df = df_rates
    
#     rates_data_df.columns = rates_names

#     df_cpi = pd.DataFrame(fred.get_series(cpi_tickers[0])['1993-01-01':])

#     for i in cpi_tickers[1:]:
#         df_cpi = df_cpi.merge(pd.DataFrame(fred.get_series(i)['1993-01-01':]),left_index=True, right_index=True,how='outer')
        
    
#     print(df_cpi)


        
if __name__ == "__main__":
    twin_deficit()
    basic_model()
    # real_yield_differentials()