import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

data_path  = './data'
df = pd.read_csv(data_path+"/before_mp/O2O_itemcoding_output_20261408_newline_fixed.csv.gz",encoding='gb18030')
df_store_target = pd.read_csv(data_path+"/store_target/target_shop_qc_20261408.csv",encoding='gb18030')


df = pd.merge(df,
              df_store_target[['STOREID','SHOPID','STORENAME', 'PROVINCE', 'CITY', 'SHOPTYPE','PLATFORMNAME']],
              left_on = "store_id",
              right_on="STOREID",
              how='inner')
df_check=df.groupby(['store_id',"SHOPTYPE","PROVINCE","PLATFORMNAME"])['sales_value'].sum().reset_index()
df_check['cell'] = df_check['SHOPTYPE']+"_"+df_check['PROVINCE']+df_check['PLATFORMNAME']
# formula = 'sales_value ~ C(SHOPTYPE) + C(PROVINCE) +C(PLATFORMNAME)'
formula = "sales_value~C(cell)"
model = ols(formula, df_check[df_check['SHOPTYPE']!="dwh"]).fit()
aov_table = anova_lm(model, typ=2).round(3).fillna('')
SS_model = aov_table['sum_sq'].iloc[:-1].sum()  
SS_residual = aov_table['sum_sq'].iloc[-1]  

# Total sum of squares
SS_total = SS_model + SS_residual

# Explained variance
explained_variance = SS_model / SS_total
print(f'Explained Variance: {explained_variance:.3f}')
df_check_mt = df_check[(df_check['PLATFORMNAME']=="MEITUAN")&
                       (df_check['SHOPTYPE']!='dwh')]
tukey = pairwise_tukeyhsd(endog=df_check_mt['sales_value'], groups=df_check_mt['cell'], alpha=0.05)
print(tukey)


history_data_path = r'C:\Users\feyu6001\OneDrive - NIQ\O2O Implementation\new'
# df_report = pd.read_csv(history_data_path+"/O2O_final_history_20261408.csv.gz",encoding='utf8')
df_dup = df_report[df_report.duplicated(['store_id','prod_id'],keep=False)]
df_dup=df_dup.drop_duplicates()

df_report[df_report.store_id.isin(df_dup.store_id.unique())]
# df_dup_store = pd.read_excel(data_path+"/target_243.xlsx")
# df_dup_store = df_dup_store[df_dup_store.STATUS==2]

# df_report[df_report.store_id.isin(df_dup_store['STOREID'].unique())].groupby("mp")['store_id'].nunique()