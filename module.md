# data cleaning
- store target , inner join before mp on store_id,STOREID
- s=

# ibd evaluation
- anova计算ibd evaluation
    - province
    - shoptype
    - 对dwh，chain
    - min store count，30
- fuller
  - CAT X MBD
  - evaluation:
  
``` If a correction was proposed its impact is assessed by looking at the effect on expanded
figures for the different participating IBD’s.
LOOP over IBD’s and compute for each IBD:
winsorized projected total value + relative standard error
winsorized projected total number of units + relse
projected price (projected total value / projected total number of units) + relse
projected ACV weighted selling distribution
projected ACV weighted promo distribution
number of stores selling
total number of stores in sample
Check whether the corrections above are « significant » by looping over all stores with a 
correction and for each of these stores we loop over the relevant IBD’s. If they are significant
for 1 or more IBD’s then it will be applied:
ValDiff=xf*(original_val-corr_val) / proj_total_val_IBD
VolDiff=xf*(original_vol-corr_vol) / proj_total_vol_IBD
PDiff=(1+ValDiff)/(1+VolDiff)-1
Compute extra tolerance if item has promotional activity
TolVal=z_value_val x relse_val_Ibd x Extra_Tol
TolPrice=z_value_price x relse_price_Ibd x Extra_Tol
If abs(ValDiff > TolVal) then SIGNIFICANT
If abs(Pdiff > TolPrice) then SIGNIFICANT
```