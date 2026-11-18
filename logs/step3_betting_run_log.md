# Step 3 - Betting simulation (flat £1 stakes, test seasons)

Matches: 760. Mean Bet365 margin on test: 0.050.
ROI = profit / staked. CI = 95% bootstrap over matches. Draw bets in bets_D.

## Odds: b365 · season: 2024/25
```
    model feature_set            strategy  bets  bets_D  win_rate  avg_odds  profit     roi  roi_ci_low  roi_ci_high
Benchmark           -         always_home   380       0    0.4079    2.7699  -67.40 -0.1774     -0.2959      -0.0544
Benchmark           - bookmaker_favourite   380       0    0.5421    1.8576  -14.45 -0.0380     -0.1352       0.0593
       LR  non_market           favourite   380       0    0.5316    1.9130  -16.40 -0.0432     -0.1432       0.0591
       LR      market           favourite   380       0    0.5395    1.8581  -17.13 -0.0451     -0.1420       0.0549
       RF    combined           favourite   380       0    0.5368    1.8686  -17.60 -0.0463     -0.1431       0.0502
       LR    combined           favourite   380       0    0.5342    1.8698  -20.39 -0.0537     -0.1497       0.0506
       RF      market           favourite   380       0    0.5316    1.8638  -24.10 -0.0634     -0.1592       0.0290
       RF  non_market           favourite   380       1    0.5263    1.8879  -26.81 -0.0706     -0.1662       0.0293
      KNN      market           favourite   380      13    0.5237    1.8996  -29.64 -0.0780     -0.1728       0.0222
      KNN    combined           favourite   380       7    0.5079    1.9012  -44.43 -0.1169     -0.2093      -0.0259
      KNN  non_market           favourite   380       0    0.5000    1.9227  -50.01 -0.1316     -0.2259      -0.0373
      KNN      market               value   347     107    0.4294    3.1895   -0.54 -0.0016     -0.1356       0.1466
       LR  non_market               value   325      74    0.3415    3.5339  -11.59 -0.0357     -0.1970       0.1469
       RF    combined               value   329      74    0.4164    3.2209  -19.78 -0.0601     -0.1999       0.0941
       LR    combined               value   266      52    0.4060    3.2285  -28.13 -0.1058     -0.2494       0.0382
       RF  non_market               value   348     100    0.2874    4.2429  -47.74 -0.1372     -0.2926       0.0504
      KNN    combined               value   363     120    0.3058    4.1044  -50.61 -0.1394     -0.2892       0.0271
       LR      market               value   219     108    0.3242    3.7594  -44.79 -0.2045     -0.3725      -0.0309
       RF      market               value   313      94    0.3866    3.2261  -67.40 -0.2153     -0.3401      -0.0932
      KNN  non_market               value   361     110    0.2133    4.7491  -89.89 -0.2490     -0.4126      -0.0736
```

## Odds: b365 · season: 2025/26
```
    model feature_set            strategy  bets  bets_D  win_rate  avg_odds  profit     roi  roi_ci_low  roi_ci_high
Benchmark           -         always_home   380       0    0.4263    2.6478  -35.13 -0.0924     -0.2120       0.0269
Benchmark           - bookmaker_favourite   380       0    0.4895    1.9012  -41.68 -0.1097     -0.2046      -0.0153
      KNN    combined           favourite   380       7    0.4947    1.9916  -23.23 -0.0611     -0.1655       0.0422
       LR      market           favourite   380       0    0.4947    1.9024  -36.42 -0.0958     -0.1934      -0.0004
       LR  non_market           favourite   380       0    0.4868    1.9622  -36.78 -0.0968     -0.1968       0.0017
      KNN      market           favourite   380      12    0.4842    1.9423  -40.38 -0.1063     -0.2069      -0.0128
       RF  non_market           favourite   380       0    0.4789    1.9627  -44.97 -0.1183     -0.2165      -0.0191
       LR    combined           favourite   380       0    0.4842    1.9144  -45.50 -0.1197     -0.2130      -0.0230
      KNN  non_market           favourite   380       0    0.4684    2.0131  -48.29 -0.1271     -0.2286      -0.0276
       RF    combined           favourite   380       0    0.4789    1.9078  -50.85 -0.1338     -0.2277      -0.0403
       RF      market           favourite   380       0    0.4763    1.9052  -54.25 -0.1428     -0.2371      -0.0495
      KNN    combined               value   349     100    0.3352    3.5499   -2.39 -0.0068     -0.1786       0.1732
      KNN      market               value   332      96    0.3946    3.0012  -14.01 -0.0422     -0.1922       0.1131
       RF      market               value   261      78    0.3985    2.9310  -23.95 -0.0918     -0.2479       0.0652
      KNN  non_market               value   358      75    0.2514    4.2782  -43.87 -0.1225     -0.2947       0.0617
       RF    combined               value   265      56    0.3623    2.9606  -35.86 -0.1353     -0.2945       0.0279
       LR  non_market               value   329      25    0.3070    3.5123  -48.67 -0.1479     -0.3005       0.0109
       LR      market               value    81      47    0.2716    4.6554  -12.41 -0.1532     -0.4887       0.1807
       RF  non_market               value   335      51    0.2657    3.9190  -60.21 -0.1797     -0.3329      -0.0227
       LR    combined               value   240      24    0.3458    2.9680  -54.18 -0.2258     -0.3698      -0.0827
```

## Odds: b365 · season: all
```
    model feature_set            strategy  bets  bets_D  win_rate  avg_odds  profit     roi  roi_ci_low  roi_ci_high
Benchmark           -         always_home   760       0    0.4171    2.7088 -102.53 -0.1349     -0.2161      -0.0497
Benchmark           - bookmaker_favourite   760       0    0.5158    1.8794  -56.13 -0.0739     -0.1404      -0.0065
       LR  non_market           favourite   760       0    0.5092    1.9376  -53.18 -0.0700     -0.1372      -0.0007
       LR      market           favourite   760       0    0.5171    1.8802  -53.55 -0.0705     -0.1371      -0.0026
       LR    combined           favourite   760       0    0.5092    1.8921  -65.89 -0.0867     -0.1550      -0.0194
      KNN    combined           favourite   760      14    0.5013    1.9464  -67.66 -0.0890     -0.1528      -0.0181
       RF    combined           favourite   760       0    0.5079    1.8882  -68.45 -0.0901     -0.1576      -0.0201
      KNN      market           favourite   760      25    0.5039    1.9209  -70.02 -0.0921     -0.1604      -0.0228
       RF  non_market           favourite   760       1    0.5026    1.9253  -71.78 -0.0944     -0.1596      -0.0249
       RF      market           favourite   760       0    0.5039    1.8845  -78.35 -0.1031     -0.1664      -0.0329
      KNN  non_market           favourite   760       0    0.4842    1.9679  -98.30 -0.1293     -0.1958      -0.0595
      KNN      market               value   679     203    0.4124    3.0974  -14.55 -0.0214     -0.1257       0.0862
      KNN    combined               value   712     220    0.3202    3.8326  -53.00 -0.0744     -0.1893       0.0443
       LR  non_market               value   654      99    0.3242    3.5230  -60.26 -0.0921     -0.2062       0.0273
       RF    combined               value   594     130    0.3923    3.1048  -55.64 -0.0937     -0.1996       0.0162
       RF  non_market               value   683     151    0.2767    4.0841 -107.95 -0.1581     -0.2790      -0.0362
       RF      market               value   574     172    0.3920    3.0919  -91.35 -0.1591     -0.2548      -0.0542
       LR    combined               value   506      76    0.3775    3.1049  -82.31 -0.1627     -0.2676      -0.0559
      KNN  non_market               value   719     185    0.2323    4.5146 -133.76 -0.1860     -0.3029      -0.0721
       LR      market               value   300     155    0.3100    4.0013  -57.20 -0.1907     -0.3409      -0.0290
```

## Odds: max · season: 2024/25
```
    model feature_set            strategy  bets  bets_D  win_rate  avg_odds  profit     roi  roi_ci_low  roi_ci_high
Benchmark           -         always_home   380       0    0.4079    2.9015  -58.15 -0.1530     -0.2762      -0.0231
Benchmark           - bookmaker_favourite   380       0    0.5421    1.9039   -5.63 -0.0148     -0.1144       0.0848
       LR  non_market           favourite   380       0    0.5316    1.9629   -7.31 -0.0192     -0.1223       0.0850
       LR      market           favourite   380       0    0.5395    1.9043   -8.41 -0.0221     -0.1215       0.0805
       RF    combined           favourite   380       0    0.5368    1.9160   -8.68 -0.0228     -0.1224       0.0759
       LR    combined           favourite   380       0    0.5342    1.9174  -11.57 -0.0304     -0.1291       0.0762
       RF      market           favourite   380       0    0.5316    1.9106  -15.43 -0.0406     -0.1392       0.0542
       RF  non_market           favourite   380       1    0.5263    1.9364  -18.22 -0.0479     -0.1462       0.0542
      KNN      market           favourite   380      13    0.5237    1.9501  -21.13 -0.0556     -0.1527       0.0468
      KNN    combined           favourite   380       7    0.5079    1.9508  -36.14 -0.0951     -0.1896      -0.0014
      KNN  non_market           favourite   380       0    0.5000    1.9739  -42.02 -0.1106     -0.2067      -0.0141
       LR  non_market               value   380      96    0.3526    3.8063   21.77  0.0573     -0.1044       0.2397
      KNN      market               value   377     120    0.3926    3.7445  -15.21 -0.0403     -0.1778       0.1075
       RF    combined               value   378      92    0.3915    3.6643  -17.24 -0.0456     -0.1913       0.1092
       RF      market               value   376     112    0.4043    3.6558  -17.67 -0.0470     -0.1938       0.1001
       RF  non_market               value   377     107    0.2891    4.7321  -26.68 -0.0708     -0.2413       0.1134
       LR    combined               value   375     102    0.3627    3.9922  -32.30 -0.0861     -0.2295       0.0563
      KNN    combined               value   379     128    0.2982    4.4716  -44.17 -0.1165     -0.2738       0.0515
      KNN  non_market               value   380     120    0.2211    5.0591  -70.72 -0.1861     -0.3559      -0.0033
       LR      market               value   361     181    0.2992    4.3414  -70.45 -0.1952     -0.3366      -0.0458
```

## Odds: max · season: 2025/26
```
    model feature_set            strategy  bets  bets_D  win_rate  avg_odds  profit     roi  roi_ci_low  roi_ci_high
Benchmark           -         always_home   380       0    0.4263    2.7703  -25.72 -0.0677     -0.1911       0.0574
Benchmark           - bookmaker_favourite   380       0    0.4895    1.9462  -33.94 -0.0893     -0.1862       0.0061
      KNN    combined           favourite   380       7    0.4947    2.0415  -14.85 -0.0391     -0.1460       0.0665
       LR  non_market           favourite   380       0    0.4868    2.0120  -28.48 -0.0749     -0.1763       0.0258
       LR      market           favourite   380       0    0.4947    1.9483  -28.51 -0.0750     -0.1749       0.0226
      KNN      market           favourite   380      12    0.4842    1.9905  -32.42 -0.0853     -0.1884       0.0109
       RF  non_market           favourite   380       0    0.4789    2.0124  -36.96 -0.0973     -0.1975       0.0034
       LR    combined           favourite   380       0    0.4842    1.9612  -37.55 -0.0988     -0.1952       0.0000
      KNN  non_market           favourite   380       0    0.4684    2.0669  -40.57 -0.1068     -0.2111      -0.0044
       RF    combined           favourite   380       0    0.4789    1.9547  -43.27 -0.1139     -0.2100      -0.0189
       RF      market           favourite   380       0    0.4763    1.9528  -46.79 -0.1231     -0.2193      -0.0277
      KNN      market               value   370     109    0.3676    3.5125  -11.13 -0.0301     -0.1790       0.1321
      KNN    combined               value   376      98    0.3032    3.9130  -26.60 -0.0707     -0.2296       0.1020
       LR      market               value   312     125    0.3429    3.8021  -35.27 -0.1130     -0.2709       0.0419
      KNN  non_market               value   375      77    0.2373    4.6445  -45.83 -0.1222     -0.2944       0.0687
       RF      market               value   352      93    0.3750    3.2966  -43.38 -0.1232     -0.2635       0.0160
       LR  non_market               value   370      21    0.3000    3.8598  -59.34 -0.1604     -0.3057      -0.0107
       RF  non_market               value   368      51    0.2609    4.2671  -63.70 -0.1731     -0.3280      -0.0086
       RF    combined               value   365      74    0.3342    3.5717  -68.78 -0.1884     -0.3254      -0.0523
       LR    combined               value   345      51    0.3188    3.5235  -85.27 -0.2472     -0.3855      -0.1184
```

## Odds: max · season: all
```
    model feature_set            strategy  bets  bets_D  win_rate  avg_odds  profit     roi  roi_ci_low  roi_ci_high
Benchmark           -         always_home   760       0    0.4171    2.8359  -83.87 -0.1104     -0.1949      -0.0226
Benchmark           - bookmaker_favourite   760       0    0.5158    1.9251  -39.57 -0.0521     -0.1201       0.0171
       LR  non_market           favourite   760       0    0.5092    1.9875  -35.79 -0.0471     -0.1159       0.0239
       LR      market           favourite   760       0    0.5171    1.9263  -36.92 -0.0486     -0.1168       0.0212
       LR    combined           favourite   760       0    0.5092    1.9393  -49.12 -0.0646     -0.1341       0.0049
      KNN    combined           favourite   760      14    0.5013    1.9962  -50.99 -0.0671     -0.1323       0.0050
       RF    combined           favourite   760       0    0.5079    1.9353  -51.95 -0.0684     -0.1372       0.0036
      KNN      market           favourite   760      25    0.5039    1.9703  -53.55 -0.0705     -0.1402       0.0011
       RF  non_market           favourite   760       1    0.5026    1.9744  -55.18 -0.0726     -0.1388      -0.0012
       RF      market           favourite   760       0    0.5039    1.9317  -62.22 -0.0819     -0.1468      -0.0105
      KNN  non_market           favourite   760       0    0.4842    2.0204  -82.59 -0.1087     -0.1770      -0.0373
      KNN      market               value   747     229    0.3802    3.6296  -26.34 -0.0353     -0.1385       0.0746
       LR  non_market               value   750     117    0.3267    3.8327  -37.57 -0.0501     -0.1599       0.0620
       RF      market               value   728     205    0.3901    3.4821  -61.05 -0.0839     -0.1860       0.0252
      KNN    combined               value   755     226    0.3007    4.1934  -70.77 -0.0937     -0.2060       0.0216
       RF    combined               value   743     166    0.3634    3.6188  -86.02 -0.1158     -0.2139      -0.0082
       RF  non_market               value   745     158    0.2752    4.5025  -90.38 -0.1213     -0.2378      -0.0001
      KNN  non_market               value   755     197    0.2291    4.8532 -116.55 -0.1544     -0.2751      -0.0344
       LR      market               value   673     306    0.3195    4.0914 -105.72 -0.1571     -0.2604      -0.0451
       LR    combined               value   720     153    0.3417    3.7676 -117.57 -0.1633     -0.2564      -0.0589
```
