      strategy  accuracy  llm_judge_score  parse_success  cost_usd  latency_s
      
0    zero_shot         0                1          False  0.000033   2.309344
1    zero_shot         0                1          False  0.000031   2.836648
2    zero_shot         0                1          False  0.000033   2.149019
3    zero_shot         0                1          False  0.000033   2.257954
4    zero_shot         0                1          False  0.000035   2.073362
5    zero_shot         0                1          False  0.000035   2.304874
6    zero_shot         0                1          False  0.000032   3.401141
7    zero_shot         1                3           True  0.000032   2.319531
8    zero_shot         0                1          False  0.000034   2.159276
9    zero_shot         0                1          False  0.000040   2.046529
10    few_shot         3                4           True  0.000048   2.171048
11    few_shot         2                4           True  0.000048   2.126744
12    few_shot         3                4           True  0.000050   2.164516
13    few_shot         3                4           True  0.000049   2.141755
14    few_shot         3                4           True  0.000049   2.321651
15    few_shot         3                4           True  0.000050   2.364356
16    few_shot         3                4           True  0.000048   2.032093
17    few_shot         2                4           True  0.000050   2.369185
18    few_shot         2                3           True  0.000049   2.029492
19    few_shot         3                4           True  0.000051   2.184011
20  structured         3                4           True  0.000037   2.027314
21  structured         3                4           True  0.000037   2.033538
22  structured         3                4           True  0.000039   2.382904
23  structured         3                4           True  0.000038   2.027217
24  structured         3                4           True  0.000038   2.413876
25  structured         3                4           True  0.000039   2.023030
26  structured         3                4           True  0.000037   2.117614
27  structured         2                4           True  0.000039   2.078345
28  structured         3                4           True  0.000038   2.023546
29  structured         3                4           True  0.000040   2.120546
30         cot         3                4           True  0.000040   2.119315
31         cot         3                4           True  0.000040   2.545515
32         cot         3                4           True  0.000042   2.117862
33         cot         3                4           True  0.000041   2.113659
34         cot         3                4           True  0.000041   2.028692
35         cot         3                4           True  0.000042   2.114071
36         cot         3                4           True  0.000040   2.021881
37         cot         2                4           True  0.000042   2.044288
38         cot         3                4           True  0.000041   2.038396
39         cot         3                4           True  0.000043   2.087794


Strategy     Accuracy (mean of 3)  Parse rate  Judge score  Total cost ($)  Latency p50 (s)
                                                                                  
cot                          2.9         1.0          4.0        0.000411            2.101
few_shot                     2.7         1.0          3.9        0.000492            2.168
structured                   2.9         1.0          4.0        0.000381            2.056
zero_shot                    0.1         0.1          1.2        0.000339            2.281