
import re
import sys

log = """
>> start: 0.000002
>> after gameWatcher: 0.000064
>> after 3: 0.001424
>> after 3.1: 0.851951
>> after 3.2: 0.851998
>> after 3.3: 0.852007
>> after 4: 0.852022
>> after 5: 2.257159
>> after 6: 2.257273
>> after 7: 4.878993
>> after 8: 4.879610
>> after 9: 4.881773
>> after 10: 6.666337
>> after 11: 7.986948
>> after 12: 7.986976
>> after LAST: 7.986989
>> start: 0.000003
>> after gameWatcher: 0.000064
>> after 3: 0.001349
>> after 3.1: 1.626849
>> after 3.2: 1.626911
>> after 3.3: 1.626919
>> after 4: 1.626940
>> after 5: 2.967036
>> after 6: 2.967164
>> after 7: 2.967185
>> after 8: 4.599206
>> after 9: 4.601447
>> after 10: 5.651764
>> after 11: 6.362270
>> after 12: 6.362298
>> after LAST: 6.362309
>> start: 0.000003
>> after gameWatcher: 0.000437
>> after 3: 0.001908
>> after 3.1: 1.525161
>> after 3.2: 1.525241
>> after 3.3: 1.525264
>> after 4: 1.525303
>> after 5: 2.843150
>> after 6: 2.843372
>> after 7: 3.719180
>> after 8: 4.476155
>> after 9: 4.481553
>> after 10: 6.002796
>> after 11: 7.317315
>> after 12: 7.317341
>> after LAST: 7.317355
>> start: 0.000006
>> after gameWatcher: 0.000166
>> after 3: 0.001490
>> after 3.1: 0.877304
>> after 3.2: 0.877366
>> after 3.3: 0.877377
>> after 4: 0.877390
>> after 5: 2.261278
>> after 6: 2.261430
>> after 7: 2.261442
>> after 8: 3.886379
>> after 9: 3.890414
>> after 10: 5.408188
>> after 11: 6.724979
>> after 12: 6.725006
>> after LAST: 6.725019
>> start: 0.000004
>> after gameWatcher: 0.000081
>> after 3: 0.001397
>> after 3.1: 1.583958
>> after 3.2: 1.584012
>> after 3.3: 1.584023
>> after 4: 1.584048
>> after 5: 2.902553
>> after 6: 2.902674
>> after 7: 4.533823
>> after 8: 4.534497
>> after 9: 4.536663
>> after 10: 6.343672
>> after 11: 7.658684
>> after 12: 7.658713
>> after LAST: 7.658728
>> start: 0.000002
>> after gameWatcher: 0.000063
>> after 3: 0.001368
>> after 3.1: 1.587164
>> after 3.2: 1.587218
>> after 3.3: 1.587224
>> after 4: 1.587247
>> after 5: 2.903244
>> after 6: 4.218564
>> after 7: 4.218591
>> after 8: 5.535130
>> after 9: 5.537293
>> after 10: 7.311713
>> after 11: 8.624247
>> after 12: 8.624276
>> after LAST: 8.624290
>> start: 0.000002
>> after gameWatcher: 0.000091
>> after 3: 0.001406
>> after 3.1: 1.518909
>> after 3.2: 1.518966
>> after 3.3: 1.518976
>> after 4: 1.518998
>> after 5: 2.820878
>> after 6: 2.820998
>> after 7: 4.403703
>> after 8: 4.404385
>> after 9: 4.406644
>> after 10: 6.184722
>> after 11: 7.494985
>> after 12: 7.495013
>> after LAST: 7.495025
>> start: 0.000003
>> after gameWatcher: 0.000477
>> after 3: 0.002967
>> after 3.1: 1.501917
>> after 3.2: 1.501978
>> after 3.3: 1.501989
>> after 4: 1.502005
>> after 5: 2.826399
>> after 6: 3.492272
>> after 7: 3.492302
>> after 8: 4.153593
>> after 9: 4.801555
>> after 10: 6.567445
>> after 11: 7.873445
>> after 12: 7.873471
>> after LAST: 7.873484
>> start: 0.000001
>> after gameWatcher: 0.000053
>> after 3: 0.001338
>> after 3.1: 1.574027
>> after 3.2: 1.574083
>> after 3.3: 1.574094
>> after 4: 1.574118
>> after 5: 2.875473
>> after 6: 2.875592
>> after 7: 4.486992
>> after 8: 4.487701
>> after 9: 4.489929
>> after 10: 6.260913
>> after 11: 7.565757
>> after 12: 7.565787
>> after LAST: 7.565804
>> start: 0.000005
>> after gameWatcher: 0.000192
>> after 3: 0.001503
>> after 3.1: 1.541443
>> after 3.2: 1.541494
>> after 3.3: 1.541505
>> after 4: 1.541526
>> after 5: 2.192585
>> after 6: 2.192690
>> after 7: 2.192698
>> after 8: 3.773147
>> after 9: 3.775535
>> after 10: 5.276381
>> after 11: 5.932092
>> after 12: 5.932120
>> after LAST: 5.932130
>> start: 0.000003
>> after gameWatcher: 0.000098
>> after 3: 0.001402
>> after 3.1: 1.536122
>> after 3.2: 1.536198
>> after 3.3: 1.536205
>> after 4: 1.536221
>> after 5: 3.127680
>> after 6: 3.127822
>> after 7: 4.737101
>> after 8: 4.737685
>> after 9: 4.739981
>> after 10: 6.634417
>> after 11: 7.944436
>> after 12: 7.944464
>> after LAST: 7.944477
>> start: 0.000002
>> after gameWatcher: 0.000065
>> after 3: 0.001361
>> after 3.1: 1.611038
>> after 3.2: 1.611095
>> after 3.3: 1.611105
>> after 4: 1.611123
>> after 5: 2.907177
>> after 6: 2.907293
>> after 7: 2.907305
>> after 8: 4.486924
>> after 9: 4.489268
>> after 10: 6.346636
>> after 11: 7.652791
>> after 12: 7.652811
>> after LAST: 7.652824
>> start: 0.000002
>> after gameWatcher: 0.000063
>> after 3: 0.001347
>> after 3.1: 0.869581
>> after 3.2: 0.869637
>> after 3.3: 0.869649
>> after 4: 0.869663
>> after 5: 2.260075
>> after 6: 2.260197
>> after 7: 3.854209
>> after 8: 3.854920
>> after 9: 3.857248
>> after 10: 5.752414
>> after 11: 7.073569
>> after 12: 7.073597
>> after LAST: 7.073610
>> start: 0.000007
>> after gameWatcher: 0.000208
>> after 3: 0.001580
>> after 3.1: 1.538998
>> after 3.2: 1.539049
>> after 3.3: 1.539058
>> after 4: 1.539080
>> after 5: 2.843306
>> after 6: 2.843425
>> after 7: 2.843436
>> after 8: 4.434078
>> after 9: 4.436249
>> after 10: 6.207424
>> after 11: 6.867935
>> after 12: 6.867964
>> after LAST: 6.867974
>> start: 0.000003
>> after gameWatcher: 0.000099
>> after 3: 0.001376
>> after 3.1: 1.644211
>> after 3.2: 1.644280
>> after 3.3: 1.644299
>> after 4: 1.644330
>> after 5: 2.933627
>> after 6: 2.933744
>> after 7: 4.535944
>> after 8: 4.536326
>> after 9: 4.538540
>> after 10: 5.577916
>> after 11: 7.218220
>> after 12: 7.218249
>> after LAST: 7.218267
>> start: 0.000002
>> after gameWatcher: 0.000102
>> after 3: 0.001661
>> after 3.1: 1.668185
>> after 3.2: 1.668601
>> after 3.3: 1.668617
>> after 4: 1.668642
>> after 5: 3.366247
>> after 6: 3.366388
>> after 7: 3.366396
>> after 8: 4.997877
>> after 9: 6.319118
>> after 10: 7.163667
>> after 11: 7.821695
>> after 12: 7.821723
>> after LAST: 7.821734
"""

blocks = []
current_block = []
for line in log.strip().split('\n'):
    if line.startswith('>> start:'):
        if current_block:
            blocks.append(current_block)
        current_block = [line]
    else:
        current_block.append(line)
if current_block:
    blocks.append(current_block)

diffs = {}

for block in blocks:
    last_time = 0
    for line in block:
        match = re.search(r'>> (.*): ([\d.]+)', line)
        if match:
            label = match.group(1)
            time = float(match.group(2))
            if label != 'start':
                diff = time - last_time
                if label not in diffs:
                    diffs[label] = []
                diffs[label].append(diff)
            last_time = time

print(f"{'Label':<20} | {'Min (ms)':<10} | {'Max (ms)':<10} | {'Mean (ms)':<10}")
print("-" * 56)
for label in diffs:
    d = diffs[label]
    min_val = min(d) * 1000
    max_val = max(d) * 1000
    mean_val = (sum(d) / len(d)) * 1000
    print(f"{label:<20} | {min_val:10.2f} | {max_val:10.2f} | {mean_val:10.2f}")
