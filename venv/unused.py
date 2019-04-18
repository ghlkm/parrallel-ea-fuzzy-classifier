def cal_crow(correctlist, rulelist, layer):
    # list都是从 大 到 小 排好序才进来的
    for i in layer:
        if correctlist.index(i[2]) == 0 or correctlist.index(i[2]) == len(correctlist) - 1 or rulelist.index(
                i[3]) == 0 or rulelist.index(i[3]) == len(rulelist) - 1:
            i[4] = sys.maxsize
        else:
            i[4] = correctlist[correctlist.index(i[2]) - 1] - correctlist[correctlist.index(i[2]) + 1] + \
                   rulelist[rulelist.index(i[3]) - 1] - rulelist[rulelist.index(i[3]) + 1]
            i[4] = i[4] / len(layer)