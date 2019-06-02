d=['dataset','wine','Glass', 'Breast cancer', 'gesture', 'sat', 'avila']
i=['instance',178, 214, 699, 1743, 6435, 20867]
a=['attribute', 13, 9, 9, 32, 36, 10]
c=['label', 3, 6, 2, 5, 6, 12]
for index, value in enumerate(d):
    print('\\hline')
    print(value, '&', i[index] ,'&' , a[index],'&',c[index],'\\\\')
    print('\\hline')