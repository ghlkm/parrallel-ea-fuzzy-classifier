#coding=utf-8
# 本题为考试多行输入输出规范示例，无需提交，不计分。
import sys
import math
if __name__ == "__main__":
    # 读取第一行的n
    l = sys.stdin.readline().strip()
    l=list(map(int, l.split()))
    m=int(l[0])
    n=int(l[1])
    line=[]
    for i in range(n):
        line.append(int(input()))
    line=sorted(line, reverse=True)
    tmp=math.floor(m/line[0])
    if tmp*line[0]==m:
        tmp-=1
    ans=tmp
    last=line[0]
    left=m-line[0]*tmp
    for i in line[1:]:
        over=min((last, left))
        tmp=math.floor(over/i)
        if tmp*i==over and i!=1:
            tmp-=1
        ans+=tmp
        left-=tmp*i
        last=i
        if left<=0:
            break
        # print(tmp, i)
    if last==1:
        print(ans)
    else:
        print(-1)