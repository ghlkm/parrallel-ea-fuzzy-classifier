class Mylist(list):
    def __str__(self):
        d=','
        return d.join(self)

def test1(s):
    print(Mylist(s))

def test2(s):
    d=','
    print(d.join(s))

gs=['a', 'b', 'c']
test1(gs)
test2(gs)
