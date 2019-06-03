

import sys, getopt
def read_arg():
    """
    f: migration fre
    n: migration num
    :return:
    """
    opts, args = getopt.getopt(sys.argv[1:], "f:n:")
    for op, val in opts:
        print(val)


if __name__ == '__main__':
    read_arg()