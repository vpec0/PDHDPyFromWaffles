



def run() :
    import sys
    d = ReadData(sys.argv[1])
    print(f'{d = }')



def ReadData(fname) :
    from gzip import open as gzopen
    from pickle import load as pklload
    with gzopen(fname, 'rb') as f :
        data = []
        while True :
            try:
                d = pklload(f)
                data.append(d)
            except EOFError :
                break
    return data




def SaveToGZ(fname, data) :
    with gzip.open(fname, 'wb', compresslevel=3) as outf :
        dump(data, outf)


def OpenGZ(fname) :
    return gzip.open(fname, 'wb', compresslevel=3)


if __name__ == '__main__' :
    run()
