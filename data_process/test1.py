#!/usr/bin/python3
import sys
import json

if __name__ == '__main__':
    for line in sys.stdin:
        print(line)
        #line = '{"a":[1,2,3],"b":[4,5,6]}'
        dict = json.loads(line)
        ls = []
        for v in dict.values():
            ls.insert(1, list(v))
        vector1 = tuple(ls[0])
        vector2 = tuple(ls[1])
        v = sum(p * q for p, q in zip(vector1, vector2))
        data = {'result': str(v)}
        print(json.dumps(data), end='\n')
        sys.stdout.flush()
