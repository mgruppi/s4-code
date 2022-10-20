import sqlite3
import os
import re


if __name__ == '__main__':
    path = '/data/nela/'
    files = ['NELA-GT-2018/nela-gt-2018.db', 'NELA-GT-2019/nela-gt-2019.db',
             'NELA-GT-2020/nela-gt-2020.db', 'NELA-GT-2021/nela-gt-2021.db']
    
    total_count = 0
    for f in files:
        con = sqlite3.connect(os.path.join(path, f))
        query = 'SELECT content FROM newsdata'
        result = con.execute(query)

        count = 0
        for r in result:
            tokens = re.split('\s+', r[0])
            count += len(tokens)
            
        print(f, count)
