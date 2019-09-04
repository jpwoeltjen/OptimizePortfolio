from ftplib import FTP
import pandas as pd
import hashlib
from datetime import datetime


def download(file_name='usa.txt'):
    """
    Downloads and saves as csv shorting rates of shortable stocks via
    Interactive Brokers. file_name='' prints the directory.
    """
    ftp = FTP('ftp3.interactivebrokers.com')
    ftp.login(user='shortstock', passwd=' ')

    if file_name == '':
        print(ftp.retrlines('LIST'))
        return

    hasher = hashlib.md5()

    with open('../data/'+file_name, 'wb') as file:
        ftp.retrbinary('RETR '+file_name, file.write)

    with open('../data/'+file_name+'.md5', 'wb') as file:
        ftp.retrbinary('RETR '+file_name+'.md5', file.write)
        ftp.quit()

    with open('../data/'+file_name) as file:
        text = file.read()
        hasher.update(text.encode('utf-8'))
        text = text.splitlines(True)
        assert text[0][:4] == '#BOF'
        assert text[-1][:4] == '#EOF'
        last_update = text[0][5:-1]
        text = text[1:-1]

    with open('../data/'+file_name, "w") as file:
        for line in text:
            file.write(line)

    with open('../data/'+file_name+'.md5') as file:
        hash_value = file.read()
    assert hash_value == hasher.hexdigest()

    data = pd.read_csv('../data/'+file_name, sep="|", header=0).iloc[:, :8]
    data.to_csv('../data/'+file_name[:-3]+'csv', header=True)
    last_update = datetime.strptime(last_update, '%Y.%m.%d|%H:%M:%S')
    print('Successfully downloaded shorting information | last_update:',
          last_update)


if __name__ == "__main__":
    print('Type in file name (e.g. usa.txt):')
    f = input()
    if f == '':
        print('No valid input')
    else:
        print('... downloading', f)
    download(f)
