import os


if os.path.exists('601318.SH_5min.csv'):
    data = pd.read_csv('601318.SH_5min.csv', index_col=0)
    print('Read sussessful')
#     print('head: \n', data.head())
#     print('tail: \n', data.tail())
else:
    print('File not exist')