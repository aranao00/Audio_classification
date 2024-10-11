import os

# Loss 및 정확도 기록 용 코드입니다.

def prif(*args, file_name='/tf/nasw/trainlog.txt', endswith='\n'):
    string_data=' '.join(map(str, args)) + endswith 
    with open(file_name, 'a') as f:
        f.write(string_data)