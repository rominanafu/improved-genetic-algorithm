from src.IGA_IGAm import solution
from glob import glob
import pandas as pd

tests = ['C1', 'C2', 'R1', 'R2', 'RC1', 'RC2']
vcap = [200, 700, 200, 1000, 200, 1000]
for i in range(len(tests)):
    t = tests[i]
    cap = vcap[i]
    ans = input(f"\nRun Solomon {t} tests? [y/n]: ")
    if (ans.lower() == 'y'):
        files_path = glob(f'data/solomon_dataset/{t}/*.csv')
        for j in range(len(files_path)):
            path = files_path[j]
            instance = f'{t}{j+1:02d}'
            ans = input(f"\nRun {instance} instance? [y/n]: ")
            if (ans.lower() == 'y'):
                solution(path, cap, instance)
