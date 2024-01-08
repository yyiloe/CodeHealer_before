import pickle
import os
path_statements='/home/dancher/Programs/Projects/trans2/TRANSFER-master/fault_localization/binary_classification/d4j_data/statements.pkl'
path_faulty_statements='/home/dancher/Programs/Projects/trans2/TRANSFER-master/fault_localization/ranking_task/faulty_statement_set.pkl'
version_path='/home/dancher/Programs/Projects/trans2/TRANSFER-master/program_repair/automatic_fix/versions.txt'
out_put='/home/dancher/Programs/Projects/trans2/TRANSFER-master/program_repair/automatic_fix/SuspiciousCodePositions'
f=open(path_statements,'rb')
statements=pickle.load(f)

f=open(path_faulty_statements,'rb')
faulty_statements=pickle.load(f)

with open(version_path, "r") as file:
	versions = file.readlines()
versions = [version.strip() for version in versions]

for version in versions:
    out_put_dir=os.path.join(out_put,version)
    if not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)
    
    with open(os.path.join(out_put_dir,'ranking.txt'),'w') as file:
        statements_list=faulty_statements[version]
        for s in statements_list:
            for s_ in s:
                if s_ in statements[version]:
                    file.write(s_+"\n")
            