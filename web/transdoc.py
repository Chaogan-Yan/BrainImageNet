import os
import shutil
source_path = os.path.abspath('./DataUpload/2020_03_23_09_48_59/UNZIP')
out_dir='./DataUpload/2020_03_23_09_48_59/wc1' 
def trans(path,file_dir):
    if os.path.exists(path):                 
        for root, dirs, files in os.walk(path):
            for file in files:
                src_file = os.path.join(root, file)
                file_new=os.path.join(file_dir, file)
                if file[0:3]=='wc1':
                    shutil.copyfile(src_file,file_new)
                elif file[0:4]=='mwc1':
                    shutil.copyfile(src_file,file_new)                                         
trans(source_path,out_dir)