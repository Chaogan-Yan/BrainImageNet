import os
# print (os.path.dirname(__file__))
# print (os.path.abspath(__file__))
# print (os.path.abspath(os.path.dirname(__file__)))
# print (os.path.dirname(os.path.abspath(__file__)))

source_path = os.path.abspath('./T1/UNZIP')

                
def generates(path):
    if os.path.exists(path):
        sub=[]
        file_write_obj = open(path+'/subid.txt', 'w')
        for root, dirs, files in os.walk(path):
            for file in files:
                src_file = os.path.join(root, file)
                if src_file[len(src_file)-4:len(src_file)]=='.nii':
                    sub.append(src_file)
                    file_write_obj.write(src_file+' \n')
                elif src_file[len(src_file)-7:len(src_file)]=='.nii.gz':
                    sub.append(src_file)
                    file_write_obj.write(src_file+' \n')
generates(source_path)
