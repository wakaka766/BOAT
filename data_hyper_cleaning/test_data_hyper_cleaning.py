import os
import time
import shutil

t0 = time.strftime("%Y_%m_%d_%H_%M_%S")
args = 'hyper_cleaning\method_test'
dynamic_methodlist = (["NGD"], ["DI", "NGD"], ["GDA", "NGD", "DI"], ["DI", "NGD", "GDA"])
dynamic_method_dm = (["NGD", "DM"], ["NGD", "DM", "GDA"])
hyper_methodlist = (
["CG"], ["CG", "PTT"], ["RAD"], ["RAD", "PTT"], ["RAD", "RGT"], ["PTT", "RAD", "RGT"], ["FD"], ["FD", "PTT"], ["NS"],
["NS", "PTT"], ["IGA"])
hyper_method_dm = (["RAD"], ["CG"])
fogm_method = (["VSM"], ["VFM"], ["MESM"],["PGDM"])
# m='Darts_W_RHG'
folder = r'C:\Users\ASUS\Documents\GitHub\BOAT\data_hyper_cleaning'
folder = os.path.join(folder, args, t0)
print(folder)
if not os.path.exists(folder):
    os.makedirs(folder)
batfolder = os.path.join(folder, 'set.bat')
ganfolder = os.path.join(folder, 'data_hyper_cleaning.py')
shutil.copyfile('data_hyper_cleaning.py', ganfolder)
utilfolder = os.path.join(folder, 'util_file.py')
shutil.copyfile('util_file.py', utilfolder)
with open(batfolder, 'w') as f:
    k = 0
    for dynamic_method in dynamic_methodlist:
        for hyper_method in hyper_methodlist:
            k += 1
            print("Comb.{}:".format(k))
            print('dynamic_method:', dynamic_method, ' hyper_method:', hyper_method)
            f.write('python data_hyper_cleaning.py --dynamic_method {} --hyper_method {} \n'.format(
                ','.join([dynamic for dynamic in dynamic_method]), ','.join([hyper for hyper in hyper_method])))

    for dynamic_method in dynamic_method_dm:
        for hyper_method in hyper_method_dm:
            k += 1
            print("Comb.{}:".format(k))
            print('dynamic_method:', dynamic_method, ' hyper_method:', hyper_method)
            f.write('python data_hyper_cleaning.py --dynamic_method {} --hyper_method {} \n'.format(','.join(
                [dynamic for dynamic in dynamic_method]), ','.join([hyper for hyper in hyper_method])))
    for hyper_method in fogm_method:
        k += 1
        print("Comb.{}:".format(k))
        print('hyper_method:', hyper_method)
        f.write('python data_hyper_cleaning.py --fo_gm {} \n'.format(hyper_method[0]))

os.chdir(folder)
os.system(batfolder)
