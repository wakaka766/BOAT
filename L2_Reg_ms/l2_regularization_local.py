# import os
# import time
# import shutil
#
# t0 = time.strftime("%Y_%m_%d_%H_%M_%S")
# args = 'l2_regularization\method_test'
# dynamic_methodlist = (["NGD"], ["DI", "NGD"], ["GDA", "NGD", "DI"], ["DI", "NGD", "GDA"])
# dynamic_method_dm = (["NGD", "DM"], ["NGD", "DM", "GDA"])
# hyper_methodlist = (
# ["CG"], ["CG", "PTT"], ["RAD"], ["RAD", "PTT"], ["RAD", "RGT"], ["PTT", "RAD", "RGT"], ["FD"], ["FD", "PTT"], ["NS"],
# ["NS", "PTT"], ["IGA"])
# hyper_method_dm = (["RAD"], ["CG"])
# fogm_method = (["VSM"], ["VFM"], ["MESM"],["PGDM"])
# # m='Darts_W_RHG'
# folder = 'C:/Users/ASUS/Documents/GitHub/BOAT/L2_Reg'
# folder = os.path.join(folder,args, t0)
# # folder = 'data_hyper_cleaning'
# # folder = os.path.join(args, t0)
# print(folder)
# if not os.path.exists(folder):
#     os.makedirs(folder)
# batfolder = os.path.join(folder, 'set.bat')
# ganfolder = os.path.join(folder, 'l2_regularization.py')
# shutil.copyfile('l2_regularization.py', ganfolder)
# # utilfolder = os.path.join(folder, 'util_file.py')
# # shutil.copyfile('util_file.py', utilfolder)
# with open(batfolder, 'w') as f:
#     k = 0
#     for dynamic_method in dynamic_methodlist:
#         for hyper_method in hyper_methodlist:
#             k += 1
#             print("Comb.{}:".format(k))
#             print('dynamic_method:', dynamic_method, ' hyper_method:', hyper_method)
#             f.write('python l2_regularization.py --dynamic_method {} --hyper_method {} \n'.format(
#                 ','.join([dynamic for dynamic in dynamic_method]), ','.join([hyper for hyper in hyper_method])))
#
#     for dynamic_method in dynamic_method_dm:
#         for hyper_method in hyper_method_dm:
#             k += 1
#             print("Comb.{}:".format(k))
#             print('dynamic_method:', dynamic_method, ' hyper_method:', hyper_method)
#             f.write('python l2_regularization.py --dynamic_method {} --hyper_method {} \n'.format(','.join(
#                 [dynamic for dynamic in dynamic_method]), ','.join([hyper for hyper in hyper_method])))
#     for hyper_method in fogm_method:
#         k += 1
#         print("Comb.{}:".format(k))
#         print('hyper_method:', hyper_method)
#         f.write('python l2_regularization.py --fo_gm {} \n'.format(hyper_method[0]))
# print('right!')
# # os.chdir(folder)
# os.system(batfolder)

import os
import time
import shutil
import platform

# 获取当前时间
t0 = time.strftime("%Y_%m_%d_%H_%M_%S")
args = 'l2_regularization/method_test'  # 使用相对路径
# dynamic_methodlist = (["NGD"], ["DI", "NGD"], ["GDA", "NGD"], ["GDA", "NGD", "DI"], ["DI", "NGD", "GDA"])
# dynamic_method_dm = (["NGD", "DM"], ["NGD", "DM", "GDA"])
# hyper_methodlist = (
#     ["CG"], ["CG", "PTT"], ["RAD"], ["RAD", "PTT"], ["RAD", "RGT"], ["PTT", "RAD", "RGT"], ["FD"], ["FD", "PTT"], ["NS"],
#     ["NS", "PTT"], ["IGA"]
# )
# hyper_method_dm = (["RAD"], ["CG"])
fogm_method = (["VSM"], ["VFM"], ["MESM"], ["PGDM"])

# 获取当前脚本所在的目录（相对路径）
base_folder = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的绝对路径
folder = os.path.join(base_folder, args, t0)  # 构建相对路径

print(folder)
if not os.path.exists(folder):
    os.makedirs(folder)

# 修改 bat 文件为 sh 文件
script_extension = '.bat' if platform.system() == "Windows" else '.sh'
script_file = os.path.join(folder, 'set' + script_extension)

# 将 Python 文件复制到目标文件夹
ganfolder = os.path.join(folder, 'l2_regularization.py')
shutil.copyfile(os.path.join(base_folder,'l2_regularization.py'), ganfolder)
# utilfolder = os.path.join(folder, 'util_file.py')
# shutil.copyfile(os.path.join(base_folder,'util_file.py'), utilfolder)

# 创建批处理或 shell 脚本
with open(script_file, 'w') as f:
    k = 0
    # for dynamic_method in dynamic_methodlist:
    #     for hyper_method in hyper_methodlist:
    #         k += 1
    #         print("Comb.{}:".format(k))
    #         print('dynamic_method:', dynamic_method, ' hyper_method:', hyper_method)
    #         f.write('python l2_regularization.py --dynamic_method {} --hyper_method {} \n'.format(
    #             ','.join([dynamic for dynamic in dynamic_method]), ','.join([hyper for hyper in hyper_method])))
    #
    # for dynamic_method in dynamic_method_dm:
    #     for hyper_method in hyper_method_dm:
    #         k += 1
    #         print("Comb.{}:".format(k))
    #         print('dynamic_method:', dynamic_method, ' hyper_method:', hyper_method)
    #         f.write('python l2_regularization.py --dynamic_method {} --hyper_method {} \n'.format(','.join(
    #             [dynamic for dynamic in dynamic_method]), ','.join([hyper for hyper in hyper_method])))

    for hyper_method in fogm_method:
        k += 1
        print("Comb.{}:".format(k))
        print('hyper_method:', hyper_method)
        f.write('python l2_regularization.py --fo_gm {} \n'.format(hyper_method[0]))

# 如果是 Ubuntu 系统, 使得脚本具有执行权限
if platform.system() != "Windows":
    os.chmod(script_file, 0o775)  # 给 sh 文件执行权限

print('right!')

# 切换到指定文件夹并运行脚本
# os.chdir(folder)
if platform.system() == "Windows":
    os.system(script_file)  # Windows 下运行 .bat 文件
else:
    os.system(f"bash {script_file}")  # Ubuntu 下运行 .sh 文件



