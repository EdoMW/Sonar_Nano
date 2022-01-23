from pathlib import *

# r'D:\Users\NanoProject\Sonar_Nano\playing.py'


a = Path(r'D:\Users\path_lib')
print(type(a))
a = a / 'TB_summary.txt'
print(a)

# print([x for x in path.iterdir() if x.is_dir()])
# print(a.parts)
# print(pathlib.Path(__file__))
#
# window_path = pathlib.WindowsPath(__file__)
# window_pure_path = pathlib.PureWindowsPath(__file__)
#
# print(window_path.parents[1])
# data_folder = pathlib.Path(__file__).parents[0].joinpath('data_folder')
#
# if not data_folder.exists():
#     data_folder.mkdir()
# print(data_folder)
#
# data_folder.joinpath('my_file.txt').touch(exist_ok=True)
# data_folder.joinpath('my_fileder').mkdir(exist_ok=True)