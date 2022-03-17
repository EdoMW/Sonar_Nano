# Sonar_Nano

grape = ID, mask, pixel center, world center, angle, pixel width
, pixel length , sprayed, dist_to_center

dist_to_center = euclidean distance between pixel_center to center of image.

ID = unique ID per grape cluster.

grapes = grapes detected in image

target_bank = list of grapes, sorted first by sprayed/ not sprayed, second by a chosen parameter

in our case for example always the grape that is in the opposite direction of advancement

import packages and the ReadFromRobot class- do not change

list of parameters to tune:

    step_size = 0.1
    
    sleep_time = 2.9
    
    safety_dist = 0.20
    
    distance = 0.75
    
    same_grape_distance_threshold =0.04m
    
    show_images = True (default) show images
    
    Vertical step size = step_size * 0.9
    
    amount of horizontal steps before moving the platform  = 1
    
    DETECTION_MIN_CONFIDENCE = 0.8

    DETECTION_NMS_THRESHOLD = 0.1
    
Requirements (python 3.7):

absl-py==1.0.0
astor==0.8.1
attrs==21.4.0
Automat==20.2.0
backcall==0.2.0
cached-property==1.5.2
certifi==2021.10.8
charset-normalizer==2.0.12
colorama==0.4.4
constantly==15.1.0
cycler==0.11.0
decorator==5.1.1
docopt==0.6.2
enum-compat==0.0.3
fonttools==4.29.1
gast==0.2.2
google-pasta==0.1.6
grpcio==1.44.0
h5py==2.10.0
hyperlink==21.0.0
idna==3.3
imageio==2.16.0
importlib-metadata==4.11.1
imutils==0.5.4
incremental==21.3.0
ipython==7.31.1
jedi==0.18.1
joblib @ file:///tmp/build/80754af9/joblib_1635411271373/work
Keras==2.1.5
Keras-Applications==1.0.8
Keras-Preprocessing==1.0.5
kiwisolver==1.3.2
Markdown==3.3.6
matplotlib==3.5.1
matplotlib-inline==0.1.3
mkl-fft==1.3.1
mkl-random @ file:///C:/ci/mkl_random_1626186163140/work
mkl-service==2.4.0
networkx==2.6.3
nidaqmx==0.6.0
numpy @ file:///C:/ci_310/numpy_and_numpy_base_1643798589088/work
opencv-python==4.5.5.62
opt-einsum==2.3.2
packaging==21.3
pandas==1.3.5
parso==0.8.3
pickleshare==0.7.5
Pillow==9.0.1
pipreqs==0.4.11
prompt-toolkit==3.0.28
protobuf==3.19.4
Pygments==2.11.2
pyparsing==3.0.7
python-dateutil==2.8.2
pytz==2021.3
pyueye==4.95.0
PyWavelets==1.2.0
PyYAML==6.0
requests==2.27.1
scikit-image==0.19.2
scikit-learn @ file:///C:/ci/scikit-learn_1642599122269/work
scipy @ file:///C:/ci/scipy_1641555141383/work
six @ file:///tmp/build/80754af9/six_1644875935023/work
sty==1.0.4
tensorboard==1.15.0
tensorflow-estimator==1.15.1
tensorflow-gpu==1.15.0
termcolor==1.1.0
threadpoolctl @ file:///Users/ktietz/demo/mc3/conda-bld/threadpoolctl_1629802263681/work
tifffile==2021.11.2
traitlets==5.1.1
Twisted @ git+https://github.com/twisted/twisted.git@2cd9e005f228d999342914a6aff39818784223fb
twisted-iocpsupport==1.0.2
typing_extensions==4.1.1
urllib3==1.26.8
wcwidth==0.2.5
Werkzeug==2.0.3
wincertstore==0.2
wrapt==1.11.1
yarg==0.1.9
zipp==3.7.0
zope.interface==5.4.0
