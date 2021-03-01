import sys
import subprocess
import pkg_resources

def install_and_import(packages):
    required = {*packages}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed

    if missing:
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

def main():

    install_and_import(['nidaqmx', 'numpy', 'scipy', 'matplotlib', 'pickle', 'sklearn'])

if __name__ == "__main__":
    main()