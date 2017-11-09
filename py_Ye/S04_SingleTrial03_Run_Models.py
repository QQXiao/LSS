#!/opt/fmritools/pylib/anaconda/bin/python

import os
import glob
import time
import subprocess

# Set this to the directory all of the Sub### directories live in
basedir = '/seastor/zhifang/Testing_effect'

# Set this to the directory where you'll dump all the fsf files
fsfdir = '%s/Scripts/fsfs/singletrial'%(basedir)

# learning
fsf_files = sorted(glob.glob("%s/Sub[0-9][0-9]_learning[0-9].fsf"%(fsfdir)))
for fsf_file in list(fsf_files):
    os.system('feat %s'%(fsf_file))

# learning_NoMem
fsf_files = sorted(glob.glob("%s/Sub[0-9][0-9]_learning[0-9]_NoMem.fsf"%(fsfdir)))
for fsf_file in list(fsf_files):
    os.system('feat %s'%(fsf_file))

# testing
fsf_files = sorted(glob.glob("%s/Sub[0-9][0-9]_testing[0-9].fsf"%(fsfdir)))
for fsf_file in list(fsf_files):
    os.system('feat %s'%(fsf_file))

# localizer
fsf_files = sorted(glob.glob("%s/Sub[0-9][0-9]_localizer[0-9].fsf"%(fsfdir)))
for fsf_file in list(fsf_files):
    os.system('feat %s'%(fsf_file))

# check status
while True:
    time.sleep(450)
    num = subprocess.check_output('qstat | grep feat5 | wc -l',
    shell=True)
    if int(num[:-1]) == 0:
        break

print 'All feat jobs has been done!\n'
