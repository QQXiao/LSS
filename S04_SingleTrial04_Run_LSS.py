#!/opt/fmritools/pylib/anaconda/bin/python
from os.path import join as opj
import os
import glob
import subprocess
import shutil
import time

basedir = '/seastor/zhifang/Testing_effect'
logdir = '%s/Scripts'%(basedir)

learning_list = sorted(glob.glob(opj(basedir,'Sub[0-9][0-9]','analysis',
                                     'singletrial_learning[1-6].feat')))
testing_list = sorted(glob.glob(opj(basedir,'Sub[0-9][0-9]','analysis',
                                     'singletrial_testing[1-6].feat')))
localizer_list = sorted(glob.glob(opj(basedir,'Sub[0-9][0-9]','analysis',
                                     'singletrial_localizer[1-6].feat')))

# learning
for cur_feat in learning_list:
    iSub = cur_feat.split('/')[4]
    iRun = cur_feat.split('/')[-1].split('_')[-1][:-5]

    logfile = '%s_le%s'%(iSub,iRun[-1])

    do_lss = ('fsl_sub ' +
              '-l %s '%(logdir) +
              '-N %s '%(logfile) +
              'python '
              '%s/Scripts/H02_Calculate_SingleTrial_BetaSeries.py '%(basedir) +
              '--fsldir \'%s\' '%(cur_feat) +
              '--whichevs 1 ' +
              '--numorigev 1 ' +
              '-motpars ' +
              '-tempderiv ' +
              '-lss'
              )
    subprocess.call(do_lss, shell=True)

# testing
for cur_feat in testing_list:
    iSub = cur_feat.split('/')[4]
    iRun = cur_feat.split('/')[-1].split('_')[-1][:-5]

    logfile = '%s_te%s'%(iSub,iRun[-1])

    do_lss = ('fsl_sub ' +
              '-l %s '%(logdir) +
              '-N %s '%(logfile) +
              'python '
              '%s/Scripts/H02_Calculate_SingleTrial_BetaSeries.py '%(basedir) +
              '--fsldir \'%s\' '%(cur_feat) +
              '--whichevs 1 ' +
              '--numorigev 2 ' +
              '-motpars ' +
              '-tempderiv ' +
              '-lss'
              )
    subprocess.call(do_lss, shell=True)

# localizer
for cur_feat in localizer_list:
    iSub = cur_feat.split('/')[4]
    iRun = cur_feat.split('/')[-1].split('_')[-1][:-5]

    logfile = '%s_lo%s'%(iSub,iRun[-1])

    do_lss = ('fsl_sub ' +
              '-l %s '%(logdir) +
              '-N %s '%(logfile) +
              'python '
              '%s/Scripts/H02_Calculate_SingleTrial_BetaSeries.py '%(basedir) +
              '--fsldir \'%s\' '%(cur_feat) +
              '--whichevs 1 ' +
              '--numorigev 2 ' +
              '-motpars ' +
              '-tempderiv ' +
              '-lss'
              )
    subprocess.call(do_lss, shell=True)

# check status
while True:
    time.sleep(60)
    num = subprocess.check_output('qstat | grep Sub | wc -l',
                                  shell=True)
    if int(num[:-1]) == 0:
        break

# check files
featlist = learning_list + testing_list + localizer_list
for cur_feat in featlist:
    if not os.path.isfile('%s/betaseries/ev1_lss_tmap.nii.gz'%(cur_feat)):
        print '%s/betaseries/ev1_lss_tmap.nii.gz Not Found!'%(cur_feat)
    if not os.path.isfile('%s/betaseries/ev1_lss_tmap.nii.gz'%(cur_feat)):
        print '%s/betaseries/ev1_lss_betas_normalized.nii.gz Not Found!'%(cur_feat)
    if not os.path.isfile('%s/betaseries/ev1_lss_tmap.nii.gz'%(cur_feat)):
        print '%s/betaseries/ev1_lss.nii.gz Not Found!'%(cur_feat)

while 1:
    res = raw_input('Clean up log files? y/n\n')
    if res == 'y':
        junklist = sorted(glob.glob('%s/Sub[0-9][0-9]_*.*'%(logdir)))
        for junk in list(junklist):
            os.remove(junk)
        print 'Log files has been cleaned.'
        break
    elif res == 'n':
        print 'Log files will not be cleaned.'
        break

print 'All process should been done!'
