#!/opt/fmritools/pylib/anaconda/bin/python
from os.path import join as opj
import os
import glob
import subprocess
import shutil
import time

# define directory
basedir = '/seastor/zhifang/Testing_effect'
patterndir = '%s/Pattern'%(basedir)
tmapdir = '%s/SingleTrial_Tmap'%(patterndir)
e2refdir = '%s/Pattern/Transforms/epi2refepi'%(basedir)
refdir = '%s/Pattern/Bold_Data/example_func'%(basedir)
logdir = '%s/Scripts'%(basedir)

# check directory
if os.path.exists(tmapdir) == False:
    os.makedirs(tmapdir)

# define reference run
refrun = 'learning1'

# loop for tmap
tmaplist = sorted(glob.glob(opj(basedir,'Sub[0-9][0-9]','analysis',
                                'singletrial_*.feat','betaseries',
                                'ev1_lss_betas_normalized.nii.gz')))
for cur_tmap in tmaplist:
    iSub = cur_tmap.split('/')[4]
    iRun = cur_tmap.split('/')[6].split('_')[1][:-5]

    # apply transformation to each tmap, get aligned tmap
    if iRun != refrun:
        infile = cur_tmap
        reffile = opj(refdir,'%s_learning1_example_func.nii.gz'%(iSub))
        outfile = opj(tmapdir,'%s_%s_tmap.nii.gz'%(iSub, iRun))
        transmatrix = opj(e2refdir,'%s_%s_2ref_0GenericAffine.mat'%(iSub, iRun))
        logfile = 'Coreg_%s_%s'%(iSub, iRun)
        apply_transform = ('fsl_sub ' +
                           '-l %s '%(logdir) +
                           '-N %s '%(logfile) +
                           'antsApplyTransforms ' +
                           '--dimensionality 3 ' +
                           '--input-image-type 3 ' +
                           '--input %s '%(infile) +
                           '--reference-image %s '%(reffile) +
                           '--output %s '%(outfile) +
                           '--transform %s '%(transmatrix) +
                           '--float 1')
        subprocess.call(apply_transform, shell=True)
    elif iRun == refrun:
        infile = cur_tmap
        outfile = opj(tmapdir,'%s_%s_tmap.nii.gz'%(iSub, iRun))
        shutil.copy2(infile, outfile)

# check status
while True:
    time.sleep(60)
    num = subprocess.check_output('qstat | grep Coreg_ | wc -l', shell=True)
    if int(num[:-1]) == 0:
        break

# clean up
print 'All jobs has been done. Please check the log files!\n'
while 1:
    res = raw_input('Clean up log files? y/n\n')
    if res == 'y':
        junklist = sorted(glob.glob('%s/Coreg_*.*'%(logdir)))
        for junk in list(junklist):
            os.remove(junk)
        print 'Log files has been cleaned.'
        break
    elif res == 'n':
        print 'Log files will not be cleaned.'
        break
