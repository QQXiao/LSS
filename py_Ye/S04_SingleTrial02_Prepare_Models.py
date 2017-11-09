#!/opt/fmritools/pylib/anaconda/bin/python
from os.path import join as opj
import os
import glob
import scipy.io as sio
import pandas as pd

basedir = '/seastor/zhifang/Testing_effect'
fsfdir = '%s/Scripts/fsfs'%(basedir)
outdir = '%s/singletrial'%(fsfdir)
# Check directory
if os.path.exists(outdir)==False:
    os.makedirs(outdir)
os.system('rm %s/*.fsf'%(outdir))

# learning
modellist = sorted(glob.glob(opj(basedir,'Sub[0-9][0-9]','behav',
                                 'SingleTrial_learning',
                                 'learning[1-6]_all.txt')))
for cur_model in list(modellist):
    iSub = cur_model.split('/')[4]
    iRun = cur_model.split('/')[-1].split('_')[0]
    replacements = {'Sub01':iSub,'learning1':iRun}
    # open template .fsf
    with open("%s/template_learning_singletrial.fsf"%(fsfdir)) as infile:
        # open a new .fsf file
        with open("%s/%s_%s.fsf"%(outdir,iSub,iRun),'w') as outfile:
            for line in infile:
                for src, target in replacements.iteritems():
                    line = line.replace(src, target)
                outfile.write(line)

# testing
modellist = sorted(glob.glob(opj(basedir,'Sub[0-9][0-9]','behav',
                                 'SingleTrial_testing',
                                 'testing[1-3]_all.txt')))
for cur_model in list(modellist):
    iSub = cur_model.split('/')[4]
    iRun = cur_model.split('/')[-1].split('_')[0]
    replacements = {'Sub01':iSub,'testing1':iRun}
    # open template .fsf
    with open("%s/template_testing_singletrial.fsf"%(fsfdir)) as infile:
        # open a new .fsf file
        with open("%s/%s_%s.fsf"%(outdir,iSub,iRun),'w') as outfile:
            for line in infile:
                for src, target in replacements.iteritems():
                    line = line.replace(src, target)
                outfile.write(line)

# localizer
modellist = sorted(glob.glob(opj(basedir,'Sub[0-9][0-9]','behav',
                                 'SingleTrial_localizer',
                                 'localizer[1-3]_all.txt')))
for cur_model in list(modellist):
    iSub = cur_model.split('/')[4]
    iRun = cur_model.split('/')[-1].split('_')[0]
    replacements = {'Sub01':iSub,'localizer1':iRun}
    # open template .fsf
    with open("%s/template_localizer_singletrial.fsf"%(fsfdir)) as infile:
        # open a new .fsf file
        with open("%s/%s_%s.fsf"%(outdir,iSub,iRun),'w') as outfile:
            for line in infile:
                for src, target in replacements.iteritems():
                    line = line.replace(src, target)
                outfile.write(line)
