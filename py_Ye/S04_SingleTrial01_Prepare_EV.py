#!/opt/fmritools/pylib/anaconda/bin/python
from os.path import join as opj
import os
import glob
import shutil
import scipy.io as sio
import pandas as pd
import numpy as np
import itertools

basedir = '/seastor/zhifang/Testing_effect'
resultdir = '%s/Results'%(basedir)
behavdir = '%s/Behav'%(basedir)

# define subject list
SubjectList = sorted(glob.glob(opj(basedir,'Sub[0-9][0-9]')))

# load behavior data
learning = pd.read_csv(opj(resultdir,'Behavior_Learning.csv'))
testing = pd.read_csv(opj(resultdir,'Behavior_Testing.csv'))
localizer = pd.read_csv(opj(resultdir,'Behavior_Localizer.csv'))
testing_grate = pd.read_csv(opj(resultdir,'Behavior_Testing_Grate.csv'))
localizer_grate = pd.read_csv(opj(resultdir,'Behavior_Localizer_Grate.csv'))

# loop for subjects
for cur_sub in SubjectList:
    iSub = cur_sub.split('/')[-1]

    ## learning
    cur_dir = opj(basedir,iSub,'behav','SingleTrial_learning')
    if os.path.exists(cur_dir) == False:
        os.makedirs(cur_dir)
    learning_subj = learning.loc[learning.SubjID==int(iSub[-2:]),:]
    ### loop for run
    for iRun in range(1,7):
        filename = opj(cur_dir,'learning%d_all.txt'%(iRun))
        onset=learning_subj.loc[(learning_subj.RunID==iRun),'AOnset']
        ev = pd.DataFrame({'Onset':onset.values,
                          'Duration':[2.0]*onset.shape[0],
                          'Weight':[1.0]*onset.shape[0]},
                          columns=['Onset','Duration','Weight'])
        ev.to_csv(filename, sep='\t', header=False, index=False,
                  float_format='%.4f')

    ## testing
    cur_dir = opj(basedir,iSub,'behav','SingleTrial_testing')
    if os.path.exists(cur_dir) == False:
        os.makedirs(cur_dir)
    testing_subj = testing.loc[testing.SubjID==int(iSub[-2:]),:]
    for iRun in range(1,4):
        filename = opj(cur_dir,'testing%d_all.txt'%(iRun))
        onset=testing_subj.loc[(testing_subj.RunID==iRun),'AOnset']
        ev = pd.DataFrame({'Onset':onset.values,
                          'Duration':[4.0]*onset.shape[0],
                          'Weight':[1.0]*onset.shape[0]},
                          columns=['Onset','Duration','Weight'])
        ev.to_csv(filename, sep='\t', header=False, index=False,
                  float_format='%.4f')

    ## localizer
    cur_dir = opj(basedir,iSub,'behav','SingleTrial_localizer')
    if os.path.exists(cur_dir) == False:
        os.makedirs(cur_dir)
    localizer_subj = localizer.loc[localizer.SubjID==int(iSub[-2:]),:]
    for iRun in range(1,4):
        filename = opj(cur_dir,'localizer%d_all.txt'%(iRun))
        onset=localizer_subj.loc[(localizer_subj.RunID==iRun),'AOnset']
        ev = pd.DataFrame({'Onset':onset.values,
                          'Duration':[3.5]*onset.shape[0],
                          'Weight':[1.0]*onset.shape[0]},
                          columns=['Onset','Duration','Weight'])
        ev.to_csv(filename, sep='\t', header=False, index=False,
                  float_format='%.4f')
