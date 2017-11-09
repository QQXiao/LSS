#!/opt/fmritools/pylib/anaconda/bin/python
"""pybetaseries: a module for computing beta-series regression on fMRI data

Includes:
pybetaseries: main function
spm_hrf: helper function to generate double-gamma HRF
"""

from mvpa2.misc.fsl.base import *
from mvpa2.datasets.mri import fmri_dataset
import numpy as np
import nibabel
import scipy.stats
from scipy.ndimage import convolve1d
from scipy.sparse import spdiags
from scipy.linalg import toeplitz
from mvpa2.datasets.mri import *
import os, sys
from copy import copy
import argparse

class pybetaseries():

    def __init__(self,fsfdir,tempderiv,motpars,time_res):
        #Sets up dataset wide variables
        self.tempderiv=tempderiv
        self.motpars=motpars
        self.time_res=time_res

        if not os.path.exists(fsfdir):
            print 'ERROR: %s does not exist!'%fsfdir

        if not fsfdir.endswith('/'):
            fsfdir=''.join([fsfdir,'/'])

        self.fsfdir=fsfdir

        fsffile=''.join([self.fsfdir,'design.fsf'])
        desmatfile=''.join([self.fsfdir,'design.mat'])

        design=read_fsl_design(fsffile)

        self.desmat=FslGLMDesign(desmatfile)

        self.nevs=self.desmat.mat.shape[1]
        self.ntp=self.desmat.mat.shape[0]

        self.TR=round(design['fmri(tr)'],2)

        self.hrf=spm_hrf(self.time_res)

        self.time_up=np.arange(0,self.TR*self.ntp+self.time_res, self.time_res)

        self.max_evtime=self.TR*self.ntp - 2
        self.n_up=len(self.time_up)

        if not os.path.exists(fsfdir+'betaseries'):
            os.mkdir(fsfdir+'betaseries')

        # load bold data
        maskimg=''.join([fsfdir,'mask.nii.gz'])
        self.raw_data=fmri_dataset(fsfdir+'filtered_func_data.nii.gz',mask=maskimg)
        voxmeans = np.mean(self.raw_data.samples,axis=0)
        self.data=self.raw_data.samples-voxmeans
        self.nvox=self.raw_data.nfeatures
        cutoff=design['fmri(paradigm_hp)']/self.TR
        ## get hp kernel similar to what feat does
        self.F=get_smoothing_kernel(cutoff, self.ntp)

    # method for least square single beta estimate
    def lss(self,whichevs,numorigev):
        method='lss'
        print "Calculating lss: Ev %s" %(whichevs[0])
        # define nuisance regressor (including motion parameters)
        nuisance = otherevs(whichevs,numorigev,self.tempderiv,self.motpars)
        # load ev file
        ons=FslEV3(self.fsfdir+'custom_timing_files/ev%d.txt'%int(whichevs[0]))
        # nuisance regressor in feat design matrix
        dm_nuisanceevs = self.desmat.mat[:, nuisance]
        # number of trials
        ntrials=len(ons['onsets'])
        # initialize variables
        beta_maker=np.zeros((ntrials,self.ntp))
        beta_maker_all=np.zeros((ntrials,self.nevs+1,self.ntp))
        dm_trials=np.zeros((self.ntp,ntrials))
        # loop for trials
        for t in range(ntrials):
            if ons['onsets'][t] > self.max_evtime:
                continue
            # build model for each trial
            dm_trial=np.zeros(self.n_up)
            window_ons = [np.where(self.time_up==x)[0][0]
                          for x in self.time_up
                          if ons['onsets'][t] <= x <
                          ons['onsets'][t]+ons['durations'][t]]

            dm_trial[window_ons]=1
            dm_trial=np.convolve(dm_trial,self.hrf)[0:int(self.ntp/self.time_res*self.TR):int(self.TR/self.time_res)]
            dm_trials[:,t]=dm_trial
        #  apply hp to design matrix
        dm_full=np.dot(self.F,dm_trials)
        dm_full=dm_full - np.kron(np.ones((dm_full.shape[0],dm_full.shape[1])),np.mean(dm_full,0))[0:dm_full.shape[0],0:dm_full.shape[1]]
        # calculate beta values
        for p in range(len(dm_full[1,:])):
            target=dm_full[:,p]
            dmsums=np.sum(dm_full,1)-dm_full[:,p]
            des_loop=np.hstack((target[:,np.newaxis],dmsums[:,np.newaxis],dm_nuisanceevs))
            beta_maker_loop=np.linalg.pinv(des_loop)
            beta_maker[p,:]=beta_maker_loop[0,:]
	    print(p)
            beta_maker_all[p,:,:]=beta_maker_loop
        # this uses Jeanette's trick of extracting the beta-forming vector for each
        # trial and putting them together, which allows estimation for all trials
        # at once
        betas_lss = np.dot(beta_maker,self.data)
        ni = map2nifti(self.raw_data,betas_lss)
        ni.to_filename(self.fsfdir+'betaseries/ev%d_%s.nii.gz'%(int(whichevs[0]),method))

        # calculate univariate noise normalization t value
        se = np.zeros((ntrials,self.nvox))
        sigma = np.zeros((ntrials,self.nvox))
        for p in range(len(dm_full[1,:])):
            target=dm_full[:,p]
            dmsums=np.sum(dm_full,1)-dm_full[:,p]
            des_loop=np.hstack((target[:,np.newaxis],dmsums[:,np.newaxis],dm_nuisanceevs))
            beta_loop = np.dot(beta_maker_all[p,:,:],self.data)
            residual = self.data - np.dot(des_loop,beta_loop)
            sigma_loop = scipy.var(residual,axis=0)
            sigma[p,:] = sigma_loop
            se_maker_loop = scipy.linalg.inv(np.dot(des_loop.T,des_loop))[0,0]
            se_loop = scipy.sqrt(sigma_loop*se_maker_loop)
            se[p,:] = se_loop
        tvalues = betas_lss/se
        betas_lss_normalized = betas_lss/scipy.sqrt(sigma)
        ni = map2nifti(self.raw_data,data=tvalues)
        ni.to_filename(self.fsfdir+'betaseries/ev%d_%s_tmap.nii.gz'%(int(whichevs[0]),method))
        ni = map2nifti(self.raw_data,data=betas_lss_normalized)
        ni.to_filename(self.fsfdir+'betaseries/ev%d_%s_betas_normalized.nii.gz'%(int(whichevs[0]),method))

    # method for least square all beta estimate
    def lsa(self,whichevs,numorigev):
        method='lsa'
        print "Calculating lsa"
        # nuisance regressor in feat design matrix
        nuisance = otherevs(whichevs,numorigev,self.tempderiv,self.motpars)
        # nuisance regressor in feat design matrix
        dm_nuisanceevs = self.desmat.mat[:, nuisance]
        # initialize variables
        all_onsets=[]
        all_durations=[]
        all_conds=[]  # condition marker
        # loop for condition (if more than 1 condition)
        for e in range(len(whichevs)):
            ev=whichevs[e]
            ons=FslEV3(self.fsfdir+'custom_timing_files/ev%d.txt'%int(ev))
            all_onsets=np.hstack((all_onsets,ons['onsets']))
            all_durations=np.hstack((all_durations,ons['durations']))
            all_conds=np.hstack((all_conds,np.ones(len(ons['onsets']))*(ev)))
        # number of trials
        ntrials = len(all_onsets)
        # initialize variables
        betas_lsa_all = np.zeros((self.nvox,ntrials))
        dm_trials = np.zeros((self.ntp,ntrials))
        dm_full = []
        for t in range(ntrials):
            if all_onsets[t] > self.max_evtime:
                continue
            dm_trial=np.zeros(self.n_up)
            window_ons = [np.where(self.time_up==x)[0][0]
                      for x in self.time_up
                      if all_onsets[t] <= x < all_onsets[t] + all_durations[t]]
            dm_trial[window_ons]=1
            dm_trial=np.convolve(dm_trial,self.hrf)[0:int(self.ntp/self.time_res*self.TR):int(self.TR/self.time_res)]
            dm_trials[:,t]=dm_trial
        # apply hp to design matrix
        dm_full = np.dot(self.F,dm_trials)
        dm_full = dm_full - np.kron(np.ones((dm_full.shape[0],dm_full.shape[1])),np.mean(dm_full,0))[0:dm_full.shape[0],0:dm_full.shape[1]]
        # add nuisance regressor to design matrix
        dm_full = np.hstack((dm_full,dm_nuisanceevs))
        # calculate beta vector
        betas_lsa_all = np.dot(np.linalg.pinv(dm_full),self.data)
        # select beta corresponds to trial regressor
        betas_lsa = betas_lsa_all[0:ntrials,:]

        # calculate univariate noise normalization t value
        residual = self.data - np.dot(dm_full,betas_lsa_all)
        sigma = scipy.var(residual,axis=0)
        # loop for regressors
        se_maker = scipy.linalg.inv(np.dot(dm_full.T,dm_full)).diagonal()[0:ntrials]
        se_maker = np.array([se_maker]*self.nvox).T
        se = scipy.sqrt(np.array([sigma]*ntrials).shape * se_maker)
        # univariate noise normalization
        tvalues = betas_lsa/se
        betas_lsa_normalized = betas_lsa/scipy.sqrt(sigma)

        for e in whichevs:
            ni=map2nifti(self.raw_data,data=betas_lsa[np.where(all_conds==(e))[0],:])
            ni.to_filename(self.fsfdir+'betaseries/ev%d_%s.nii.gz'%(int(e),method))
            ni=map2nifti(self.raw_data,data=tvalues[np.where(all_conds==(e))[0],:])
            ni.to_filename(self.fsfdir+'betaseries/ev%d_%s.nii.gz'%(int(e),method))
            ni=map2nifti(self.raw_data,data=betas_lsa_normalized[np.where(all_conds==(e))[0],:])
            ni.to_filename(self.fsfdir+'betaseries/ev%d_%s.nii.gz'%(int(e),method))


# generate high-pass filter kernel
def get_smoothing_kernel(cutoff, ntp):
    sigN2 = (cutoff/(np.sqrt(2.0)))**2.0
    K = toeplitz(1
                 /np.sqrt(2.0*np.pi*sigN2)
                 *np.exp((-1*np.array(range(ntp))**2.0/(2*sigN2))))
    K = spdiags(1./np.sum(K.T, 0).T, 0, ntp, ntp)*K
    H = np.zeros((ntp, ntp)) # Smoothing matrix, s.t. H*y is smooth line
    X = np.hstack((np.ones((ntp, 1)), np.arange(1, ntp+1).T[:, np.newaxis]))
    for  k in range(ntp):
        W = np.diag(K[k, :])
        Hat = np.dot(np.dot(X, np.linalg.pinv(np.dot(W, X))), W)
        H[k, :] = Hat[k, :]

    F = np.eye(ntp) - H
    return F

# deal with nuisance regressors
def otherevs(whichevs,numorigev,tempderiv,motpars):
        #sets up the onsets and nuisance EVs for given target EV
        if tempderiv:
            nuisance=range(0,2*numorigev)
            popevs=[(ev-1)*2 for ev in whichevs]
            nuisance=[i for i in nuisance if i not in popevs]

            if motpars:
                # modified by Zhifang Ye, 5/23/2016
                # in Feat 6.00, if we used Standard Motion Parameters, there's 6
                # motion parameters added to the design matrix
                nuisance.extend(range(2*numorigev,(6+2*numorigev)))

        else:
            nuisance=range(0,numorigev)
            popevs=[(ev-1) for ev in whichevs]
            nuisance=[i for i in nuisance if i not in popevs]

            if motpars:
                nuisance.extend(range(numorigev,6+numorigev))

        return nuisance

# spm HRF
def spm_hrf(TR,p=[6,16,1,1,6,0,32]):
    """ An implementation of spm_hrf.m from the SPM distribution

    Arguments:

    Required:
    TR: repetition time at which to generate the HRF (in seconds)

    Optional:
    p: list with parameters of the two gamma functions:
                                                         defaults
                                                        (seconds)
       p[0] - delay of response (relative to onset)         6
       p[1] - delay of undershoot (relative to onset)      16
       p[2] - dispersion of response                        1
       p[3] - dispersion of undershoot                      1
       p[4] - ratio of response to undershoot               6
       p[5] - onset (seconds)                               0
       p[6] - length of kernel (seconds)                   32

    """
    p=[float(x) for x in p]
    fMRI_T = 16.0
    TR=float(TR)
    dt  = TR/fMRI_T
    u   = np.arange(p[6]/dt + 1) - p[5]/dt
    hrf=scipy.stats.gamma.pdf(u,p[0]/p[2],scale=1.0/(dt/p[2])) - scipy.stats.gamma.pdf(u,p[1]/p[3],scale=1.0/(dt/p[3]))/p[4]
    good_pts=np.array(range(np.int(p[6]/TR)))*fMRI_T
    hrf=hrf[list(good_pts)]
    hrf = hrf/np.sum(hrf);
    return hrf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fsldir',dest='fsldir',default='',help='Path to Target FSL Directory')
    parser.add_argument('--whichevs', dest='whichevs',type=int, nargs='*',help='List of EVs to Compute Beta Series for. Number corresponds to orig EV number in FEAT.')
    parser.add_argument('--numorigev',dest='numorigev',type=int,help='Total number of orig EVs in Feat model')
    parser.add_argument('-motpars',dest='motpars',action='store_true',
            default=False,help='Include tag if motion parameters are to be included in model. The code assumes that motion parameters are the first 6 EVs (12 if including temporal derivative EVs) after the real EVs in the Feat design matrix.')
    parser.add_argument('-tempderiv',dest='tempderiv',action='store_true',
            default=False,help='Include tag if the original design matrix includes temporal derivates. The code assumes that temporal derivatives are immediately after each EV/motion parameter in the Feat design matrix.')
    parser.add_argument('--timeres',dest='timeres',type=float,default=0.001, help='Time resolution for convolution.')
    parser.add_argument('-lsa',dest='lsa',action='store_true',
            default=False,help='Include tag to compute lsa.')
    parser.add_argument('-lss',dest='lss',action='store_true',
            default=False,help='Include tag to compute lss.')

    args = parser.parse_args()

    # create instance
    pybeta = pybetaseries(args.fsldir,args.tempderiv,args.motpars,args.timeres)
    if args.lss:
        for ev in args.whichevs:
            pybeta.lss([ev],args.numorigev)
    if args.lsa:
        pybeta.lsa(args.whichevs,args.numorigev)
