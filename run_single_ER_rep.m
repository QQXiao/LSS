function run_single_ER(sub,r)
addpath /seastor/helenhelen/scripts/NIFTI
%addpath /opt/fmritools/spm
addpath ~/DQ/project/TS/pattern/LSS
basedir='/seastor/helenhelen/TS';
labelfile=sprintf('%s/behavior/label/sub%02d_run%d.mat',basedir,sub,r);
load(labelfile);
label_list=ts(:,2);
featdir=sprintf('%s/sub%02d/analysis/singletrial_run%d.feat',basedir,sub,r);
	
single_event_model_ls2_rep(featdir,label_list);
