function run_single_ER(sub,c,r,s)
%sub=2;r=1;
condname={'encoding','test'};
addpath /seastor/helenhelen/scripts/NIFTI
addpath /opt/fmritools/matlabtoolbox/spm/spm8
addpath ~/DQ/project/gitrepo/LSS
basedir='/seastor/helenhelen/ISR_2015';
featdir=sprintf('%s/ISR%02d/analysis/zsingle_%s_run%d_set%d.feat',basedir,sub,condname{c},r,s);
single_event_model_ls2(featdir);
