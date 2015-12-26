function run_single_ER(sub,c,r,s)
%sub=1;c=1;r=1;s=1;
condname={'encoding','test'};
addpath /seastor/helenhelen/scripts/NIFTI
addpath /opt/fmritools/spm/spm5
addpath ~/DQ/project/ISR_2015/pre/LSS
basedir='/seastor/helenhelen/ISR_2015';
featdir=sprintf('%s/ISR%02d/analysis/ms_singletrial_%s_run%d_set%d',basedir,sub,condname{c},r,s);
condfile=sprintf('%s/behav/cond_list/sub%02d_%s_run%d_set%d.txt',basedir,sub,condname{c},r,s);
%single_event_model_ls2(featdir);
%if c==3
single_event_model_ls2_ms(featdir,condfile);
%else
%single_event_model_ls2(featdir,condfile);
%end
