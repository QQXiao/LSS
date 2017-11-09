function out = single_event_model_ls2(featdir,write_std_image,use_raw, fsldir)
    % Usage: single_event_model_ls2(featdir,write_std_image,use_raw)
    % Takes a feat directory, and runs a ls and ls with 1 at a time
    % estimation on the filtered_func_data.img file within.
    %
    %  CURRENTLY THERE IS NO OPTION TO STANDARIZE YOUR DATA AND R/L SWAP
    %  PROBABLY WILL HAPPEN  YOU MUST USE FSLORIENT -FORCERADIOLOGICAL TO FIX THIS IF IT
    %  HAPPENS!!!
    %
    % write_std_image - CURRENTLY TURNED OFF set to 1 to create standard space files
    % use_raw - set to 1 to use raw data rather than filtered_func_data
    %
    % Returns a 4D nifti file of beta estimates for least squares and one
    % at a time least squares
    %
    % Jeanette Mumford 12/16/2010:  
    %      Adapted from single_event_model.m from the ridge regression code
   
    % Modified by Xiaoqian Xiao 12/26/2015
    %	Adapted for normalization each voxels accoding to its nosize level. 
    

    if ~exist('write_std_image')
      write_std_image=0;
    end
   
    if (~exist('use_raw')) | use_raw==0
      use_raw=0;
      raw_flag='';
    end
    if use_raw==1,
      raw_flag='_raw';
    end
    
    save_pth=[featdir,'/stats'];

    design = read_fsl_design2(featdir);
    
    %load onsets

    onscond=[];
    ons=[];
    %condnames={};
    goodconds=[];
    
    for evnum=1:design.evs_orig-1 % ignore the last regressor
      %condnames(evnum)={eval(sprintf('design.evtitle%d',evnum))};
      onsetf_loop=eval(sprintf('design.custom%d', evnum));
      ons_loop=load(onsetf_loop);
      %onsetf_loop=sprintf('%s/custom_timing_files/ev%d.txt',featdir, evnum);
      %ons_loop=load(eval(onsetf_loop));
      % added by RP, 11/8/09
      % skips over single-column onset files (e.g., motion parameters)
      if size(ons_loop,2)==3,
         goodconds=[goodconds evnum];
      	 ons_loop=ons_loop(ons_loop(:,3)~=0,:);
      	 ons=[ons;ons_loop];
         onscond=[onscond ones(1,size(ons_loop,1))*evnum];
      end

    end
    goodconds=unique(goodconds);

    nruns = length(ons);
    
    
    %load in nuisance regressors.  NOTE THIS IS TAILORED FOR THIS ANALYSIS
    nuisance=read_design_mat([featdir, '/design.mat']);
    nuisance=nuisance(:, [end-1:end]);
   
    %load mask
    maskf = strcat(featdir, '/mask.nii.gz');
    maskfile = load_nii_zip(maskf);
    mask = maskfile.img;
    
    %import data and convert from int16 into double
    if use_raw,
      %I need to add a catch if .fsf has .nii.gz extension already
      filenamecheck=dir(design.feat_files{1});
      if sum(filenamecheck.name((end-5):end)=='nii.gz')~=6
        dataf=strcat(design.feat_files{1},'.nii.gz');
      else
        dataf=strcat(design.feat_files{1});
      end
      
    else
      dataf = strcat(featdir, '/filtered_func_data.nii.gz');
    end
    
    datafile = load_nii_zip(dataf);
    data = datafile.img;
    data = double(data); 
    

    TR=design.tr;
    ntp=design.npts;
    onsets = round((ons(:,1)+TR)/TR);
    
    %create a 2D matrix of data
    data2D=zeros(ntp, sum(mask(:)));
    for i=1:ntp
        dat_loop=data(:,:,:,i);
        data2D(i,:)=dat_loop(mask==1);
    end
    
    

    %single event regressors
    X_single = zeros(ntp,length(onsets));
    hrf = spm_hrf(TR);
    trial = zeros(length(onsets),ntp+length(hrf)-1);

    
    for t = 1:length(onsets)
    
        ssf = zeros(1,ntp);
        ssf(onsets(t))=1;
        trial(t,:) = conv(ssf,hrf); 
        X_single(:,t) = trial(t,1:ntp)';
    
    end
  
    X_single=[X_single, nuisance];
    
   % We need to HP filter the design.  This is an approximation to what 
   % FSL does.  Note the data are already HP filtered and I'm (Jeanette)
   % assuming that this approximation is close enough to what has been
   % done to the data.  The reason I'm not starting with the original
   % is because I would like to retain the smoothing and scaling that was 
   % done to filtered_func_data
   %hpfile=sprintf('%s/absbrainthresh.txt',featdir);
   %hp=load(hpfile);
   %cut=hp/design.tr;
   cut=design.paradigm_hp/design.tr;
   sigN2=(cut/(sqrt(2)))^2;
   K=toeplitz(1/sqrt(2*pi*sigN2)*exp(-[0:(ntp-1)].^2/(2*sigN2)));
   K=spdiags(1./sum(K')', 0, ntp,ntp)*K;
   
    H = zeros(ntp,ntp); % Smoothing matrix, s.t. H*y is smooth line 
    X = [ones(ntp,1) (1:ntp)'];
     for  k = 1:ntp
       W = diag(K(k,:));
       Hat = X*pinv(W*X)*W;
       H(k,:) = Hat(k,:);
     end    

   F=eye(ntp)-H;
   
   X_single_hp=F*X_single;
   
   
   nparam=size(X_single, 2);
   nt=size(X_single, 1);
   beta_maker=zeros(nparam, nt);
   
   %This yields the final design for least squares.
   ncolX=size(X_single_hp, 2);
   for i=1:ncolX
       X_single_hp(:,i)=X_single_hp(:,i)-mean(X_single_hp(:,i));
   end
   
   %create the design matrix for the 1 at a time model (LS-S)
   %Gui, note that I put all other trials into a single regressor.
   %Although I have tried a separate regressor for each trial type,
   %it didn't work well BUT the data I was using had 4 trial
   %types.  It might be okay with 2 trial types.  There is some
   %commented out code below that could be edited for this
   %purpose.  
   for i=1:nparam
       des_loop=[X_single_hp(:,i), sum(X_single_hp, 2)-X_single_hp(:,i), nuisance];
       %hp filter
       %des_loop=F*des_loop;
       %mean center
       des_loop=des_loop-kron(ones(length(des_loop),1),mean(des_loop));
       beta_maker_loop=pinv(des_loop);
       beta_maker(i,:)=beta_maker_loop(1,:);    
   end
   
   
   %This is code you can edit if you want to break up the "other
   %trial" regressor into two or more (here I had 4)
   %for i=1:nparam
   %    keep_loop=ones(size(condlist));
   %    keep_loop(i)=0;
   %    conlist_loop=condlist(keep_loop==1);
   %    X_omit=X_single(:, keep_loop==1); 
   %    des_loop=[X_single(:,i), sum(X_omit(:,conlist_loop==1),2), sum(X_omit(:,conlist_loop==2),2), sum(X_omit(:,conlist_loop==3),2), sum(X_omit(:,conlist_loop==4),2), nuisance];
   %    
   %    des_loop=F*des_loop;
   %    des_loop=des_loop-kron(ones(length(des_loop),1),mean(des_loop));
   % 
   %    beta_maker_loop=pinv(des_loop);
   %    beta_maker(i,:)=beta_maker_loop(1,:);
   %end
   
   beta_est2D=beta_maker*data2D;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   %added by XXQ
	res_est=data2D-pinv(beta_maker)*beta_est2D;
	%cov_res=(1/nt)*(res_est'*res_est); %n(voxel) * n(voxel) variance-covariance matrix
	for v=1:size(res_est,2)
	res_std(v)=mean(res_est(:,v).*res_est(:,v))^0.5;
	end
	%[t,r]=eig(cov_res);
	%p=size(cov_res,1);
	%b=zeros(p); 
	%for i=1:p
    	%b=b+r(i,i)^-.5*t(:,i)*t(:,i)'; % variance-covariance matix * (-0.5)
	%end	
	res_data=zeros(size(data));
	for t=1:nt
	data_loop=res_data(:,:,:,1)*0;
	data_loop(mask==1)=res_est(t,:);
	res_data(:,:,:,t)=data_loop;
	end
      fname_res = sprintf('%s/res_ls_one_at_time.nii',save_pth);
      res_est_nii = datafile;     %uses 'datafile' as a templat                         
      res_est_nii.img = res_data(:,:,:,:);  %setting the image data         
      res_est_nii.hdr.dime.dim(5)=nt;   %Replace 204 with number of onsets=nruns  
      res_est_nii.hdr.dime.datatype=16;   % how the computer stores the data            
      res_est_nii.hdr.dime.bitpix=32;     %how the computer stores the data             
      save_untouch_nii(res_est_nii, fname_res);  %finally saving the data          
      system(sprintf('gzip -f %s',fname_res));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

   beta_lm2D=pinv(X_single_hp)*data2D;
   %put it into a 4D data set
   
   beta_est=data(:,:, :, 1:nparam)*0;
   beta_est_lm=data(:,:, :, 1:nparam)*0;
   
   for i=1:nparam
       data_loop=beta_est(:, :, :, 1)*0;
       data_loop(mask==1)=beta_est2D(i,:);
       %% added by XXQ
       data_loop_t=beta_est(:, :, :, 1)*0;                                                                                          
       %data_loop_pre_w=beta_est(:, :, :, 1)*0;                                                                                          
       data_loop_t(mask==1)=beta_est2D(i,:)./res_std;
       %data_loop_pre_w(mask==1)=beta_est2D(i,:)*b;
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       data_loop_lm=beta_est(:, :, :, 1)*0;
       data_loop_lm(mask==1)=beta_lm2D(i, :);
       
       beta_est(:,:,:,i)=data_loop;
       beta_est_lm(:,:,:,i)=data_loop_lm;
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       beta_est_t(:,:,:,i)=data_loop_t;
       %beta_est_pre_w(:,:,:,i)=data_loop_pre_w;
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   end
   
    %Save data
    fprintf('\nSaving data...\n');

    %use datafile as a template for 4D beta_est data
    %  You'll need to edit this to reflect the path where you want to
    % save your results
    mat_file = strcat(featdir, '/reg/example_func2standard.mat');
    std_ref=sprintf('%s/data/standard/MNI152_T1_2mm.nii.gz', ...
                    getenv('FSLDIR'))
    
    for condition=goodconds,
      condition_ons=find(onscond==condition);
      fprintf('writing RR image for condition %d...\n', condition);
      fname_beta_est = sprintf('%s/pe%dls_one_at_time%s.nii',save_pth,condition,raw_flag);
      beta_est_nii = datafile;     %uses 'datafile' as a templat
      beta_est_nii.img = beta_est(:,:,:,condition_ons);  %setting the image data 
      beta_est_nii.hdr.dime.dim(5)=length(condition_ons);   %Replace 204 with number of onsets=nruns  
      beta_est_nii.hdr.dime.datatype=16;   % how the computer stores the data
      beta_est_nii.hdr.dime.bitpix=32;     %how the computer stores the data
      save_untouch_nii(beta_est_nii, fname_beta_est);  %finally saving the data
      system(sprintf('gzip -f %s',fname_beta_est));

      condition_ons=find(onscond==condition);
      fname_beta_est_t = sprintf('%s/t_pe%dls_one_at_time%s.nii',save_pth,condition,raw_flag);
      beta_est_nii = datafile;     %uses 'datafile' as a templat
      beta_est_nii.img = beta_est_t(:,:,:,condition_ons);  %setting the image data 
      beta_est_nii.hdr.dime.dim(5)=length(condition_ons);   %Replace 204 with number of onsets=nruns  
      beta_est_nii.hdr.dime.datatype=16;   % how the computer stores the data
      beta_est_nii.hdr.dime.bitpix=32;     %how the computer stores the data
      save_untouch_nii(beta_est_nii, fname_beta_est_t);  %finally saving the data
      system(sprintf('gzip -f %s',fname_beta_est_t));

      %condition_ons=find(onscond==condition);
      %fname_beta_est_pre_w = sprintf('%s/pre_w_pe%dls_one_at_time%s.nii',save_pth,condition,raw_flag);
      %beta_est_nii = datafile;     %uses 'datafile' as a templat
      %beta_est_nii.img = beta_est_pre_w(:,:,:,condition_ons);  %setting the image data 
      %beta_est_nii.hdr.dime.dim(5)=length(condition_ons);   %Replace 204 with number of onsets=nruns  
      %beta_est_nii.hdr.dime.datatype=16;   % how the computer stores the data
      %beta_est_nii.hdr.dime.bitpix=32;     %how the computer stores the data
      %save_untouch_nii(beta_est_nii, fname_beta_est_pre_w);  %finally saving the data
      %system(sprintf('gzip -f %s',fname_beta_est_pre_w));
      
      condition_ons=find(onscond==condition);
      fprintf('writing RR image for condition %d...\n',condition);
      fname_beta_ls_est = sprintf('%s/pe%dls_all%s.nii',save_pth,condition,raw_flag);
      beta_est_nii = datafile;     %uses 'datafile' as a templat
      beta_est_nii.img = beta_est_lm(:,:,:,condition_ons);  %setting the image data 
      beta_est_nii.hdr.dime.dim(5)=length(condition_ons);   %Replace 204 with number of onsets=nruns  
      beta_est_nii.hdr.dime.datatype=16;   % how the computer stores the data
      beta_est_nii.hdr.dime.bitpix=32;     %how the computer stores the data
      save_untouch_nii(beta_est_nii, fname_beta_ls_est);  %finally saving the data
      system(sprintf('gzip -f %s',fname_beta_ls_est));
      
      %flips the image so it is correct in fslview
      %system('echo FSLOUTPUTTYPE=NIFTI_GZ')
      %system(sprintf(['%s/bin/fslorient -forceradiological %s.gz'], fsldir,fname_beta_est));
      %system(sprintf(['%s/bin/fslorient -forceradiological %s.gz'], fsldir,fname_beta_ls_est));

      %if write_std_image,
        
       % system(sprintf(['%s/bin/fslconf/fsl.sh'], fsldir));  
          
        %fprintf('writing standard space image for %s...\n',condnames{condition});
        %std_beta_est = sprintf('%s/pe%dls_one_at_time%s_std.nii',save_pth, ...
        %                         condition,raw_flag);
        %system(sprintf(['%s/bin/applyxfm4D %s.gz %s %s %s -singlematrix'], fsldir,fname_beta_est, std_ref,std_beta_est, mat_file));
        
        %fprintf('writing standard space image for %s...\n',condnames{condition});
        %std_beta_ls_est = sprintf('%s/pe%dls_all%s_std.nii',save_pth, ...
        %                         condition,raw_flag);
        %system(sprintf(['%s/bin/applyxfm4D %s.gz %s %s %s -singlematrix'], fsldir, fname_beta_ls_est, std_ref,std_beta_ls_est, mat_file));
        
      %end
      
  
    end
    
  
