function dat=read_design_mat(design_file)
%%reads in an fsl design.mat file by creating a design_only.mat file 
% without the fsl header info and reading that into Matlab
%
%dat=read_design_mat(design_file)
%
%dat=design matrix data
%design_file= path to design.mat file of interest (located in .feat dir)


system(['sed ''1,/\Matrix/d'' ', design_file, ' > design_only.mat']);
dat=load('design_only.mat',  '-ascii');
