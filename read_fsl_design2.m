function [fmri]=read_fsl_design2(featdir)
%Read in fsl design.fsf file.  
%Doesn't load design matrix, but loads rest of .fsf info
% Russ Poldrack, 7/2/04 Edited by Jeanette Mumford 10/1/2006

warning off MATLAB:divideByZero

if ~exist('featdir'),
 help read_fsl_design
 return;
end;


dir_loc=fileparts(which('read_fsl_design2'));
SED_CONVERTER=[ dir_loc,'/designconvert2.sed'];

% check that SED_CONVERTER exists
if ~exist(SED_CONVERTER,'file'),
  fprintf('%s does not exist!\n',SED_CONVERTER);
  return;
end;

% design: a cell matrix containing filenames for either SPM.mat file
% (for SPM analysis) or FSL 3-column onset files
% make sure featdir exists and has data in it

current_dir=pwd;
try,
  cd(featdir);
catch,
  fprintf('problem changing to featdir (%s)\n',featdir);
  return;
end;

% convert design.fsf to design.m

[s,w]=system(sprintf('sed -f %s design.fsf > design.m',SED_CONVERTER));
if s,
   fprintf('problem converting design.fsf to design.m\n');
   fprintf('make sure the program sed is in your path\n');
   fprintf(' e.g. click My Computer, Properties, Advanced, Environment Variables\n');
   fprintf('      then cdouble-lick on Path and add ;c:\cygwin\bin to end and press OK \n');
  return;
end;

design_file=which('design');
%fprintf('Loading design from:\n%s\n',design_file);
try,
  design;
catch,
   fprintf('problem loading design.m\n');
  return;
end;

