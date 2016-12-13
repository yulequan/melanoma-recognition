%% WRITING TO HDF5
filename='trial.h5';

num_total_samples=1000;
% to simulate data being read from disk / generated etc.
data_disk=rand(2,3,4,5,num_total_samples); 
label_disk=ones(2,3,4,1,num_total_samples); 

chunksz=100;
created_flag=false;
totalct=0;
for batchno=1:num_total_samples/chunksz
  fprintf('batch no. %d\n', batchno);
  last_read=(batchno-1)*chunksz;

  % to simulate maximum data to be held in memory before dumping to hdf5 file 
  batchdata=data_disk(:,:,:,:,last_read+1:last_read+chunksz); 
  batchlabs=label_disk(:,:,:,:,last_read+1:last_read+chunksz);

  % store to hdf5
  startloc=struct('dat',[1,1,1,1,totalct+1], 'lab', [1,1,1,1,totalct+1]);
  curr_dat_sz=store2hdf5(filename, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
  created_flag=true;% flag set so that file is created only once
  totalct=curr_dat_sz(end);% updated dataset size (#samples)
end

% display structure of the stored HDF5 file
h5disp(filename);

%% READING FROM HDF5

% Read data and labels for samples #100 to 199
data_rd=h5read(filename, '/data', [1 1 1 1 100], [2, 3, 4, 5, 100]);
label_rd=h5read(filename, '/label', [1 1 1 1 100], [2,3,4,1,100]);
fprintf('Testing ...\n');
try 
  assert(isequal(data_rd, single(data_disk(:,:,:,:,100:199))), 'Data do not match');
  assert(isequal(label_rd, single(label_disk(:,:,:,:,100:199))), 'Labels do not match');

  fprintf('Success!\n');
catch err
  fprintf('Test failed ...\n');
  getReport(err)
end

%delete(filename);

% CREATE list.txt containing filename, to be used as source for HDF5_DATA_LAYER
FILE=fopen('list.txt', 'w');
fprintf(FILE, '%s', filename);
fclose(FILE);
fprintf('HDF5 filename listed in %s \n', 'list.txt');

% NOTE: In net definition prototxt, use list.txt as input to HDF5_DATA as: 
% layer {
%   name: "data"
%   type: "HDF5Data"
%   top: "data"
%   top: "labelvec"
%   hdf5_data_param {
%     source: "/path/to/list.txt"
%     batch_size: 64
%   }
% }
