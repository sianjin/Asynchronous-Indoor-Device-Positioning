% Generate the training and test datasets.
%
%   - STAs are placed at random positions, and the test positions are drawn
%     separately from the training positions (no position is shared).
%   - Impairments are drawn independently for every sample.
%   - The SNR, line-of-sight flag and detection flag of every sample are saved.
%   - The loop over STAs runs on a parallel pool when Parallel Computing
%     Toolbox is available. The ray-tracing result is cached in rays_*.mat
%     and finished datasets are skipped, so the script can be restarted.
%   - The test positions are simulated under several impairment conditions.
%
% Output: one data_<set>_<condition>.mat file per dataset, to be copied
% into the python folder.

% load office map
mapFileName = "office.stl";

% Set the RNG for reproducibility
S = RandStream("mt19937ar","Seed",5489);
RandStream.setGlobalStream(S);

txArraySize = [4 1]; % Linear transmit array
rxArraySize = [4 1]; % Linear receive array
chanBW      = "CBW40";

numTrainSTAs = 3000; % Number of training positions
numTestSTAs  = 500;  % Number of held-out test positions
snrs = [10 15 20];
maxNumReflections = 1;

% Impairment conditions. The nominal clock offset is the worst case allowed
% by IEEE 802.11 between two devices (+-20 ppm each).
nominal      = struct("ClockOffsetPPM",40, "CFOJitterStd",2,"PhaseNoiseStd",[0.005 0.02],"MaxSpeed",1.5);
synchronized = struct("ClockOffsetPPM",0,  "CFOJitterStd",0,"PhaseNoiseStd",[0 0],       "MaxSpeed",1.5);
clockOnly    = struct("ClockOffsetPPM",40, "CFOJitterStd",2,"PhaseNoiseStd",[0 0],       "MaxSpeed",1.5);
phaseOnly    = struct("ClockOffsetPPM",0,  "CFOJitterStd",0,"PhaseNoiseStd",[0.005 0.02],"MaxSpeed",1.5);
severe       = struct("ClockOffsetPPM",100,"CFOJitterStd",2,"PhaseNoiseStd",[0.02 0.04], "MaxSpeed",1.5);

trainConditions = struct("nominal",nominal,"synchronized",synchronized);
testConditions  = struct("nominal",nominal,"synchronized",synchronized, ...
    "clockOnly",clockOnly,"phaseOnly",phaseOnly,"severe",severe);

% Create environment and perform ray tracing for all transmitters and
% receivers. The result is cached, so a restart does not repeat it
raysFileName = sprintf("rays_%d_%d.mat",numTrainSTAs,numTestSTAs);
if isfile(raysFileName)
    load(raysFileName,"APs","trainSTAs","testSTAs","trainRays","testRays");
else
    % The APs are identical in both calls
    [APs,trainSTAs] = dlPositioningCreateEnvironment(txArraySize,rxArraySize,numTrainSTAs,"random");
    [~,testSTAs]    = dlPositioningCreateEnvironment(txArraySize,rxArraySize,numTestSTAs,"random");

    pm = propagationModel("raytracing", ...
        "CoordinateSystem","cartesian", ...
        "SurfaceMaterial","wood", ...
        "MaxNumReflections",maxNumReflections);
    trainRays = raytrace(APs,trainSTAs,pm,"Map",mapFileName);
    testRays  = raytrace(APs,testSTAs,pm,"Map",mapFileName);
    save(raysFileName,"APs","trainSTAs","testSTAs","trainRays","testRays");
end
apPositions = [APs.AntennaPosition];

% Configure
cfg = heRangingConfig('ChannelBandwidth',chanBW, ...
    "NumTransmitAntennas",prod(txArraySize), ...
    "SecureHELTF",false);
cfg.User{1}.NumSpaceTimeStreams = prod(txArraySize);

% Generate and save datasets. Every dataset has its own seed, and datasets
% that already exist are skipped, so the script can be restarted
generateAndSave("train",trainConditions,trainRays,trainSTAs,APs,cfg,snrs,apPositions,100);
generateAndSave("test",testConditions,testRays,testSTAs,APs,cfg,snrs,apPositions,200);

function generateAndSave(setName,conditions,rays,STAs,APs,cfg,snrs,apPositions,baseSeed)
category_names = {'conference_room';'desk1';'desk2';'desk3';'desk4';'office';'storage'};
setName = char(setName);
names = fieldnames(conditions);
for c = 1:numel(names)
    fileName = ['data_',setName,'_',names{c},'.mat'];
    if isfile(fileName)
        disp(['Skipping ',fileName,' (already exists)'])
        continue
    end
    impairments = conditions.(names{c});
    disp(['Generating ',setName,' set, condition ',names{c},'...'])
    tStart = tic;
    [X,labels] = dlPositioningGenerateDataSet(rays,STAs,APs,cfg,snrs,impairments,baseSeed+c);
    disp(['Done in ',num2str(round(toc(tStart)/60,1)),' minutes.'])

    position = labels.position;
    classification = double(categorical(labels.class(:), category_names(:)));
    snr = labels.snr;
    los = labels.los;
    detected = labels.detected;
    clockOffsetPPM = labels.clockOffsetPPM;
    phaseNoiseStd = labels.phaseNoiseStd;

    save(fileName,'X','position','classification','snr','los','detected', ...
        'clockOffsetPPM','phaseNoiseStd','impairments','apPositions','-v7')
end
end
