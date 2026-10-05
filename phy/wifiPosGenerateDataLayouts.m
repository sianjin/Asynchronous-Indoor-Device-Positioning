% Generate datasets with many AP layouts.
%
% Every layout has its own number of APs (3 to 6) at random positions at the
% AP height of the office, and its own random STA positions. Three datasets
% are written with the nominal impairments:
%   data_layouts_train.mat        training layouts
%   data_layouts_test_unseen.mat  layouts that are not in the training set
%   data_layouts_test_seen.mat    training layouts with new STA positions
% and the unseen layouts are also written under four other impairment
% settings, with the same STA positions:
%   data_layouts_test_unseen_<setting>.mat, <setting> = synchronized,
%   clockOnly, phaseOnly, severe
%
% The CIRs of a layout with fewer than the maximum number of APs are padded
% with all-zero APs. The loop over STAs runs on a parallel pool when Parallel
% Computing Toolbox is available. Datasets that already exist are skipped, so
% the script can be restarted.

% Set to true for a quick test of the script (a few minutes). The test writes
% files with the prefix "smoke_", which can be deleted afterwards.
smokeTest = false;

% load office map
mapFileName = "office.stl";

txArraySize = [4 1]; % Linear transmit array
rxArraySize = [4 1]; % Linear receive array
chanBW      = "CBW40";

numAPsRange       = [3 6]; % Minimum and maximum number of APs of a layout
numTrainLayouts   = 500;   % Number of training layouts
numTestLayouts    = 60;    % Number of unseen test layouts
numSeenLayouts    = 60;    % Number of training layouts that are tested with new STA positions
numSTAsPerLayout  = 16;    % STA positions per layout (training and unseen test)
numSTAsSeenLayout = 8;     % New STA positions per training layout in the seen test
snrs = [10 15 20];
maxNumReflections = 1;
filePrefix = "";

if smokeTest
    numTrainLayouts = 6; numTestLayouts = 3; numSeenLayouts = 3;
    numSTAsPerLayout = 4; numSTAsSeenLayout = 2;
    filePrefix = "smoke_";
end

% Impairment settings (as in wifiPosGenerateData.m)
nominal      = struct("ClockOffsetPPM",40, "CFOJitterStd",2,"PhaseNoiseStd",[0.005 0.02],"MaxSpeed",1.5);
synchronized = struct("ClockOffsetPPM",0,  "CFOJitterStd",0,"PhaseNoiseStd",[0 0],       "MaxSpeed",1.5);
clockOnly    = struct("ClockOffsetPPM",40, "CFOJitterStd",2,"PhaseNoiseStd",[0 0],       "MaxSpeed",1.5);
phaseOnly    = struct("ClockOffsetPPM",0,  "CFOJitterStd",0,"PhaseNoiseStd",[0.005 0.02],"MaxSpeed",1.5);
severe       = struct("ClockOffsetPPM",100,"CFOJitterStd",2,"PhaseNoiseStd",[0.02 0.04], "MaxSpeed",1.5);

% Draw all layouts first, so that the seen test uses the training layouts
S = RandStream("mt19937ar","Seed",5489);
RandStream.setGlobalStream(S);
xAP = [0.1 4.9]; % Range of AP positions (m)
yAP = [0.1 7.9];
zAP = 2.1;       % AP height (m)
layouts = cell(1,numTrainLayouts+numTestLayouts);
for k = 1:numel(layouts)
    numAPs = randi(numAPsRange);
    layouts{k} = [xAP(1)+diff(xAP)*rand(1,numAPs); ...
                  yAP(1)+diff(yAP)*rand(1,numAPs); ...
                  zAP*ones(1,numAPs)];
end
trainIdx = 1:numTrainLayouts;
testIdx  = numTrainLayouts+(1:numTestLayouts);
seenIdx  = 1:numSeenLayouts;

pm = propagationModel("raytracing", ...
    "CoordinateSystem","cartesian", ...
    "SurfaceMaterial","wood", ...
    "MaxNumReflections",maxNumReflections);

% Configure
cfg = heRangingConfig('ChannelBandwidth',chanBW, ...
    "NumTransmitAntennas",prod(txArraySize), ...
    "SecureHELTF",false);
cfg.User{1}.NumSpaceTimeStreams = prod(txArraySize);

setup = struct("txArraySize",txArraySize,"rxArraySize",rxArraySize,"snrs",snrs, ...
    "maxNumAPs",numAPsRange(2),"mapFileName",mapFileName);

% Generate and save datasets. Every dataset has its own seeds. The files of
% one call share the layouts, the STA positions and the ray tracing
generateAndSave(filePrefix+"data_layouts_train.mat",{nominal}, ...
    layouts,trainIdx,numSTAsPerLayout,1e5,pm,cfg,setup);
generateAndSave(filePrefix+"data_layouts_test_seen.mat",{nominal}, ...
    layouts,seenIdx,numSTAsSeenLayout,3e5,pm,cfg,setup);
generateAndSave(filePrefix+"data_layouts_test_unseen"+["","_synchronized","_clockOnly","_phaseOnly","_severe"]+".mat", ...
    {nominal,synchronized,clockOnly,phaseOnly,severe}, ...
    layouts,testIdx,numSTAsPerLayout,2e5,pm,cfg,setup);

function generateAndSave(fileNames,conditions,layouts,layoutIdx,numSTAs,baseSeed,pm,cfg,setup)
% Generate one dataset per impairment setting in CONDITIONS and save it under
% the corresponding name in FILENAMES. Files that already exist are skipped.
fileNames = cellstr(fileNames);
todo = find(~cellfun(@isfile,fileNames));
for c = setdiff(1:numel(fileNames),todo)
    disp(['Skipping ',fileNames{c},' (already exists)'])
end
if isempty(todo)
    return
end
disp(['Generating ',strjoin(fileNames(todo),', '),' (',num2str(numel(layoutIdx)),' layouts)...'])
tStart = tic;

maxNumAPs = setup.maxNumAPs;
numSNRs = numel(setup.snrs);
samplesPerLayout = numSTAs*numSNRs;
numSamples = numel(layoutIdx)*samplesPerLayout;

% Labels that are the same for all impairment settings
position = zeros(3,numSamples);
snr = zeros(1,numSamples);
layout = zeros(1,numSamples);
numAPs = zeros(1,numSamples);
apPositions = zeros(3,maxNumAPs,numSamples);
present = false(maxNumAPs,numSamples);
los = false(maxNumAPs,numSamples);

% Outputs that depend on the impairment setting
X = cell(1,numel(fileNames));
detected = cell(1,numel(fileNames));
for c = todo
    detected{c} = false(maxNumAPs,numSamples);
end

for k = 1:numel(layoutIdx)
    apPos = layouts{layoutIdx(k)};
    M = size(apPos,2);

    % Random STA positions of this layout (reproducible for every layout)
    RandStream.setGlobalStream(RandStream("mt19937ar","Seed",baseSeed+k));
    [APs,STAs] = dlPositioningCreateEnvironment(setup.txArraySize,setup.rxArraySize,numSTAs,"random",apPos);

    % Ray tracing for all APs and STAs of the layout
    rays = raytrace(APs,STAs,pm,"Map",setup.mapFileName);
    idx = (k-1)*samplesPerLayout+(1:samplesPerLayout);

    % Packet simulation under every impairment setting
    for c = todo
        [cir,labels] = dlPositioningGenerateDataSet(rays,STAs,APs,cfg,setup.snrs,conditions{c},baseSeed+k);
        if isempty(X{c})
            X{c} = zeros([size(cir,1) size(cir,2) maxNumAPs numSamples],'single');
        end
        X{c}(:,:,1:M,idx) = cir;
        detected{c}(1:M,idx) = labels.detected;
    end

    position(:,idx) = labels.position;
    snr(idx) = labels.snr;
    layout(idx) = layoutIdx(k);
    numAPs(idx) = M;
    apPositions(:,1:M,idx) = repmat(apPos,1,1,samplesPerLayout);
    present(1:M,idx) = true;
    los(1:M,idx) = labels.los;

    if mod(k,ceil(numel(layoutIdx)/20))==0
        disp(['  layout ',num2str(k),' of ',num2str(numel(layoutIdx)),' done, ', ...
            num2str(round(toc(tStart)/60,1)),' minutes elapsed.'])
    end
end

for c = todo
    saveDataSet(fileNames{c},X{c},position,snr,layout,numAPs,apPositions,present,los,detected{c},conditions{c});
    disp(['Saved ',fileNames{c}])
end
disp(['Done in ',num2str(round(toc(tStart)/60,1)),' minutes.'])
end

function saveDataSet(fileName,X,position,snr,layout,numAPs,apPositions,present,los,detected,impairments) %#ok<INUSD>
save(fileName,'X','position','snr','layout','numAPs','apPositions','present','los','detected', ...
    'impairments','-v7')
end
