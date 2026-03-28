% load office map
mapFileName = "office.stl";
viewer = siteviewer("SceneModel",mapFileName,"Transparency",0.25);

% Set the RNG for reproducibility
S = RandStream("mt19937ar","Seed",5489); 
RandStream.setGlobalStream(S);

txArraySize = [4 1]; % Linear transmit array
rxArraySize = [4 1]; % Linear receive array
chanBW      = "CBW40";

distribution  = "uniform";
staSeparation = 0.5; % STA separation in meters. This parameter is used only when the distribution is uniform.
numSTAs       = 500; % Number of STAs. This parameter is used only when the distribution is random.

if distribution == "uniform"
    [APs,STAs] = dlPositioningCreateEnvironment(txArraySize,rxArraySize,staSeparation,"uniform");
else
    [APs,STAs] = dlPositioningCreateEnvironment(txArraySize,rxArraySize,numSTAs,"random");
end
show(APs)
show(STAs,"ShowAntennaHeight",false,"IconSize",[16 16]);
disp(['Simulating a scenario with ',num2str(width(APs)),' APs and ',num2str(width(STAs)),' STAs with ',char(distribution),' distribution...'])

% Perform ray tracing for all transmitters and receivers in parallel
pm = propagationModel("raytracing", ...
    "CoordinateSystem","cartesian", ...
    "SurfaceMaterial","wood", ...
    "MaxNumReflections",1);
rays = raytrace(APs,STAs,pm,"Map",mapFileName);
size(rays)

% Show ray tracing result
hide(STAs);
show(STAs(ceil(width(STAs)/2)),'IconSize',[32 32]);
plot([rays{:,ceil(width(rays)/2)}],'ColorLimits',[50 95]);

snrs = [10 15 20];

% Configure
cfg = heRangingConfig('ChannelBandwidth',chanBW, ...
    "NumTransmitAntennas",prod(txArraySize), ...
    "SecureHELTF",false);
cfg.User{1}.NumSpaceTimeStreams = prod(txArraySize);

% Generate dataset
[features,labels] = dlPositioningGenerateDataSet(rays,STAs,APs,cfg,snrs);

% Split dataset
[training,validation] = dlPositioningSplitDataSet(features,labels,0.2);

% Save data for python processing, and copy the saved data into the python
% folder
saveDataToPython(training,validation);

%% 
% Copy the python processed data from python folder, and load the python
% processed data
task = "localization"; % "localization" | "positioning"
if task == "localization"
    localizationData = load('localization_classification_baseline_best_20260327_230853.mat');
    category_names = localizationData.category_names(:);
    predicted_positions_indices = localizationData.predicted_classes;
    YPred = categorical(predicted_positions_indices, 1:numel(category_names), category_names);
else % positioning
    positioningData = load('positioning_regression_baseline_best_20260327_230614.mat');
    YPred = positioningData.predicted_positions;
end

metric = dlPositioningPlotResults(mapFileName,validation.Y,YPred,task);


