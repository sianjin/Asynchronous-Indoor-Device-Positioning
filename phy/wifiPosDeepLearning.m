% load office map
mapFileName = "office.stl";
viewer = siteviewer("SceneModel",mapFileName,"Transparency",0.25);

% Set the RNG for reproducibility
S = RandStream("mt19937ar","Seed",5489); 
RandStream.setGlobalStream(S);

txArraySize = [4 1]; % Linear receive array
rxArraySize = [2 1]; % Linear receive array
chanBW      = "CBW40";

distribution  = "uniform";
staSeparation = 0.5; % STA separation in meters. This parameter is used only when the distribution is uniform.
numSTAs       = 500; % Number of STAs. This parameter is used only when the distribution is random.

task = "localization"; % | "positioning";

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

% Show ray tracing result
hide(STAs);
show(STAs(ceil(width(STAs)/2)),'IconSize',[32 32]);
plot([rays{:,ceil(width(rays)/2)}],'ColorLimits',[50 95]);

snrs = [10 15 20];

% Configure
generateData = false;
if generateData
cfg = heRangingConfig('ChannelBandwidth',chanBW, ...
    "NumTransmitAntennas",prod(txArraySize), ...
    "SecureHELTF",false);
cfg.User{1}.NumSpaceTimeStreams = prod(rxArraySize);

% Generate dataset
[features,labels] = dlPositioningGenerateDataSet(rays,STAs,APs,cfg,snrs);

% Split dataset
[training,validation] = dlPositioningSplitDataSet(features,labels,0.2);
else
    load("data.mat");
end
% Layers
layers = [

imageInputLayer(size(training.X, 1:3))

convolution2dLayer(3,256,"Padding","same")
batchNormalizationLayer
reluLayer

averagePooling2dLayer(2,"Stride",2,"Padding","same")

convolution2dLayer(3,256,"Padding","same")
batchNormalizationLayer
reluLayer

averagePooling2dLayer(2,"Stride",2,"Padding","same")

convolution2dLayer(3,256,"Padding","same")
batchNormalizationLayer
reluLayer

averagePooling2dLayer(2,"Stride",2,"Padding","same")

convolution2dLayer(3,256,"Padding","same")
batchNormalizationLayer
reluLayer

averagePooling2dLayer(2,"Stride",2,"Padding","same")

dropoutLayer(0.2)];

% Task-Specific Output Head
if task == "localization" % Room/Area Classification
    % Assuming 7 classes (e.g., 7 different rooms or zones)
    layers = [
        layers
        fullyConnectedLayer(7)
        softmaxLayer];
    lossFcn = "crossentropy";
    trainingMetric = "accuracy";

else % Positioning (Coordinate Regression)
    % Outputs [x, y, z] coordinates
    layers = [
        layers
        fullyConnectedLayer(3)];
    lossFcn = "mse";
    trainingMetric = "rmse";
end

if task == "localization" % classification
    valY = validation.Y.classification;
    trainY = training.Y.classification;
else % positioning (regression)
    valY = validation.Y.regression;
    trainY = training.Y.regression;
end

miniBatchSize = 256;

validationFrequency = floor(size(training.X,4)/miniBatchSize);

options = trainingOptions("adam", ...
    "MiniBatchSize",miniBatchSize, ...
    "MaxEpochs",100, ...
    "InitialLearnRate",1e-4,...
    "Metrics",trainingMetric,...
    "Shuffle","every-epoch", ...
    "ValidationData",{validation.X,valY'}, ...
    "ValidationFrequency",validationFrequency, ...
    "Verbose",true, ...
    "ResetInputNormalization",true, ...
    "ExecutionEnvironment","auto");

% Train network
net = trainnet(training.X,trainY.',layers,lossFcn,options);

if task == "localization"
    YScores = minibatchpredict(net,validation.X);
    YPred = scores2label(YScores,categories(labels.class));
else % positioning
    YPred = minibatchpredict(net,validation.X);
end

metric = dlPositioningPlotResults(mapFileName,validation.Y,YPred,task);

