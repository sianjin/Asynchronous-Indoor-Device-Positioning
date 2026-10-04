% Show the ray-tracing scenario in Site Viewer: the office, the four APs
% used for the datasets, one example STA and the propagation rays between
% them. The APs and the ray-tracing settings are the same as in
% wifiPosGenerateData.m.

% load office map
mapFileName = "office.stl";
viewer = siteviewer("SceneModel",mapFileName,"Transparency",0.25);

txArraySize = [4 1]; % Linear transmit array
rxArraySize = [4 1]; % Linear receive array
maxNumReflections = 1;

% Position of the example STA in meters [x; y; z]
staPosition = [2.5; 4.0; 1.3];

% Create the APs (and one STA, which is then moved to the example position)
[APs,STA] = dlPositioningCreateEnvironment(txArraySize,rxArraySize,1,"random");
STA.AntennaPosition = staPosition;
STA.Name = "STA";

show(APs,"IconSize",[32 32])
show(STA,"ShowAntennaHeight",false,"IconSize",[32 32]);

% Perform ray tracing between all APs and the STA
pm = propagationModel("raytracing", ...
    "CoordinateSystem","cartesian", ...
    "SurfaceMaterial","wood", ...
    "MaxNumReflections",maxNumReflections);
rays = raytrace(APs,STA,pm,"Map",mapFileName);

% Show ray tracing result (color: path loss in dB)
plot([rays{:}],"ColorLimits",[50 95]);

% Report which APs have a line-of-sight path to the STA
for i = 1:numel(APs)
    if isempty(rays{i})
        disp(['AP ',num2str(i),': no path'])
    else
        disp(['AP ',num2str(i),': ',num2str(numel(rays{i})),' rays, line of sight: ', ...
            mat2str(any([rays{i}.LineOfSight]))])
    end
end
