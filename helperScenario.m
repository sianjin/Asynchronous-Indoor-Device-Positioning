function [scattererPos,anchorPos,anchorVel,rectBlockers] = helperScenario(boundLimit,numAnchors,numScatterers)
% helperScenario creates the scatterers, anchors, and blockers position and
% shapes
% 
% Inputs:
%   boundLimit - 3D bound limit of the scenario
%   numAnchors - number of anchors
%   numScatterers - number of scatters
%
% Outputs:
%   scattererPos - scatterer positions
%   anchorPos - anchor positions
%   anchorVel - anchor velocities
%   rectBlockers - rectangle struct array

% Setup scatterers
xScatter = boundLimit(1,1) + (boundLimit(1,2)-boundLimit(1,1))*rand(1,numScatterers);
yScatter = boundLimit(2,1) + (boundLimit(2,2)-boundLimit(2,1))*rand(1,numScatterers);
zScatter = boundLimit(3,1) + (boundLimit(3,2)-boundLimit(3,1))*rand(1,numScatterers);
scattererPos = [xScatter;yScatter;zScatter];

% Add anchors
xAnchor = boundLimit(1,1) + (boundLimit(1,2)-boundLimit(1,1))*rand(1,numAnchors);
yAnchor = boundLimit(2,1) + (boundLimit(2,2)-boundLimit(2,1))*rand(1,numAnchors);
zAnchor = boundLimit(3,1) + (boundLimit(3,2)-boundLimit(3,1))*rand(1,numAnchors);
anchorPos = [xAnchor;yAnchor;zAnchor];
anchorVel = zeros(3,numAnchors);

% Wall blockers
wallCenters = [1.25    -3.5    2.5;         % x-coordinates
    3   1.25   -3.75;                       % y-coordinates  
    1      1     1];                        % z-coordinates
wallNormals = [1     0    -1;               % x-components
    0     1     0;                          % y-components
    0     0     0];                         % z-components
wallWidths = [4, 3, 2];                     % wall lengths
wallHeights = 2*ones(1,3);                  % 2m tall walls
rectBlockers = helperCreateRectBlockers(wallCenters, wallNormals, wallWidths, wallHeights);
end