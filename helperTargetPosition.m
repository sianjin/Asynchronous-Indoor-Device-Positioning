function [tgtPos,targetVel] = helperTargetPosition(boundLimit)
% helperTargetPosition configures a target position in a scenario boundary
% 
% Inputs:
%   boundLimit - 3D bound limit of the scenario
%
% Outputs:
%   tgtPos - target position
%   targetVel - target velocity

xTarget = boundLimit(1,1) + (boundLimit(1,2)-boundLimit(1,1))*rand;
yTarget = boundLimit(2,1) + (boundLimit(2,2)-boundLimit(2,1))*rand;
zTarget = boundLimit(3,1) + (boundLimit(3,2)-boundLimit(3,1))*rand;
tgtPos = [xTarget;yTarget;zTarget];
vxTarget = -2 + 4*rand;
vyTarget = -2 + 4*rand;
vzTarget = -2 + 4*rand;
targetVel = [vxTarget;vyTarget;vzTarget];
end