function helperPlotScenario(rectBlockers,anchorPos,tgtPos,scattererPos,los)
% helperPlotScenario plots a scenario showing positions of anchors, target,
% scatters and blockers
% 
% Inputs:
%   rectBlockers - rectangle struct array
%   anchorPos - anchor positions
%   tgtPos - target position
%   scattererPos - scatterer positions
%   los - LOS conditions

figure; hold on; grid on;
xlabel('X'); ylabel('Y'); zlabel('Z');
anchorlabel = {'Anchor 1', 'Anchor 2', 'Anchor 3', ...
    'Anchor 4', 'Anchor 5', 'Anchor 6'};
view(3);
plot3(anchorPos(1,:), anchorPos(2,:), anchorPos(3,:), 'b^', 'MarkerSize', 10, 'DisplayName', 'Anchors');
plot3(tgtPos(1), tgtPos(2), tgtPos(3), 'o', 'MarkerSize', 12, 'MarkerEdgeColor', 'yellow', 'MarkerFaceColor', 'yellow', 'DisplayName', 'Target');
plot3(scattererPos(1,:), scattererPos(2,:), scattererPos(3,:), 'c*', 'MarkerSize', 10, 'DisplayName', 'Scatterers');
for l = 1:size(anchorPos,2)
    if los(l)
        color = 'g';
    else
        color = 'r';
    end
    fl = plot3([tgtPos(1) anchorPos(1,l)], [tgtPos(2) anchorPos(2,l)], [tgtPos(3) anchorPos(3,l)], ...
        color, 'LineWidth', 2);
    set(fl, 'HandleVisibility', 'off');
    text(anchorPos(1,l), anchorPos(2,l), anchorPos(3,l), anchorlabel{l}, ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
end
for k = 1:numel(rectBlockers)
    helperPlotRectangle3D(rectBlockers(k), 'Wall');
end
legend;
title('Indoor Scenario');
end