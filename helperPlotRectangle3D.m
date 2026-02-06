function helperPlotRectangle3D(rect, label, varargin)
% helperPlotRectangle3D plots a 3D rectangle defined by a rect struct
% 
% Inputs:
%   rect - struct with fields: center, normal, u, v, width, height
%   label - string for the rectangle label (optional)
%   varargin - additional plotting options (Name-Value pairs)
%
% Optional Name-Value pairs:
%   'Color' - face color (default: [0.7 0.7 0.9])
%   'Alpha' - transparency (default: 0.7)
%   'EdgeColor' - edge color (default: 'k')
%   'LineWidth' - edge line width (default: 1.5)
%   'ShowDebug' - show debug info (default: false)

% Default values
if nargin < 2 || isempty(label)
    label = '';
end

% Parse optional arguments
p = inputParser;
addParameter(p, 'Color', [0.7 0.7 0.9], @(x) isnumeric(x) && length(x) == 3);
addParameter(p, 'Alpha', 0.7, @(x) isnumeric(x) && x >= 0 && x <= 1);
addParameter(p, 'EdgeColor', 'k', @(x) ischar(x) || (isnumeric(x) && length(x) == 3));
addParameter(p, 'LineWidth', 1.5, @(x) isnumeric(x) && x > 0);
addParameter(p, 'ShowDebug', false, @islogical);
parse(p, varargin{:});

faceColor = p.Results.Color;
alpha = p.Results.Alpha;
edgeColor = p.Results.EdgeColor;
lineWidth = p.Results.LineWidth;

% Extract rectangle properties
center = rect.center;
u = rect.u;  % width direction vector
v = rect.v;  % height direction vector
w = rect.width;
h = rect.height;

% Calculate the four corners of the rectangle
% The rectangle extends w/2 in both +u and -u directions
% and h/2 in both +v and -v directions
halfWidth = w/2;
halfHeight = h/2;

corner1 = center + halfWidth*u + halfHeight*v;  % top-right
corner2 = center - halfWidth*u + halfHeight*v;  % top-left
corner3 = center - halfWidth*u - halfHeight*v;  % bottom-left
corner4 = center + halfWidth*u - halfHeight*v;  % bottom-right

% Create vertices matrix (each row is a vertex)
vertices = [corner1'; corner2'; corner3'; corner4'];

% Define faces (counterclockwise when viewed from the normal direction)
faces = [1 2 3 4];

% Plot the rectangle
hold on;
fl = patch('Vertices', vertices, 'Faces', faces, ...
      'FaceColor', faceColor, ...
      'EdgeColor', edgeColor, ...
      'LineWidth', lineWidth, ...
      'FaceAlpha', alpha);
set(fl, 'HandleVisibility', 'off');

% Add label if provided
if ~isempty(label)
    % Place label at the center of the rectangle
    text(center(1), center(2), center(3), label, ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'middle', ...
         'FontWeight', 'bold', ...
         'FontSize', 10, ...
         'BackgroundColor', 'white', ...
         'EdgeColor', 'black', ...
         'Margin', 2);
end

end