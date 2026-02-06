function rects = helperCreateRectBlockers(centers, normals, widths, heights)
% helperCreateRectBlockers creates one or more rectangular blocker structs
% 
% Inputs:
%   centers - 3xN matrix (each column is a center)
%   normals - 3xN matrix (each column is a normal)
%   widths - 1xN or Nx1 vector (width for each rectangle)
%   heights - 1xN or Nx1 vector (height for each rectangle)
%
% Outputs:
%   rects - 1xN struct array
% 
% Convention: 
%   Width (u) is always in XY plane, perpendicular to wall normal
%   Height (v) is always along Z-axis (vertical)

N = size(centers, 2);
rects = repmat(struct('center', [], 'normal', [], 'u', [], 'v', [], 'width', [], 'height', []), 1, N);

for i = 1:N
    center = centers(:,i);
    normal = normals(:,i) / norm(normals(:,i));
    w = widths(i);
    h = heights(i);
    
    % Height direction is always along Z-axis (vertical)
    v = [0; 0; 1];
    
    % Width direction: perpendicular to normal in XY plane
    % Project normal onto XY plane and rotate 90 degrees
    normal_xy = [normal(1); normal(2); 0];
    
    % Handle case where normal is purely vertical (parallel to Z)
    if norm(normal_xy) < 1e-10
        % If normal is vertical, width can be any horizontal direction
        u = [1; 0; 0];  % Default to X direction
    else
        % Rotate the XY projection by 90 degrees to get width direction
        % Rotation by 90 degrees: [x,y] -> [-y,x]
        u = [-normal_xy(2); normal_xy(1); 0];
        u = u / norm(u);
    end
    
    rects(i).center = center;
    rects(i).normal = normal;
    rects(i).u = u;      % width direction (in XY plane)
    rects(i).v = v;      % height direction (along Z)
    rects(i).width = w;
    rects(i).height = h;
end
end