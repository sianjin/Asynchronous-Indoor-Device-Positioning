function los = helperIsLineOfSight(P1, P2, rects)
% helperIsLineOfSight Determine if two points have line-of-sight given rectangular blockers.
% 
% Inputs:
%   P1, P2 - 3x1 column vectors (positions)
%   rects - array of structs with fields:
%       center (3x1), normal (3x1), u (3x1), v (3x1), width, height
%
% Outputs:
%   los - LOS conditions

% Ensure column vectors
P1 = P1(:);
P2 = P2(:);

los = true; % assume clear line-of-sight unless blocked

for k = 1:numel(rects)
    rect = rects(k);
    C = rect.center(:);
    n = rect.normal(:);  % Don't re-normalize, trust the input
    u = rect.u(:);       % Use the u vector from createRectBlockers
    v = rect.v(:);       % Use the v vector from createRectBlockers
    w = rect.width;
    h = rect.height;
    
    % Line direction vector
    dir = P2 - P1;
    
    % Check if line is parallel to the plane
    denom = dot(dir, n);
    if abs(denom) < 1e-10
        continue; % Line is parallel to plane, no intersection
    end
    
    % Find intersection parameter t
    t = dot(C - P1, n) / denom;
    
    % Check if intersection is within the line segment [P1, P2]
    if t < 0 || t > 1
        continue; % Intersection not on segment
    end
    
    % Calculate intersection point
    I = P1 + t * dir;
    
    % Project intersection point onto rectangle's local coordinates
    du = dot(I - C, u);  % Distance along width direction
    dv = dot(I - C, v);  % Distance along height direction
    
    % Check if intersection point is within rectangle bounds
    if abs(du) <= w/2 && abs(dv) <= h/2
        los = false; % Blocked by this rectangle
        return;
    end
end
end