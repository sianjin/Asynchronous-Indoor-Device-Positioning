classdef helperXYPositionHeatmap < handle
    properties
        ROI % [xmin xmax; ymin ymax]
        clims = [10,50];
    end
        
    methods
        function obj = helperXYPositionHeatmap(varargin)
            for i = 1:2:(nargin-1)
                obj.(varargin{i}) = varargin{i+1};
            end
        end

        function [x, y, xyr_mag_matrix] = positionHeatMapData(obj, rar, r, theta, rxPos)
            % Define grid in X-Y plane
            x = obj.ROI(1, 1) : 0.125 : obj.ROI(1, 2);
            y = obj.ROI(2, 1) : 0.125 : obj.ROI(2, 2);
            [X, Y] = meshgrid(x, y);
            S = [X(:) Y(:) zeros(numel(X), 1)].'; % [3 x N] grid points

            % Compute range and angle from rxPos to each grid point
            [rq, aoaq] = rangeangle(S, rxPos);
            thetaq = aoaq(1,:);

            % Prepare for interpolation
            theta = theta * (-1);   % -1 to account for conj
            [thetav, Rv] = meshgrid(theta, r); % [length(r) x length(theta)]

            % Interpolate onto X-Y grid
            xyr = interp2(thetav, Rv, rar, thetaq, rq, 'linear', 0); % fill NaN with 0
            xyr_complex_matrix = reshape(xyr, numel(y), numel(x)); % reshaping to image

            % Extract magnitude in dB
            xyr_mag_matrix = mag2db(abs(xyr_complex_matrix));
        end

        function plot(obj, x, y, xyr_mag_matrix)
            % Display X-Y position heatmap
            imagesc(x, y, xyr_mag_matrix, obj.clims)
            axis on; 
            ax = gca;
            set(ax, 'YDir', 'normal', 'GridColor', [1 1 1], 'GridLineWidth', 1);
            hold on;
            colorbar;
            xlabel('X (m)');
            ylabel('Y (m)');
        end
    end
end