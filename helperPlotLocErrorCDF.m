function helperPlotLocErrorCDF(loc_error)
% helperPlotLocErrorCDF plots empirical localization error CDF
% 
% Inputs:
%   loc_error - vector of localization error

loc_error_sort = sort(loc_error);
numTest = length(loc_error);
figure
plot([0,loc_error_sort], [0,(1:numTest) / (numTest)]);
xlabel('Localization RMSE Error (m)');
ylabel('Cumulative Probability');
title('Empirical CDF');
grid on;
end