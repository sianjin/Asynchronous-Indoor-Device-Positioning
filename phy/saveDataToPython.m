function saveDataToPython(training,validation)
% Category names
category_names = {'conference_room';'desk1';'desk2';'desk3';'desk4';'office';'storage'};

% Convert classification category to indices for python processing
training.Y.classification = grp2idx(categorical(training.Y.classification, category_names(:)));
validation.Y.classification = grp2idx(categorical(validation.Y.classification, category_names(:)));

save('data.mat', 'training', 'validation', '-v7')
end