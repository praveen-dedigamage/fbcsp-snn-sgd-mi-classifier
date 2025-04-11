% -----------------------------------------------------
% ICA Cleaning Script for Raw_Training_BCIIV_2a_EEG.gdf
% -----------------------------------------------------

%% Step 1: Save current working directory
original_dir = pwd;

%% Step 2: Switch to EEGLAB directory and run EEGLAB
cd('C:/Users/USER/Documents/eeglab_current/eeglab2025.0.0');
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

%% Step 3: Return to your data directory
cd(original_dir);

%% Step 4: Load the GDF file
EEG = pop_biosig('A01T.gdf');
EEG.setname = 'A01T';

%% Step 5: Select only EEG channels (exclude EOG)
EEG = pop_select(EEG, 'channel', 1:22);

%% Step 6: Run ICA (use PCA=20 for speed)
EEG = pop_runica(EEG, 'extended', 1, 'pca', 20);

%% Step 7: Reject the first 3 ICA components
EEG = pop_subcomp(EEG, [1 2 3], 0);  % Back-project and clean

%% Step 8: 
% Detect trial indices to reject based on edftype sequence
% Fix event type issue (convert 'edftype' to string 'type')

trial_start_indices = [];
rejected_trial_indices = [];

for i = 1:length(EEG.event)
    if EEG.event(i).edftype == 768
        trial_start_indices = [trial_start_indices, i];
        % Check if next event is '1023'
        if i < length(EEG.event) && EEG.event(i + 1).edftype == 1023
            % Store trial index (in order they appear)
            rejected_trial_indices = [rejected_trial_indices, length(trial_start_indices)];
        end
    end
    EEG.event(i).type = num2str(EEG.event(i).edftype);
end
fprintf('Marked %d trials for rejection (due to 1023 after 768).\n', length(rejected_trial_indices));

%% Step 9: Epoch around motor imagery cues (−0.5 to 4.0 s)
EEG = pop_epoch(EEG, {'769', '770', '771', '772'}, [-0.2 3.0]);

%% Step 10: Baseline correction using pre-stimulus (−200 to 0 ms)
EEG = pop_rmbase(EEG, [-200 0]);

%% Step 11: Remove bad trials based on rejected_trial_indices
if ~isempty(rejected_trial_indices)
    EEG = pop_select(EEG, 'notrial', rejected_trial_indices);
    fprintf('Removed %d bad trials after epoching.\n', length(rejected_trial_indices));
else
    fprintf('No bad trials to remove after epoching.\n');
end

%% Step 12: Save the cleaned, epoched EEG
% Check EEG structure before saving
if ~isfield(EEG, 'data')
    error('❌ EEG struct is invalid — check pop_epoch or pop_rmbase earlier.');
end

% Save the cleaned, epoched EEG
EEG.setname = 'Cleaned_Epoched_EEG_A01T';
EEG = pop_saveset(EEG, 'filename', 'Cleaned_Epoched_EEG_A01T.set');

%% Step 13: Extract data and labels
X = EEG.data;  % Shape: [channels × samples × trials]

% Transpose X to [trials × samples × channels] for Python compatibility
X = permute(X, [3, 2, 1]);

% Extract labels from event types
event_mapping = containers.Map({'769', '770', '771', '772'}, [1, 2, 3, 4]);
y = zeros(EEG.trials, 1);
for i = 1:EEG.trials
    event_type = EEG.epoch(i).eventtype;
    if isKey(event_mapping, event_type)
        y(i) = event_mapping(event_type);
    else
        error('Unexpected event type: %s', event_type);
    end
end

%% Step 14: Save data and labels to a .mat file
save('EEG_python_ready_A01T.mat', 'X', 'y', '-v7.3');
