% -----------------------------------------------------
% EEG Cleaning Script for Raw_Evaluation_BCIIV_2a_EEG.gdf (No ICA)
% -----------------------------------------------------

%% Step 1: Run EEGLAB
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

%% Step 2: Load the GDF file
subjectno = '1';
sessionType = 'E';
subject = ['A0' subjectno sessionType];   % You can change this dynamically
filename = ['../Dataset/' subject '.gdf'];

EEG = pop_biosig(filename);
EEG.setname = subject;

%% Step 3: Select only EEG channels (exclude EOG)
EEG = pop_select(EEG, 'channel', 1:22);

%% Step 4: (ICA step removed)
% Originally: EEG = pop_runica(...)
% Removed ICA to retain raw EEG signals.

%% Step 5: (Component rejection step removed)
% Originally: EEG = pop_subcomp(...)
% Removed component rejection.

%% Step 6: 
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

%% Step 7: Epoch around motor imagery cues (2.8 to 6.0 s)
EEG = pop_epoch(EEG, {'768'}, [2.8, 6.0]);

%% Step 8: Baseline correction using pre-stimulus (2.8 to 3 s)
EEG = pop_rmbase(EEG, [2800 3000]);

%% Step 9: Remove bad trials based on rejected_trial_indices
if ~isempty(rejected_trial_indices)
    EEG = pop_select(EEG, 'notrial', rejected_trial_indices);
    fprintf('Removed %d bad trials after epoching.\n', length(rejected_trial_indices));
else
    fprintf('No bad trials to remove after epoching.\n');
end

%% Step 10: Save the cleaned, epoched EEG
% Check EEG structure before saving
if ~isfield(EEG, 'data')
    error('❌ EEG struct is invalid — check pop_epoch or pop_rmbase earlier.');
end

% Save the cleaned, epoched EEG
EEG.setname = ['Cleaned_Epoched_EEG_' subject];
save_filename = ['../Dataset/Cleaned_Epoched_EEG_without_ICA' subject '.set'];
EEG = pop_saveset(EEG, 'filename', save_filename);

%% Step 11: Extract data and labels
X = EEG.data;  % Shape: [channels × samples × trials]

% Transpose X to [trials × samples × channels] for Python compatibility
X = permute(X, [3, 2, 1]);

% Extract labels from event types
filename = ['../Dataset/' subject '_L.mat'];
y = load(filename).classlabel;
y(rejected_trial_indices) = [];

%% Step 12: Save data and labels to a .mat file
filename = ['../Dataset/EEG_python_ready_without_ICA_' subject '.mat'];
save(filename, 'X', 'y', '-v7.3');
