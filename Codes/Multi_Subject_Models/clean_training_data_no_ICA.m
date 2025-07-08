% -----------------------------------------------------
% EEG Cleaning Script for Raw_Training_BCIIV_2a_EEG.gdf (No ICA)
% Loop over subjects 1 to 9
% -----------------------------------------------------

%% Step 1: Run EEGLAB (run this once)
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

for subjnum = 1:9
    %% Step 2: Load the GDF file
    subjectno = num2str(subjnum);
    sessionType = 'T';
    subject = ['A0' subjectno sessionType];  % You can change this dynamically
    filename = ['../Dataset/' subject '.gdf'];
    
    EEG = pop_biosig(filename);
    EEG.setname = subject;
    
    %% Step 3: Select only EEG channels (exclude EOG)
    EEG = pop_select(EEG, 'channel', 1:22);
    
    %% Step 4: (ICA step removed)
    % Originally: EEG = pop_runica(...)
    
    %% Step 5: (Component rejection step removed)
    % Originally: EEG = pop_subcomp(...)
    
    %% Step 6: Detect trial indices to reject based on edftype sequence
    trial_start_indices = [];
    rejected_trial_indices = [];
    
    for i = 1:length(EEG.event)
        if EEG.event(i).edftype == 768
            trial_start_indices = [trial_start_indices, i];
            if i < length(EEG.event) && EEG.event(i + 1).edftype == 1023
                rejected_trial_indices = [rejected_trial_indices, length(trial_start_indices)];
            end
        end
        EEG.event(i).type = num2str(EEG.event(i).edftype);
    end
    fprintf('[%s] Marked %d trials for rejection (due to 1023 after 768).\n', subject, length(rejected_trial_indices));
    
    %% Step 7: Epoch around motor imagery cues (−0.25 to 4.0 s)
    EEG = pop_epoch(EEG, {'769', '770', '771', '772'}, [-1.0 4.0]);
    
    %% Step 8: Baseline correction using pre-stimulus (−250 to 0 ms)
    EEG = pop_rmbase(EEG, [-250 0]);
    
    %% Step 9: Remove bad trials based on rejected_trial_indices
    if ~isempty(rejected_trial_indices)
        EEG = pop_select(EEG, 'notrial', rejected_trial_indices);
        fprintf('[%s] Removed %d bad trials after epoching.\n', subject, length(rejected_trial_indices));
    else
        fprintf('[%s] No bad trials to remove after epoching.\n', subject);
    end
    
    %% Step 10: Save the cleaned, epoched EEG
    if ~isfield(EEG, 'data')
        error('❌ [%s] EEG struct is invalid — check pop_epoch or pop_rmbase earlier.', subject);
    end
    
    EEG.setname = ['Cleaned_Epoched_EEG_' subject];
    save_filename = ['../Dataset/EEG_python_ready_1063_sample_pnts' subject '.set'];
    EEG = pop_saveset(EEG, 'filename', save_filename);
    
    %% Step 11: Extract data and labels
    X = EEG.data;  % [channels × samples × trials]
    X = permute(X, [3, 2, 1]); % [trials × samples × channels]
    
    event_mapping = containers.Map({'769', '770', '771', '772'}, [1, 2, 3, 4]);
    y = zeros(EEG.trials, 1);
    for i = 1:EEG.trials
        event_type = EEG.epoch(i).eventtype;
        if isKey(event_mapping, event_type)
            y(i) = event_mapping(event_type);
        else
            error('[%s] Unexpected event type: %s', subject, event_type);
        end
    end
    
    %% Step 12: Save data and labels to a .mat file
    mat_filename = ['Dataset/EEG_python_ready_1250_sample_pnts' subject '.mat'];
    save(mat_filename, 'X', 'y', '-v7.3');
    
    fprintf('[%s] Finished and saved.\n', subject);
end
