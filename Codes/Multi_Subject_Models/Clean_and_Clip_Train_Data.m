% -----------------------------------------------------
% EEG Cleaning Script for Raw_Training_BCIIV_2a_EEG.gdf (No ICA)
% Extracts 3s Resting + 3s Motor Imagery trials from cues
% -----------------------------------------------------

%% Step 1: Run EEGLAB (only once)
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

for subjnum = 1:9
    %% Step 2: Load the GDF file
    subjectno = num2str(subjnum);
    sessionType = 'T';
    subject = ['A0' subjectno sessionType];
    filename = ['Raw_Dataset/' subject '.gdf'];
    
    EEG = pop_biosig(filename);
    EEG.setname = subject;
    
    %% Step 3: Select only EEG channels (exclude EOG)
    EEG = pop_select(EEG, 'channel', 1:22);
    
    %% Step 4–5: ICA + component rejection skipped
    
    %% Step 6: Detect trial indices to reject based on 1023
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
    fprintf('[%s] Marked %d trials for rejection (1023).\n', subject, length(rejected_trial_indices));
    
    %% Step 7: Epoch full trial from -1 to 6 s around 768
    EEG_full = pop_epoch(EEG, {'768'}, [-1.0, 6.0]);
    
    %% Step 8: Baseline correction (0–3s)
    EEG_full = pop_rmbase(EEG_full, [0 3000]);
    
    %% Step 9: Remove bad trials
    if ~isempty(rejected_trial_indices)
        EEG_full = pop_select(EEG_full, 'notrial', rejected_trial_indices);
        fprintf('[%s] Removed %d bad trials.\n', subject, length(rejected_trial_indices));
    else
        fprintf('[%s] No bad trials to remove.\n', subject);
    end
    
    %% Step 10: Extract resting and MI segments (3s each)
    fs = EEG_full.srate;            % should be 250
    Nsamples = 3 * fs;              % 750 samples = 3s
    
    X_rest = EEG_full.data(:, 1:Nsamples, :);                      % 0–3s
    X_mi   = EEG_full.data(:, Nsamples+251:2*Nsamples+250, :);           % 3–6s
    
    X_rest = permute(X_rest, [3 2 1]);  % [trials × samples × channels]
    X_mi   = permute(X_mi,   [3 2 1]);
    
    %% Step 11: Extract labels from cues
    y_mi = zeros(EEG_full.trials, 1);
    
    for i = 1:EEG_full.trials
        cue_type = EEG_full.epoch(i).eventtype;
        if iscell(cue_type)
            cue_type = cue_type{end};  % in case of cell
        end
        switch cue_type
            case '769', y_mi(i) = 1;  % left hand
            case '770', y_mi(i) = 2;  % right hand
            case '771', y_mi(i) = 3;  % feet
            case '772', y_mi(i) = 4;  % tongue
            otherwise
                error('[%s] Unexpected cue type: %s', subject, cue_type);
        end
    end
    
    y_rest = zeros(size(y_mi));  % resting = 0
    
    %% Step 12: Combine and save
    X = cat(1, X_rest, X_mi);  % [2N × 750 × 22]
    y = cat(1, y_rest, y_mi);  % [2N × 1]
    
    mat_filename = ['New_Dataset/EEG_restMI_split_3s_' subject '.mat'];
    save(mat_filename, 'X', 'y', '-v7.3');
    
    fprintf('[%s] ✅ Saved with Rest+MI split trials.\n', subject);
end
