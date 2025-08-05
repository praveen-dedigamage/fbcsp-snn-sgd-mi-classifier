% -----------------------------------------------------
% EEG Cleaning Script for Raw_Evaluation_BCIIV_2a_EEG.gdf (No ICA)
% Adds resting and MI segments as separate trials
% -----------------------------------------------------

%% Step 1: Run EEGLAB (only once at start)
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

for subjnum = 1:9
    %% Step 2: Load the GDF file
    subjectno = num2str(subjnum);
    sessionType = 'E';  % Use 'T' for training, 'E' for evaluation
    subject = ['A0' subjectno sessionType];
    filename = ['Raw_Dataset/' subject '.gdf'];

    EEG = pop_biosig(filename);
    EEG.setname = subject;

    %% Step 3: Select only EEG channels (exclude EOG)
    EEG = pop_select(EEG, 'channel', 1:22);

    %% Step 4 & 5: Skip ICA and component rejection

    %% Step 6: Detect trial start and rejected trials
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

    %% Step 7: Epoch full trial from -1 to 6s
    EEG_full = pop_epoch(EEG, {'768'}, [-1.0, 6.0]);

    %% Step 8: Baseline correction from 0–3s (resting)
    EEG_full = pop_rmbase(EEG_full, [0 3000]);

    %% Step 9: Remove bad trials
    if ~isempty(rejected_trial_indices)
        EEG_full = pop_select(EEG_full, 'notrial', rejected_trial_indices);
        fprintf('[%s] Removed %d bad trials after epoching.\n', subject, length(rejected_trial_indices));
    else
        fprintf('[%s] No bad trials to remove after epoching.\n', subject);
    end

    %% Step 10: Split into resting (-1–2s) and MI (2–6s)
    fs = EEG_full.srate;           % Should be 250 Hz
    Nsamples = 3 * fs;             % 750 samples = 3s

    X_rest = EEG_full.data(:, 1:Nsamples, :);                      % 0–3s
    X_mi   = EEG_full.data(:, Nsamples+251:2*Nsamples+250, :);           % 3–6s

    % Convert to [trials × samples × channels]
    X_rest = permute(X_rest, [3 2 1]);
    X_mi   = permute(X_mi,   [3 2 1]);

    %% Step 11: Load MI labels and create resting labels
    label_filename = ['Raw_Dataset/' subject '_L.mat'];
    y_mi = load(label_filename).classlabel;
    y_mi(rejected_trial_indices) = [];  % Remove bad trial labels
    y_rest = zeros(size(y_mi));         % Resting trials are labeled 0

    %% Step 12: Combine and save
    X = cat(1, X_rest, X_mi);   % [2N × 750 × 22]
    y = cat(1, y_rest, y_mi);   % [2N × 1]

    mat_filename = ['New_Dataset/EEG_restMI_split_3s_' subject '.mat'];
    save(mat_filename, 'X', 'y', '-v7.3');

    fprintf('[%s] Done. Saved resting + MI trials.\n', subject);
end
