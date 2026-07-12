clear; clc; close all;

%% ========================================================================
%  WR90: DNN Training — 250k points, noise + correlation, 20 outputs
%  Input  : S21 (real + imag) over 100 frequencies → 200 values
%  Output : eps_real(f1..f10) + eps_img(f1..f10) → 20 values
%
%  Architecture : 200 → 256 → 128 → 64 → 20
%% ========================================================================

%% ========================================================================
%  STEP 1 — Load database
%% ========================================================================

fprintf('Loading database...\n');
load('database_DNN_250k_noise_corr_20out.mat');

fprintf('  X size : %d x %d\n', size(X,1), size(X,2));
fprintf('  Y size : %d x %d\n\n', size(Y,1), size(Y,2));

%% ========================================================================
%  STEP 2 — Split FIRST (70 / 15 / 15)
%% ========================================================================

N      = size(X, 1);
rng(42);
idx    = randperm(N);

nTrain = round(0.70 * N);
nVal   = round(0.15 * N);

idxTrain = idx(1:nTrain);
idxVal   = idx(nTrain+1:nTrain+nVal);
idxTest  = idx(nTrain+nVal+1:end);

XTrain = X(idxTrain, :);   YTrain = Y(idxTrain, :);
XVal   = X(idxVal,   :);   YVal   = Y(idxVal,   :);
XTest  = X(idxTest,  :);   YTest  = Y(idxTest,  :);

fprintf('Data split :\n');
fprintf('  Train : %d points\n',   size(XTrain,1));
fprintf('  Val   : %d points\n',   size(XVal,1));
fprintf('  Test  : %d points\n\n', size(XTest,1));

%% ========================================================================
%  STEP 3 — Normalize AFTER split
%  Separately for eps_real and eps_img
%% ========================================================================

% Normalize X
X_mean = mean(XTrain, 1);
X_std  = std(XTrain,  0, 1);
X_std(X_std == 0) = 1;

XTrain_n = (XTrain - X_mean) ./ X_std;
XVal_n   = (XVal   - X_mean) ./ X_std;
XTest_n  = (XTest  - X_mean) ./ X_std;

% Normalize Y separately for eps_real and eps_img
Y_real_mean = mean(YTrain(:, 1:N_verif), 'all');
Y_real_std  = std(YTrain(:, 1:N_verif),  0, 'all');

Y_img_mean  = mean(YTrain(:, N_verif+1:end), 'all');
Y_img_std   = std(YTrain(:, N_verif+1:end),  0, 'all');

YTrain_n = [(YTrain(:, 1:N_verif)     - Y_real_mean) ./ Y_real_std, ...
            (YTrain(:, N_verif+1:end) - Y_img_mean)  ./ Y_img_std];

YVal_n   = [(YVal(:, 1:N_verif)       - Y_real_mean) ./ Y_real_std, ...
            (YVal(:, N_verif+1:end)   - Y_img_mean)  ./ Y_img_std];

YTest_n  = [(YTest(:, 1:N_verif)      - Y_real_mean) ./ Y_real_std, ...
            (YTest(:, N_verif+1:end)  - Y_img_mean)  ./ Y_img_std];

fprintf('Normalization done.\n\n');

%% ========================================================================
%  STEP 4 — DNN architecture
%  200 → 256 → 128 → 64 → 20
%% ========================================================================

layers = [
    featureInputLayer(200, 'Name', 'input')

    fullyConnectedLayer(256, 'Name', 'fc1')
    reluLayer('Name', 'relu1')

    fullyConnectedLayer(128, 'Name', 'fc2')
    reluLayer('Name', 'relu2')

    fullyConnectedLayer(64, 'Name', 'fc3')
    reluLayer('Name', 'relu3')

    fullyConnectedLayer(20, 'Name', 'fc_out')
    regressionLayer('Name', 'regression')
];

fprintf('DNN Architecture :\n');
fprintf('  Input   : 200 neurons (S21 real+imag sur 100 freq)\n');
fprintf('  Hidden  : 256 → 128 → 64  (ReLU)\n');
fprintf('  Output  : 20 neurons (eps_real x10 + eps_img x10)\n\n');

%% ========================================================================
%  STEP 5 — Training options
%% ========================================================================

opts = trainingOptions('adam', ...
    'MaxEpochs',           300, ...
    'MiniBatchSize',       128, ...
    'InitialLearnRate',    1e-3, ...
    'LearnRateSchedule',   'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 100, ...
    'Shuffle',             'every-epoch', ...
    'ValidationData',      {XVal_n, YVal_n}, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience',  Inf, ...
    'Plots',               'training-progress', ...
    'Verbose',             true, ...
    'ExecutionEnvironment','auto');

%% ========================================================================
%  STEP 6 — Train DNN
%% ========================================================================

fprintf('Training DNN...\n');
tic;
net = trainNetwork(XTrain_n, YTrain_n, layers, opts);
elapsed = toc;
fprintf('Training done in %.2f seconds.\n\n', elapsed);

%% ========================================================================
%  STEP 7 — Test performance
%% ========================================================================

YPred_n = predict(net, XTest_n);

% Denormalize
YPred_real = YPred_n(:, 1:N_verif)      .* Y_real_std + Y_real_mean;
YPred_img  = YPred_n(:, N_verif+1:end)  .* Y_img_std  + Y_img_mean;
YTrue_real = YTest(:, 1:N_verif);
YTrue_img  = YTest(:, N_verif+1:end);

rmse_real = sqrt(mean((YPred_real(:) - YTrue_real(:)).^2));
rmse_img  = sqrt(mean((YPred_img(:)  - YTrue_img(:)).^2));

fprintf('=============================================================\n');
fprintf('  Test Results — DNN 250k noise corr 20 outputs\n');
fprintf('=============================================================\n');
fprintf('  RMSE eps_real = %.6f\n', rmse_real);
fprintf('  RMSE eps_img  = %.6f\n', rmse_img);
fprintf('=============================================================\n\n');

%% ========================================================================
%  STEP 8 — Save
%% ========================================================================

save('DNN_trained_250k_noise_corr_20out.mat', 'net', 'X_mean', 'X_std', ...
    'Y_real_mean', 'Y_real_std', 'Y_img_mean', 'Y_img_std', ...
    'f', 'f_verif', 'idx_verif', 'N_verif');
fprintf('Trained network saved : DNN_trained_250k_noise_corr_20out.mat\n\n');

%% ========================================================================
%  STEP 9 — Plot predicted vs true
%% ========================================================================

figure('Name', 'eps_real predicted vs true', 'NumberTitle', 'off');
plot(YTrue_real(:), YPred_real(:), '.', 'MarkerSize', 3);
hold on;
minY = min([YTrue_real(:); YPred_real(:)]);
maxY = max([YTrue_real(:); YPred_real(:)]);
plot([minY maxY], [minY maxY], 'r-', 'LineWidth', 2);
xlabel('True \epsilon'''); ylabel('Predicted \epsilon''');
title(sprintf('DNN 20out — \\epsilon'' RMSE=%.4f', rmse_real));
legend('Test samples', 'Ideal'); grid on;

figure('Name', 'eps_img predicted vs true', 'NumberTitle', 'off');
plot(YTrue_img(:), YPred_img(:), '.', 'MarkerSize', 3);
hold on;
minY = min([YTrue_img(:); YPred_img(:)]);
maxY = max([YTrue_img(:); YPred_img(:)]);
plot([minY maxY], [minY maxY], 'r-', 'LineWidth', 2);
xlabel('True \epsilon'''''); ylabel('Predicted \epsilon''''');
title(sprintf('DNN 20out — \\epsilon'''' RMSE=%.4f', rmse_img));
legend('Test samples', 'Ideal'); grid on;

fprintf('Done!\n');