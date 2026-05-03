clear all;
close all;
clc;
addpath(genpath('Functions'));

data_dir = '';%Enter data directory

%Load and Visualize Data
load([data_dir, 'DC2P5S1.mat']);
load([data_dir, 'I_downwelling_res.mat']);
load([data_dir, 'attenuation.mat']);
lidar = load([data_dir, 'lidar.mat']);
lidar = lidar.depthMap;
index = find(lidar==0);
lidar(index) = nan;

measurements = measurements(1:256, 901:1156,1:247);
lidar = lidar(1:256, 901:1156);
lambda = load([data_dir, 'lambda.mat']).lambda(1:247);
attenuation = attenuation(1:247);
I_downwelling = I_downwelling_res(1:247, :);
clear I_downwelling_res
%% Finding the corelation between the water vapor and ozone features
water_absorption_bands = [4, 10, 14, 19, 23, 28, 31, 37, 42, 48, 54, 189, 230];
clear_bands = [8, 13, 17, 21, 25, 30, 34, 39, 45, 50, 59, 195, 236];
ozone_bands = 77;
ozone_clear = 81;

cor_coeff = zeros(size(water_absorption_bands));
for i = 1:length(water_absorption_bands)
    Y = I_downwelling(water_absorption_bands(i),:) - I_downwelling(clear_bands(i),:);
    X = I_downwelling(ozone_bands,:) - I_downwelling(ozone_clear,:);
    cor_coeff(i) = inv(X*X')*X*Y';
end
%% One wavelength correlation plot
pair_index = 7;

figure(1);
clf;
set(gcf, 'Position', [100, 100, 700, 925]); % [left, bottom, width, height]

% Define angle labels with correct degree symbols
angles = {'0^{\circ}', '30^{\circ}', '60^{\circ}', '70^{\circ}', '80^{\circ}', ...
          '82^{\circ}', '84^{\circ}', '86^{\circ}', '88^{\circ}', '89^{\circ}'};

% Define color gradient from dark blue to light blue
numCurves = size(I_downwelling, 2); % Number of curves
colors = [linspace(0, 0.5, numCurves)', linspace(0, 0.8, numCurves)', linspace(0.5, 1, numCurves)'];

hold on;
p1 = gobjects(numCurves, 1); % Preallocate for efficiency

% Plot each radiance curve with a unique color
for i = 1:numCurves
    p1(i) = plot(lambda, I_downwelling(:,i), 'Color', colors(i,:), 'LineWidth', 2);
end

% Add vertical lines for absorption bands
p2 = xline(lambda(water_absorption_bands(pair_index)), '-', 'Color', [0, 0, 0], 'LineWidth', 4, 'Alpha', 1);
p3 = xline(lambda(clear_bands(pair_index)), '--', 'Color', [0, 0, 0], 'LineWidth', 4, 'Alpha', 1);
p4 = xline(lambda(ozone_bands), '-', 'Color', [1,0,0], 'LineWidth', 4);
p5 = xline(lambda(ozone_clear), '--', 'Color', [1,0,0], 'LineWidth', 4);

% Axis limits and labels
xlim([min(lambda), max(lambda)]);
xlim([8.5, 10]);
ylim([10, 820]);
xlabel('Wavelength (μm)');
ylabel('Radiance (μW cm^{-2} sr^{-1} μm^{-1})');
set(gca, 'FontSize', 25);

% Define legend entries (combine angle labels + existing ones)
legend_entries = [angles, {'\lambda_1: Water vapor absorption', '\lambda_2: Transparent band', ...
    '\lambda_3: Ozone absorption', '\lambda_4: Transparent band'}];

% Apply the legend
lgd = legend([p1(:); p2; p3; p4; p5], legend_entries, ...
    'Location', 'southoutside', 'NumColumns', 3, 'Interpreter', 'tex'); % Use 'tex' interpreter

hold off;

% Save figures
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_correlation_1', 'epsc');
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_correlation_1', 'jpg');

figure(2);
clf;
set(gcf, 'Position', [100, 100, 700, 925]); % [left, bottom, width, height]

% Compute X and Y
Y = I_downwelling(water_absorption_bands(pair_index),:) - I_downwelling(clear_bands(pair_index),:);
X = I_downwelling(ozone_bands,:) - I_downwelling(ozone_clear,:);

% Define angle labels with correct degree symbols
angles = {'0^{\circ}', '30^{\circ}', '60^{\circ}', '70^{\circ}', '80^{\circ}', ...
          '82^{\circ}', '84^{\circ}', '86^{\circ}', '88^{\circ}', '89^{\circ}'};

hold on;

% Define color gradient from dark blue to light blue
numPoints = length(X);
colors = [linspace(0, 0.5, numPoints)', linspace(0, 0.8, numPoints)', linspace(0.5, 1, numPoints)'];

% Plot each data point separately with a unique color
p1 = gobjects(numPoints, 1); % Preallocate for efficiency
for i = 1:numPoints
    p1(i) = plot(X(i), Y(i), 'o', 'MarkerFaceColor', colors(i,:), 'MarkerEdgeColor', colors(i,:), 'LineWidth', 7);
end

% Plot estimated correlation line
p2 = plot(min(X):max(X), cor_coeff(pair_index) * (min(X):max(X)), '--', 'LineWidth', 3, 'Color', [0, 0, 0]); % Black dashed line

% Labels
xlabel('L_D(\lambda_4) - L_D(\lambda_3)');
ylabel('L_D(\lambda_2) - L_D(\lambda_1)');
xlim([min(X), max(X)]);
set(gca, 'FontSize', 25);

% Define legend with all angles and correlation line
legend_entries = [angles, {'Estimated Correlation'}];

% Apply the legend correctly by assigning each plotted point separately
legend([p1(:); p2], legend_entries, 'Location', "southoutside", 'NumColumns', 3, 'Interpreter', 'tex');

hold off;

% Save figures
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_correlation_2', 'epsc');
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_correlation_2', 'jpg');
%% Depth maps
cmap = colormap('parula');
cmap = [[0,0,0];flip(cmap)];
cut_off_1 = 0;
cut_off_2 = 150;

center_target_1 = [187, 86];   % calibration target 1
center_target_2 = [145, 18];   % calibration target 2
center_tree     = [70, 160];  % tree

line_index = 20;

pair_index = 5;
T_air = 289.7;
index_1 = water_absorption_bands(pair_index);
index_2 = clear_bands(pair_index);
index_3 = ozone_bands;
index_4 = ozone_clear;

d_hat_1 = bispectral_estimation(lambda, measurements, attenuation, index_1, index_2, T_air);
d_hat_2 = quadspectral_estimation(lambda, measurements,...
    attenuation, index_1, index_2, index_3, index_4, T_air, cor_coeff(pair_index) );

figure(101);
imagesc(d_hat_1);
colormap(cmap);
caxis([cut_off_1, cut_off_2]);
cb = colorbar('northoutside');
ylabel(cb, 'Distance (m)', 'FontSize', 15);
set(cb, 'Ticks', [cut_off_1, cut_off_2], 'TickLabels', {[num2str(cut_off_1)], [num2str(cut_off_2)]});
set(gca, 'FontSize', 20)
set(gca, 'Position', get(gca, 'Position') + [0, 0, 0, -0.05]);
xline(line_index, '--', 'Color', [1, 0, 0], 'LineWidth', 2);
axis image off
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_no_downwelling', 'epsc');
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_no_downwelling', 'jpg');

figure(102);
imagesc(d_hat_2);
colormap(cmap);
caxis([cut_off_1, cut_off_2]);
cb = colorbar('northoutside');
ylabel(cb, 'Distance (m)', 'FontSize', 15);
set(cb, 'Ticks', [cut_off_1, cut_off_2], 'TickLabels', {[num2str(cut_off_1)], [num2str(cut_off_2)]});
set(gca, 'FontSize', 20)
set(gca, 'Position', get(gca, 'Position') + [0, 0, 0, -0.05]);
xline(line_index, '--', 'Color', [1, 0, 0], 'LineWidth', 2);
axis image off
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_with_downwelling', 'epsc');
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_with_downwelling', 'jpg');

figure(103);
imagesc(abs(measurements(:, :, ozone_bands) - measurements(:, :, ozone_clear)));
cb = colorbar('northoutside');
caxis([0, 7]);
colormap('gray');
set(cb, 'Ticks', [0, 7]);
ylabel(cb, 'Radiance (μW cm^{-2} sr^{-1} μm^{-1})', 'FontSize', 15);
set(gca, 'FontSize', 20)
set(gca, 'Position', get(gca, 'Position') + [0, 0, 0, -0.05]);
axis image off
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_ozone_feature', 'epsc');
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_ozone_feature', 'jpg');

figure(104)
clf;
imagesc(lidar);
hold on;
text(center_target_1(2) - 5, center_target_1(1), '(f)', ...
    'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold');
text(center_target_2(2) - 5, center_target_2(1), '(g)', ...
    'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold');
text(center_tree(2) - 5, center_tree(1), '(h)', ...
    'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold');
colormap(cmap);
caxis([cut_off_1, cut_off_2]);
cb = colorbar('northoutside');
ylabel(cb, 'Distance (m)', 'FontSize', 15);
set(cb, 'Ticks', [cut_off_1, cut_off_2], 'TickLabels', {[num2str(cut_off_1)], [num2str(cut_off_2)]});
set(gca, 'FontSize', 20)
set(gca, 'Position', get(gca, 'Position') + [0, 0, 0, -0.05]);
xline(line_index, '--', 'Color', [1, 0, 0], 'LineWidth', 2);
axis image off
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_lidar', 'epsc');
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_lidar', 'jpg');

figure(105);
clf;
%set(gcf, 'Position', [100, 100, 3.5*560, 420]); 
set(gcf, 'Position', [100, 100, 3.5*560, 250]); 
plot(d_hat_1(:,line_index), 'LineWidth', 2);
hold on;
plot(d_hat_2(:,line_index), 'LineWidth', 2);
plot(lidar(:,line_index), '--', 'LineWidth', 3, 'Color',[0,0,0]);
xlim([0, 256]);
ylim([0, 200]);
legend('Bispectral (baseline)','Quadspectral', 'Lidar (ground truth)');
xlabel('Pixel Index');
ylabel('Distance (m)');
set(gca, 'FontSize', 15);  
set(gca, 'LineWidth', 1.5);  
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_profile_comparison', 'epsc');
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_profile_comparison', 'jpg');
%% Histograms
patch_size = 8;
half_patch = patch_size / 2;

% Helper function to extract patch
extract_patch = @(M, cp) M(cp(1)-half_patch+1 : cp(1)+half_patch, ...
                           cp(2)-half_patch+1 : cp(2)+half_patch);

% Extract patches (in plotting order)
patch_t1_d1    = extract_patch(d_hat_1, center_target_1);
patch_t1_d2    = extract_patch(d_hat_2, center_target_1);
patch_t1_lidar = extract_patch(lidar,    center_target_1);

patch_t2_d1    = extract_patch(d_hat_1, center_target_2);
patch_t2_d2    = extract_patch(d_hat_2, center_target_2);
patch_t2_lidar = extract_patch(lidar,    center_target_2);

patch_tree_d1    = extract_patch(d_hat_1, center_tree);
patch_tree_d2    = extract_patch(d_hat_2, center_tree);
patch_tree_lidar = extract_patch(lidar,    center_tree);

% Bin settings
all_values = [patch_t1_d1(:); patch_t1_d2(:); patch_t1_lidar(:); ...
              patch_t2_d1(:); patch_t2_d2(:); patch_t2_lidar(:); ...
              patch_tree_d1(:); patch_tree_d2(:); patch_tree_lidar(:)];

bin_width = 5;
bin_min = floor(min(all_values) / bin_width) * bin_width;
bin_max = 200;
bin_edges = bin_min:bin_width:bin_max;

% Colors
color_d1 = [0, 0, 1];   % Blue - Bispectral
color_d2 = [1, 0, 0];   % Red-ish - Quadspectral
color_gt = [0, 0, 0];   % Black - Ground Truth (Lidar)

% --- Figure: Calibration Target 1
figure(201); clf;
%set(gcf, 'Position', [560, 420, 560, 560])
set(gcf, 'Position', [560, 420, 700, 420])
h1 = histogram(patch_t1_d1(:), bin_edges, 'FaceColor', color_d1, 'FaceAlpha', 0.6); hold on;
h2 = histogram(patch_t1_d2(:), bin_edges, 'FaceColor', color_d2, 'FaceAlpha', 0.6);
h3 = histogram(patch_t1_lidar(:), bin_edges, 'FaceColor', color_gt, 'FaceAlpha', 0.6);
legend([h1, h2, h3], 'Bispectral', 'Quadspectral', 'Lidar (GT)');
xlabel('Distance (m)'); ylabel('Count');
set(gca, 'FontSize', 25)
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_histogram_calibration_target_1', 'epsc');
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_histogram_calibration_target_1', 'jpg');

% --- Figure: Calibration Target 2
figure(202); clf;
%set(gcf, 'Position', [560, 420, 560, 560])
set(gcf, 'Position', [560, 420, 700, 420])
h1 = histogram(patch_t2_d1(:), bin_edges, 'FaceColor', color_d1, 'FaceAlpha', 0.6); hold on;
h2 = histogram(patch_t2_d2(:), bin_edges, 'FaceColor', color_d2, 'FaceAlpha', 0.6);
h3 = histogram(patch_t2_lidar(:), bin_edges, 'FaceColor', color_gt, 'FaceAlpha', 0.6);
legend([h1, h2, h3], 'Bispectral', 'Quadspectral', 'Lidar (GT)');
xlabel('Distance (m)'); ylabel('Count');
set(gca, 'FontSize', 25)
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_histogram_calibration_target_2', 'epsc');
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_histogram_calibration_target_2', 'jpg');

% --- Figure: Tree
figure(203); clf;
%set(gcf, 'Position', [560, 420, 560, 560])
set(gcf, 'Position', [560, 420, 700, 420])
h1 = histogram(patch_tree_d1(:), bin_edges, 'FaceColor', color_d1, 'FaceAlpha', 0.6); hold on;
h2 = histogram(patch_tree_d2(:), bin_edges, 'FaceColor', color_d2, 'FaceAlpha', 0.6);
h3 = histogram(patch_tree_lidar(:), bin_edges, 'FaceColor', color_gt, 'FaceAlpha', 0.6);
legend([h1, h2, h3], 'Bispectral', 'Quadspectral', 'Lidar (GT)');
xlabel('Distance (m)'); ylabel('Count');
set(gca, 'FontSize', 25)
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_histogram_tree', 'epsc');
%saveas(gcf, 'Figures/Bispectral_correction/bispectral_histogram_tree', 'jpg');

% Display nanmean and nanstd
fprintf('\n--- Patch Statistics (Mean ± Std, ignoring NaNs) ---\n');

fprintf('\nCalibration Target 1:\n');
fprintf('  Bispectral (Baseline): %.2f ± %.2f m\n', nanmean(patch_t1_d1(:)), nanstd(patch_t1_d1(:)));
fprintf('  Quadspectral:           %.2f ± %.2f m\n', nanmean(patch_t1_d2(:)), nanstd(patch_t1_d2(:)));
fprintf('  Lidar (Ground Truth):   %.2f ± %.2f m\n', nanmean(patch_t1_lidar(:)), nanstd(patch_t1_lidar(:)));

fprintf('\nCalibration Target 2:\n');
fprintf('  Bispectral (Baseline): %.2f ± %.2f m\n', nanmean(patch_t2_d1(:)), nanstd(patch_t2_d1(:)));
fprintf('  Quadspectral:           %.2f ± %.2f m\n', nanmean(patch_t2_d2(:)), nanstd(patch_t2_d2(:)));
fprintf('  Lidar (Ground Truth):   %.2f ± %.2f m\n', nanmean(patch_t2_lidar(:)), nanstd(patch_t2_lidar(:)));

fprintf('\nTree:\n');
fprintf('  Bispectral (Baseline): %.2f ± %.2f m\n', nanmean(patch_tree_d1(:)), nanstd(patch_tree_d1(:)));
fprintf('  Quadspectral:           %.2f ± %.2f m\n', nanmean(patch_tree_d2(:)), nanstd(patch_tree_d2(:)));
fprintf('  Lidar (Ground Truth):   %.2f ± %.2f m\n', nanmean(patch_tree_lidar(:)), nanstd(patch_tree_lidar(:)));