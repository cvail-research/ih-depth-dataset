clear all;
close all;
clc;

addpath(genpath('Functions'));
data_dir = '';%Enter data directory
results_dir = '';%Enter results directory

K = 247;
Q = 10;

no_downwelling = load([results_dir, 'no_downwelling.mat']);
with_downwelling = load([results_dir, 'downwelling.mat']);
with_downwelling_reg = load([results_dir, 'downwelling_reg.mat']);

index = find(no_downwelling.d==0);
no_downwelling.d(index) = nan;
index = find(with_downwelling.d==0);
with_downwelling.d(index) = nan;
index = find(with_downwelling_reg.d==0);
with_downwelling_reg.d(index) = nan;

no_downwelling.V = no_downwelling.V(:,:,:,1:Q);
with_downwelling.V = with_downwelling.V(:,:,:,1:Q);
with_downwelling_reg.V = with_downwelling_reg.V(:,:,:,1:Q);

lambda = load([data_dir, 'lambda.mat']).lambda(1:K);
meas = load([data_dir, 'DC2P5S1.mat']).measurements(1:256, 901:1156,1:K);
attenuation = load([data_dir, 'attenuation.mat']).attenuation(1:K);
reflected = load([data_dir, 'I_downwelling_res.mat']).I_downwelling_res(1:K,:);
lidar = load([data_dir, 'lidar.mat']);
lidar = lidar.depthMap(1:256, 901:1156);
index = find(lidar==0);
lidar(index) = nan;

lambda = reshape(lambda, [1, 1, K]);
attenuation = reshape(attenuation, [1, 1, K]);
reflected = reshape(reflected, [1, 1, K, Q]);
%% Plot Ranging Results
center_target_1 = [187, 86];   % calibration target 1
center_target_2 = [145, 18];   % calibration target 2
center_tree     = [70, 160];  % tree

line_index = 20;

cut_off_1 = 0;
cut_off_2 = 150;

% No downwelling
figure(1);
cmap = colormap('parula');
cmap = [flip(cmap)];
clf;
imagesc(no_downwelling.d(:, :));
colormap(cmap);
clim([cut_off_1, cut_off_2]);
cb = colorbar('northoutside');
ylabel(cb, 'Distance (m)', 'FontSize', 15);
set(cb, 'Ticks', [cut_off_1, cut_off_2], 'TickLabels', {[num2str(cut_off_1)], [num2str(cut_off_2)]});
set(gca, 'FontSize', 20)
set(gca, 'Position', get(gca, 'Position') + [0, 0, 0, -0.05]);
xline(line_index, '--', 'Color', [1, 0, 0], 'LineWidth', 2);
axis image off
%saveas(gcf, 'Figures/Hyperspectral_correction/Range/hyperspectral_no_downwelling_range', 'epsc');
%saveas(gcf, 'Figures/Hyperspectral_correction/Range/hyperspectral_no_downwelling_range', 'jpg');

figure(2)
clf;
imagesc(with_downwelling.d(:, :));
colormap(cmap);
clim([cut_off_1, cut_off_2]);
cb = colorbar('northoutside');
ylabel(cb, 'Distance (m)', 'FontSize', 15);
set(cb, 'Ticks', [cut_off_1, cut_off_2], 'TickLabels', {[num2str(cut_off_1)], [num2str(cut_off_2)]});
set(gca, 'FontSize', 20)
set(gca, 'Position', get(gca, 'Position') + [0, 0, 0, -0.05]);
xline(line_index, '--', 'Color', [1, 0, 0], 'LineWidth', 2);
axis image off
%saveas(gcf, 'Figures/Hyperspectral_correction/Range/hyperspectral_with_downwelling_range', 'epsc');
%saveas(gcf, 'Figures/Hyperspectral_correction/Range/hyperspectral_with_downwelling_range', 'jpg');

figure(3)
clf;
imagesc(with_downwelling_reg.d(:, :));
colormap(cmap);
clim([cut_off_1, cut_off_2]);
cb = colorbar('northoutside');
ylabel(cb, 'Distance (m)', 'FontSize', 15);
set(cb, 'Ticks', [cut_off_1, cut_off_2], 'TickLabels', {[num2str(cut_off_1)], [num2str(cut_off_2)]});
set(gca, 'FontSize', 20)
set(gca, 'Position', get(gca, 'Position') + [0, 0, 0, -0.05]);
xline(line_index, '--', 'Color', [1, 0, 0], 'LineWidth', 2);
axis image off
%saveas(gcf, 'Figures/Hyperspectral_correction/Range/hyperspectral_with_downwelling_TV', 'epsc');
%saveas(gcf, 'Figures/Hyperspectral_correction/Range/hyperspectral_with_downwelling_TV', 'jpg');


figure(4)
cmap = colormap('parula');
cmap = [[0,0,0];flip(cmap)];
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
clim([cut_off_1, cut_off_2]);
cb = colorbar('northoutside');
ylabel(cb, 'Distance (m)', 'FontSize', 15);
set(cb, 'Ticks', [cut_off_1, cut_off_2], 'TickLabels', {[num2str(cut_off_1)], [num2str(cut_off_2)]});
set(gca, 'FontSize', 20)
set(gca, 'Position', get(gca, 'Position') + [0, 0, 0, -0.05]);
xline(line_index, '--', 'Color', [1, 0, 0], 'LineWidth', 2);
axis image off
%saveas(gcf, 'Figures/Hyperspectral_correction/Range/hyperspectral_lidar', 'epsc');
%saveas(gcf, 'Figures/Hyperspectral_correction/Range/hyperspectral_lidar', 'jpg');

figure(5);
%set(gcf, 'Position', [100, 100, 3.5*560, 420]); 
set(gcf, 'Position', [100, 100, 3.5*560, 250]); 
clf;
plot(no_downwelling.d(:,line_index), 'LineWidth', 2);
hold on;
plot(with_downwelling.d(:,line_index), 'LineWidth', 2);
plot(with_downwelling_reg.d(:,line_index), 'LineWidth', 2);
plot(lidar(:,line_index), '--', 'LineWidth', 3, 'Color',[0,0,0]);
ylim([15,200]);
xlim([0,256]);
xlabel('Pixel Index', 'FontSize', 20);
ylabel('Distance (m)', 'FontSize', 20);  
set(gca, 'LineWidth', 1.5);  
lgd = legend('Without downwelling correction','Downwelling correction','Downwelling correction (TV)', 'Lidar (ground truth)', 'Location', 'northeast', 'NumColumns', 2);
%set(gca, 'FontSize', 32);
set(gca, 'FontSize', 15);
%saveas(gcf, 'Figures/Hyperspectral_correction/Range/hyperspectral_profile_comparison_TV', 'epsc');
%saveas(gcf, 'Figures/Hyperspectral_correction/Range/hyperspectral_profile_comparison_TV', 'jpg');
%% Histograms
patch_size = 8;
half_patch = patch_size / 2;

% Helper function to extract patch
extract_patch = @(M, cp) M(cp(1)-half_patch+1 : cp(1)+half_patch, ...
                           cp(2)-half_patch+1 : cp(2)+half_patch);

% Extract patches (in plotting order)
patch_t1_d1    = extract_patch(no_downwelling.d, center_target_1);
patch_t1_d2    = extract_patch(with_downwelling.d, center_target_1);
patch_t1_tv    = extract_patch(with_downwelling_reg.d, center_target_1);
patch_t1_lidar = extract_patch(lidar, center_target_1);

patch_t2_d1    = extract_patch(no_downwelling.d, center_target_2);
patch_t2_d2    = extract_patch(with_downwelling.d, center_target_2);
patch_t2_tv    = extract_patch(with_downwelling_reg.d, center_target_2);
patch_t2_lidar = extract_patch(lidar, center_target_2);

patch_tree_d1    = extract_patch(no_downwelling.d, center_tree);
patch_tree_d2    = extract_patch(with_downwelling.d, center_tree);
patch_tree_tv    = extract_patch(with_downwelling_reg.d, center_tree);
patch_tree_lidar = extract_patch(lidar, center_tree);

% Bin settings
all_values = [patch_t1_d1(:); patch_t1_d2(:); patch_t1_tv(:); patch_t1_lidar(:); ...
              patch_t2_d1(:); patch_t2_d2(:); patch_t2_tv(:); patch_t2_lidar(:); ...
              patch_tree_d1(:); patch_tree_d2(:); patch_tree_tv(:); patch_tree_lidar(:)];

bin_width = 5;
bin_min = floor(min(all_values) / bin_width) * bin_width;
bin_max = 200;
bin_edges = bin_min:bin_width:bin_max;

% Colors
color_d1 = [0, 0, 1];       % Blue - Without downwelling correction
color_d2 = [1, 0, 0];       % Red - Downwelling correction
color_tv = [1, 1, 0];       % Yellow - Downwelling correction (TV)
color_gt = [0, 0, 0];       % Black - Lidar (Ground Truth)

% --- Figure: Calibration Target 1
figure(101); clf;
%set(gcf, 'Position', [560, 420, 560, 560])
set(gcf, 'Position', [560, 420, 700, 420])
h1 = histogram(patch_t1_d1(:), bin_edges, 'FaceColor', color_d1, 'FaceAlpha', 0.6); hold on;
h2 = histogram(patch_t1_d2(:), bin_edges, 'FaceColor', color_d2, 'FaceAlpha', 0.6);
h3 = histogram(patch_t1_tv(:), bin_edges, 'FaceColor', color_tv, 'FaceAlpha', 0.6);
h4 = histogram(patch_t1_lidar(:), bin_edges, 'FaceColor', color_gt, 'FaceAlpha', 0.6);
legend([h1, h2, h3, h4], ...
    'w/o down. corr.', ...
    'w/ down. corr.', ...
    'w/ down. corr. (TV)', ...
    'Lidar (GT)');
xlabel('Distance (m)'); ylabel('Count');
set(gca, 'FontSize', 25)
%saveas(gcf, 'Figures/Hyperspectral_correction/hyperspectral_histogram_calibration_target_1', 'epsc');
%saveas(gcf, 'Figures/Hyperspectral_correction/hyperspectral_histogram_calibration_target_1', 'jpg');

% --- Figure: Calibration Target 2
figure(102); clf;
%set(gcf, 'Position', [560, 420, 560, 560])
set(gcf, 'Position', [560, 420, 700, 420])
h1 = histogram(patch_t2_d1(:), bin_edges, 'FaceColor', color_d1, 'FaceAlpha', 0.6); hold on;
h2 = histogram(patch_t2_d2(:), bin_edges, 'FaceColor', color_d2, 'FaceAlpha', 0.6);
h3 = histogram(patch_t2_tv(:), bin_edges, 'FaceColor', color_tv, 'FaceAlpha', 0.6);
h4 = histogram(patch_t2_lidar(:), bin_edges, 'FaceColor', color_gt, 'FaceAlpha', 0.6);
legend([h1, h2, h3, h4], ...
    'w/o down. corr.', ...
    'w/ down. corr.', ...
    'w/ down. corr. (TV)', ...
    'Lidar (GT)');
xlabel('Distance (m)'); ylabel('Count');
set(gca, 'FontSize', 25)
%saveas(gcf, 'Figures/Hyperspectral_correction/hyperspectral_histogram_calibration_target_2', 'epsc');
%saveas(gcf, 'Figures/Hyperspectral_correction/hyperspectral_histogram_calibration_target_2', 'jpg');

% --- Figure: Tree
figure(103); clf;
%set(gcf, 'Position', [560, 420, 560, 560])
set(gcf, 'Position', [560, 420, 700, 420])
h1 = histogram(patch_tree_d1(:), bin_edges, 'FaceColor', color_d1, 'FaceAlpha', 0.6); hold on;
h2 = histogram(patch_tree_d2(:), bin_edges, 'FaceColor', color_d2, 'FaceAlpha', 0.6);
h3 = histogram(patch_tree_tv(:), bin_edges, 'FaceColor', color_tv, 'FaceAlpha', 0.6);
h4 = histogram(patch_tree_lidar(:), bin_edges, 'FaceColor', color_gt, 'FaceAlpha', 0.6);
legend([h1, h2, h3, h4], ...
    'w/o down. corr.', ...
    'w/ down. corr.', ...
    'w/ down. corr. (TV)', ...
    'Lidar (GT)');
xlabel('Distance (m)'); ylabel('Count');
set(gca, 'FontSize', 25)
%saveas(gcf, 'Figures/Hyperspectral_correction/hyperspectral_histogram_tree', 'epsc');
%saveas(gcf, 'Figures/Hyperspectral_correction/hyperspectral_histogram_tree', 'jpg');

% Display nanmean and nanstd
fprintf('\n--- Patch Statistics (Mean ± Std, ignoring NaNs) ---\n');

fprintf('\nCalibration Target 1:\n');
fprintf('  Without downwelling correction: %.2f ± %.2f m\n', nanmean(patch_t1_d1(:)), nanstd(patch_t1_d1(:)));
fprintf('  Downwelling correction:         %.2f ± %.2f m\n', nanmean(patch_t1_d2(:)), nanstd(patch_t1_d2(:)));
fprintf('  Downwelling correction (TV):    %.2f ± %.2f m\n', nanmean(patch_t1_tv(:)), nanstd(patch_t1_tv(:)));
fprintf('  Lidar (Ground Truth):           %.2f ± %.2f m\n', nanmean(patch_t1_lidar(:)), nanstd(patch_t1_lidar(:)));

fprintf('\nCalibration Target 2:\n');
fprintf('  Without downwelling correction: %.2f ± %.2f m\n', nanmean(patch_t2_d1(:)), nanstd(patch_t2_d1(:)));
fprintf('  Downwelling correction:         %.2f ± %.2f m\n', nanmean(patch_t2_d2(:)), nanstd(patch_t2_d2(:)));
fprintf('  Downwelling correction (TV):    %.2f ± %.2f m\n', nanmean(patch_t2_tv(:)), nanstd(patch_t2_tv(:)));
fprintf('  Lidar (Ground Truth):           %.2f ± %.2f m\n', nanmean(patch_t2_lidar(:)), nanstd(patch_t2_lidar(:)));

fprintf('\nTree:\n');
fprintf('  Without downwelling correction: %.2f ± %.2f m\n', nanmean(patch_tree_d1(:)), nanstd(patch_tree_d1(:)));
fprintf('  Downwelling correction:         %.2f ± %.2f m\n', nanmean(patch_tree_d2(:)), nanstd(patch_tree_d2(:)));
fprintf('  Downwelling correction (TV):    %.2f ± %.2f m\n', nanmean(patch_tree_tv(:)), nanstd(patch_tree_tv(:)));
fprintf('  Lidar (Ground Truth):           %.2f ± %.2f m\n', nanmean(patch_tree_lidar(:)), nanstd(patch_tree_lidar(:)));

%% Model Fits
T_air = 289.7;

pixel_reflective_panel_1 =  [143,16];
pixel_reflective_panel_2 = [160,16];
pixel_grass_area = [130;16];
pixel_tree = [40,16];
pixel_background_forest = [80,160];
pixel_sky = [5,16];

%% Reflective Panel 1
pixel_index = pixel_reflective_panel_1;
V = no_downwelling.V(pixel_index(1), pixel_index(2), 1, :);
T = no_downwelling.T(pixel_index(1), pixel_index(2));
emissivity = no_downwelling.emissivity(pixel_index(1), pixel_index(2),:);
d = no_downwelling.d(pixel_index(1), pixel_index(2));
model_fit_no_downwelling = forward_model(lambda, T, emissivity, V, reflected, attenuation, d, T_air);

V = with_downwelling.V(pixel_index(1), pixel_index(2), 1, :);
T = with_downwelling.T(pixel_index(1), pixel_index(2));
emissivity = with_downwelling.emissivity(pixel_index(1), pixel_index(2),:);
d = with_downwelling.d(pixel_index(1), pixel_index(2));
model_fit_with_downwelling = forward_model(lambda, T, emissivity, V, reflected, attenuation, d, T_air);

figure(6);
clf;
set(gcf, 'Position', [100, 100, 600, 520]); % [left, bottom, width, height]
plot(squeeze(lambda), squeeze(meas(pixel_index(1), pixel_index(2), :)), 'LineWidth',1);
hold on;
plot(squeeze(lambda), squeeze(model_fit_with_downwelling), 'LineWidth',1);
plot(squeeze(lambda), squeeze(model_fit_no_downwelling), '--', 'LineWidth',2);
xlabel('Wavelength (μm)');
ylabel('Radiance (μW cm^{-2} sr^{-1} μm^{-1})');
xlim([8,13]);
xticks([8:13]); 
%legend({'Measurement', 'Model fit (including downwelling)', 'Model fit (neglecting downwelling)'}, 'Location','southoutside');
set(gca, 'FontSize', 20);
%saveas(gcf, 'Figures\Hyperspectral_correction\Model_Fits\reflective_panel_1', 'epsc');
%saveas(gcf, 'Figures\Hyperspectral_correction\Model_Fits\reflective_panel_1', 'jpg');
%% Reflective Panel 2
pixel_index = pixel_reflective_panel_2;
V = no_downwelling.V(pixel_index(1), pixel_index(2), 1, :);
T = no_downwelling.T(pixel_index(1), pixel_index(2));
emissivity = no_downwelling.emissivity(pixel_index(1), pixel_index(2),:);
d = no_downwelling.d(pixel_index(1), pixel_index(2));
model_fit_no_downwelling = forward_model(lambda, T, emissivity, V, reflected, attenuation, d, T_air);

V = with_downwelling.V(pixel_index(1), pixel_index(2), 1, :);
T = with_downwelling.T(pixel_index(1), pixel_index(2));
emissivity = with_downwelling.emissivity(pixel_index(1), pixel_index(2),:);
d = with_downwelling.d(pixel_index(1), pixel_index(2));
model_fit_with_downwelling = forward_model(lambda, T, emissivity, V, reflected, attenuation, d, T_air);

figure(7);
clf;
set(gcf, 'Position', [100, 100, 600, 520]); % [left, bottom, width, height]
plot(squeeze(lambda), squeeze(meas(pixel_index(1), pixel_index(2), :)), 'LineWidth',1);
hold on;
plot(squeeze(lambda), squeeze(model_fit_with_downwelling), 'LineWidth',1);
plot(squeeze(lambda), squeeze(model_fit_no_downwelling), '--', 'LineWidth',2);
xlabel('Wavelength (μm)');
ylabel('Radiance (μW cm^{-2} sr^{-1} μm^{-1})');
xlim([8,13]);
xticks([8:13]); 
%legend({'Measurement', 'Model fit (including downwelling)', 'Model fit (neglecting downwelling)'}, 'Location','southoutside');
set(gca, 'FontSize', 20);
axes('Position', [0.35 0.25 0.3 0.3]);  
box on
plot(squeeze(lambda(1, 1, 70:90)), squeeze(meas(pixel_index(1), pixel_index(2), 70:90)), 'LineWidth',1);
hold on;
plot(squeeze(lambda(1,1,70:90)), squeeze(model_fit_with_downwelling(1,1,70:90)), 'LineWidth',1);
plot(squeeze(lambda(1,1,70:90)), squeeze(model_fit_no_downwelling(1,1,70:90)), '--', 'LineWidth',2);
xlim([lambda(70), lambda(90)])
%saveas(gcf, 'Figures\Hyperspectral_correction\Model_Fits\reflective_panel_2', 'epsc');
%saveas(gcf, 'Figures\Hyperspectral_correction\Model_Fits\reflective_panel_2', 'jpg');
%% Grass area
pixel_index = pixel_grass_area;
V = no_downwelling.V(pixel_index(1), pixel_index(2), 1, :);
T = no_downwelling.T(pixel_index(1), pixel_index(2));
emissivity = no_downwelling.emissivity(pixel_index(1), pixel_index(2),:);
d = no_downwelling.d(pixel_index(1), pixel_index(2));
model_fit_no_downwelling = forward_model(lambda, T, emissivity, V, reflected, attenuation, d, T_air);

V = with_downwelling.V(pixel_index(1), pixel_index(2), 1, :);
T = with_downwelling.T(pixel_index(1), pixel_index(2));
emissivity = with_downwelling.emissivity(pixel_index(1), pixel_index(2),:);
d = with_downwelling.d(pixel_index(1), pixel_index(2));
model_fit_with_downwelling = forward_model(lambda, T, emissivity, V, reflected, attenuation, d, T_air);

figure(8);
clf;
set(gcf, 'Position', [100, 100, 600, 520]); % [left, bottom, width, height]
plot(squeeze(lambda), squeeze(meas(pixel_index(1), pixel_index(2), :)), 'LineWidth',1);
hold on;
plot(squeeze(lambda), squeeze(model_fit_with_downwelling), 'LineWidth',1);
plot(squeeze(lambda), squeeze(model_fit_no_downwelling), '--', 'LineWidth',2);
xlabel('Wavelength (μm)');
ylabel('Radiance (μW cm^{-2} sr^{-1} μm^{-1})');
xlim([8,13]);
xticks([8:13]); 
%legend({'Measurement', 'Model fit (including downwelling)', 'Model fit (neglecting downwelling)'}, 'Location','southoutside');
set(gca, 'FontSize', 20);
axes('Position', [0.35 0.25 0.3 0.3]);  
box on
plot(squeeze(lambda(1, 1, 70:90)), squeeze(meas(pixel_index(1), pixel_index(2), 70:90)), 'LineWidth',1);
hold on;
plot(squeeze(lambda(1,1,70:90)), squeeze(model_fit_with_downwelling(1,1,70:90)), 'LineWidth',1);
plot(squeeze(lambda(1,1,70:90)), squeeze(model_fit_no_downwelling(1,1,70:90)), '--', 'LineWidth',2);
xlim([lambda(70), lambda(90)])
%saveas(gcf, 'Figures\Hyperspectral_correction\Model_Fits\grass', 'epsc');
%saveas(gcf, 'Figures\Hyperspectral_correction\Model_Fits\grass', 'jpg');
%% Tree area
pixel_index = pixel_tree;
V = no_downwelling.V(pixel_index(1), pixel_index(2), 1, :);
T = no_downwelling.T(pixel_index(1), pixel_index(2));
emissivity = no_downwelling.emissivity(pixel_index(1), pixel_index(2),:);
d = no_downwelling.d(pixel_index(1), pixel_index(2));
model_fit_no_downwelling = forward_model(lambda, T, emissivity, V, reflected, attenuation, d, T_air);

V = with_downwelling.V(pixel_index(1), pixel_index(2), 1, :);
T = with_downwelling.T(pixel_index(1), pixel_index(2));
emissivity = with_downwelling.emissivity(pixel_index(1), pixel_index(2),:);
d = with_downwelling.d(pixel_index(1), pixel_index(2));
model_fit_with_downwelling = forward_model(lambda, T, emissivity, V, reflected, attenuation, d, T_air);

figure(9);
clf;
set(gcf, 'Position', [100, 100, 600, 650]); % [left, bottom, width, height]
plot(squeeze(lambda), squeeze(meas(pixel_index(1), pixel_index(2), :)), 'LineWidth',1);
hold on;
plot(squeeze(lambda), squeeze(model_fit_with_downwelling), 'LineWidth',1);
plot(squeeze(lambda), squeeze(model_fit_no_downwelling), '--', 'LineWidth',2);
xlim([8,13]);
xticks([8:13]); 
xlabel('Wavelength (μm)');
ylabel('Radiance (μW cm^{-2} sr^{-1} μm^{-1})');
legend({'Measurement', 'Model fit (including downwelling)', 'Model fit (neglecting downwelling)'}, 'Location','southoutside');
set(gca, 'FontSize', 20);
axes('Position', [0.35 0.45 0.3 0.3]);  
box on
plot(squeeze(lambda(1, 1, 70:90)), squeeze(meas(pixel_index(1), pixel_index(2), 70:90)), 'LineWidth',1);
hold on;
plot(squeeze(lambda(1,1,70:90)), squeeze(model_fit_with_downwelling(1,1,70:90)), 'LineWidth',1);
plot(squeeze(lambda(1,1,70:90)), squeeze(model_fit_no_downwelling(1,1,70:90)), '--', 'LineWidth',2);
xlim([lambda(70), lambda(90)])
%saveas(gcf, 'Figures\Hyperspectral_correction\Model_Fits\tree', 'epsc');
%saveas(gcf, 'Figures\Hyperspectral_correction\Model_Fits\tree', 'jpg');
%% Background forest area
pixel_index = pixel_background_forest;
V = no_downwelling.V(pixel_index(1), pixel_index(2), 1, :);
T = no_downwelling.T(pixel_index(1), pixel_index(2));
emissivity = no_downwelling.emissivity(pixel_index(1), pixel_index(2),:);
d = no_downwelling.d(pixel_index(1), pixel_index(2));
model_fit_no_downwelling = forward_model(lambda, T, emissivity, V, reflected, attenuation, d, T_air);

V = with_downwelling.V(pixel_index(1), pixel_index(2), 1, :);
T = with_downwelling.T(pixel_index(1), pixel_index(2));
emissivity = with_downwelling.emissivity(pixel_index(1), pixel_index(2),:);
d = with_downwelling.d(pixel_index(1), pixel_index(2));
model_fit_with_downwelling = forward_model(lambda, T, emissivity, V, reflected, attenuation, d, T_air);

figure(10);
clf;
set(gcf, 'Position', [100, 100, 600, 650]); % [left, bottom, width, height]
plot(squeeze(lambda), squeeze(meas(pixel_index(1), pixel_index(2), :)), 'LineWidth',1);
hold on;
plot(squeeze(lambda), squeeze(model_fit_with_downwelling), 'LineWidth',1);
plot(squeeze(lambda), squeeze(model_fit_no_downwelling), '--', 'LineWidth',2);
xlim([8,13]);
xticks([8:13]); 
xlabel('Wavelength (μm)');
ylabel('Radiance (μW cm^{-2} sr^{-1} μm^{-1})');
legend({'Measurement', 'Model fit (including downwelling)', 'Model fit (neglecting downwelling)'}, 'Location','southoutside');
set(gca, 'FontSize', 20);
axes('Position', [0.35 0.45 0.3 0.3]);  
box on
plot(squeeze(lambda(1, 1, 70:90)), squeeze(meas(pixel_index(1), pixel_index(2), 70:90)), 'LineWidth',1);
hold on;
plot(squeeze(lambda(1,1,70:90)), squeeze(model_fit_with_downwelling(1,1,70:90)), 'LineWidth',1);
plot(squeeze(lambda(1,1,70:90)), squeeze(model_fit_no_downwelling(1,1,70:90)), '--', 'LineWidth',2);
xlim([lambda(70), lambda(90)])
%saveas(gcf, 'Figures\Hyperspectral_correction\Model_Fits\background_forest', 'epsc');
%saveas(gcf, 'Figures\Hyperspectral_correction\Model_Fits\background_forest', 'jpg');
%% Sky pixels
pixel_index = pixel_sky;
V = no_downwelling.V(pixel_index(1), pixel_index(2), 1, :);
T = no_downwelling.T(pixel_index(1), pixel_index(2));
emissivity = no_downwelling.emissivity(pixel_index(1), pixel_index(2),:);
d = no_downwelling.d(pixel_index(1), pixel_index(2));
model_fit_no_downwelling = forward_model(lambda, T, emissivity, V, reflected, attenuation, d, T_air);

V = with_downwelling.V(pixel_index(1), pixel_index(2), 1, :);
T = with_downwelling.T(pixel_index(1), pixel_index(2));
emissivity = with_downwelling.emissivity(pixel_index(1), pixel_index(2),:);
d = with_downwelling.d(pixel_index(1), pixel_index(2));
model_fit_with_downwelling = forward_model(lambda, T, emissivity, V, reflected, attenuation, d, T_air);

figure(11);
clf;
set(gcf, 'Position', [100, 100, 600, 520]); % [left, bottom, width, height]
plot(squeeze(lambda), squeeze(meas(pixel_index(1), pixel_index(2), :)), 'LineWidth',1);
hold on;
plot(squeeze(lambda), squeeze(model_fit_with_downwelling), 'LineWidth',1);
plot(squeeze(lambda), squeeze(model_fit_no_downwelling), '--', 'LineWidth',2);
xlim([8,13]);
xticks([8:13]); 
xlabel('Wavelength (μm)');
ylabel('Radiance (μW cm^{-2} sr^{-1} μm^{-1})');
%legend({'Measurement', 'Model fit (including downwelling)', 'Model fit (neglecting downwelling)'}, 'Location','southoutside');
set(gca, 'FontSize', 20);
%saveas(gcf, 'Figures\Hyperspectral_correction\Model_Fits\sky', 'epsc');
%saveas(gcf, 'Figures\Hyperspectral_correction\Model_Fits\sky', 'jpg');
%%
%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% Other Plots
%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%
%%
%downwelling_plot_name={'0^\circ', '30^\circ', '60^\circ', '70^\circ', '80^\circ',...
%    '82^\circ', '84^\circ', '86^\circ', '88^\circ', '89^\circ'};
downwelling_name={'0deg', '30deg', '60deg', '70deg', '80deg',...
    '82deg', '84deg', '86deg', '88deg', '89deg'};

for i = 1:10
    figure;
    imagesc(with_downwelling.V(:,:,1,i));
    colorbar('northoutside', 'Ticks',[0, 0.2], 'TickLabels',{'0', '0.2'})
    %title(downwelling_name(i))
    caxis([0,0.2])
    set(gca, 'FontSize', 25);
    axis off image
    %saveas(gcf, ['Figures\Hyperspectral_correction\Normalized_Solid_Angles\', downwelling_name{i}], 'epsc');  
    %saveas(gcf, ['Figures\Hyperspectral_correction\Normalized_Solid_Angles\', downwelling_name{i}], 'jpg');
end
%% K-means on emissivity
clusters = 5;
max_iterations = 100000;  % Define the same max iterations for kmeans

% For no_downwelling emissivity
emissivity = no_downwelling.emissivity;
[N,M,K] = size(emissivity);
emissivity = reshape(emissivity, [N*M, K]);

% Perform k-means clustering
rng(0);
[idx, C] = kmeans(emissivity, clusters, 'MaxIter', max_iterations);
idx = reshape(idx, [N, M]);

% Calculate mean emissivity for each cluster center
mean_emissivity = mean(C, 2);

% Sort the clusters based on the mean emissivity (from highest to lowest)
[~, sort_idx] = sort(mean_emissivity, 'descend');

% Reorder the cluster centers and indices
C = C(sort_idx, :);
new_idx = zeros(size(idx));
for i = 1:clusters
    new_idx(idx == sort_idx(i)) = i;
end

% Update the cluster index with the new order
idx = reshape(new_idx, [N, M]);

% Normalize idx values for imshow (idx already ranges from 1 to clusters)
normalized_idx = idx;

% Define vibrant pastel colors for 5 clusters
cluster_colors = [
    0.19, 0.64, 0.28;  % Grass green (like fresh grass)
    0.76, 0.70, 0.50;  % Dry grass-soil (a mix of beige and light brown)
    0.55, 0.57, 0.67;  % Metallic gray (subtle, darker metallic tone)
    0.53, 0.81, 0.92;  % Sky blue (like a clear daytime sky)
    0.82, 0.82, 0.88   % Bright metallic silver (shiny, lighter metallic tone)
];

% Display the colored idx image using imshow
figure(6)
clf;
set(gcf, 'Position', [560, 420, 560, 560])
imshow(normalized_idx, [1 clusters], 'InitialMagnification', 'fit'); % Ensure the range is from 1 to the number of clusters
colormap(cluster_colors);             % Apply the custom colormap
colorbar('northoutside','Ticks', linspace(1, clusters, clusters), ...
         'TickLabels', arrayfun(@(x) sprintf('Cluster %d', x), 1:clusters, 'UniformOutput', false)); % Correct colorbar
set(gca, 'FontSize', 17);
%saveas(gcf, 'Figures/Hyperspectral_correction/Emissivity_clusters/cluster_image_no_downwelling', 'epsc');
%saveas(gcf, 'Figures/Hyperspectral_correction/Emissivity_clusters/cluster_image_no_downwelling', 'jpg');

% Plot the cluster centers and use the same pastel colors
figure(7)
set(gcf, 'Position', [100, 100, 600, 520]); % [left, bottom, width, height]
clf;
hold on;
for i = 1:clusters
    plot(squeeze(lambda), C(i,:), 'Color', cluster_colors(i, :), 'LineWidth', 2);
end
hold off;
ylim([0.5, 1]);
xlim([8,13]);
xticks([8:13]); 
legend({'Cluster 1', 'Cluster 2', 'Cluster 3','Cluster 4','Cluster 5'}, 'Location', 'southwest', 'NumColumns',2);
xlabel('Wavelength (μm)');
ylabel('Emissivity');
set(gca, 'FontSize', 25)
%saveas(gcf, 'Figures/Hyperspectral_correction/Emissivity_clusters/cluster_spectrum_no_downwelling', 'epsc');
%saveas(gcf, 'Figures/Hyperspectral_correction/Emissivity_clusters/cluster_spectrum_no_downwelling', 'jpg');

% Repeat for the "with_downwelling" emissivity

emissivity = with_downwelling.emissivity;
[N,M,K] = size(emissivity);
emissivity = reshape(emissivity, [N*M, K]);

% Perform k-means clustering with the same max iterations
rng(0);
[idx, C] = kmeans(emissivity, clusters, 'MaxIter', max_iterations);
idx = reshape(idx, [N, M]);

% Calculate mean emissivity for each cluster center
mean_emissivity = mean(C, 2);

% Sort the clusters based on the mean emissivity (from highest to lowest)
[~, sort_idx] = sort(mean_emissivity, 'descend');

% Reorder the cluster centers and indices
C = C(sort_idx, :);
new_idx = zeros(size(idx));
for i = 1:clusters
    new_idx(idx == sort_idx(i)) = i;
end

% Update the cluster index with the new order
idx = reshape(new_idx, [N, M]);

% Normalize idx values for imshow
normalized_idx = idx;

% Display the colored idx image using imshow
figure(8)
set(gcf, 'Position', [560, 420, 560, 560])
clf;
imshow(normalized_idx, [1 clusters], 'InitialMagnification', 'fit'); % Ensure the range is from 1 to the number of clusters
colormap(cluster_colors);             % Apply the custom colormap
colorbar('northoutside', 'Ticks', linspace(1, clusters, clusters), ...
         'TickLabels', arrayfun(@(x) sprintf('Cluster %d', x), 1:clusters, 'UniformOutput', false)); % Correct colorbar
set(gca, 'FontSize', 17);
%saveas(gcf, 'Figures/Hyperspectral_correction/Emissivity_clusters/cluster_image_with_downwelling', 'epsc');
%saveas(gcf, 'Figures/Hyperspectral_correction/Emissivity_clusters/cluster_image_with_downwelling', 'jpg');

% Plot the cluster centers and use the same pastel colors
figure(9)
clf;
set(gcf, 'Position', [100, 100, 600, 520]); % [left, bottom, width, height]
hold on;
for i = 1:clusters
    plot(squeeze(lambda), C(i,:), 'Color', cluster_colors(i, :), 'LineWidth', 2);
end
hold off;
xlabel('Wavelength (μm)');
ylabel('Emissivity');
ylim([0.4, 1]);
xlim([8,13]);
xticks([8:13]); 
legend({'Cluster 1', 'Cluster 2', 'Cluster 3','Cluster 4','Cluster 5'}, 'Location', 'southwest', 'NumColumns',2);
set(gca, 'FontSize', 25)
%saveas(gcf, 'Figures/Hyperspectral_correction/Emissivity_clusters/cluster_spectrum_with_downwelling', 'epsc');
%saveas(gcf, 'Figures/Hyperspectral_correction/Emissivity_clusters/cluster_spectrum_with_downwelling', 'jpg');%% 3D point cloud
%% Temperature
cut_off_1 = 285;
cut_off_2 = 294;

figure(10)
clf;
imagesc(with_downwelling.T(:, :));
colormap('hot');
clim([cut_off_1, cut_off_2]);
cb = colorbar('northoutside');
%ylabel(cb, 'Distance (m)');
set(cb, 'Ticks', [cut_off_1, cut_off_2], 'TickLabels', {[num2str(cut_off_1),' K'], [num2str(cut_off_2), ' K']});
set(gca, 'FontSize', 20)
%set(gca, 'Position', get(gca, 'Position') + [0, 0, 0, -0.05]);
axis image off
%saveas(gcf, 'Figures/Hyperspectral_correction/Temperature/hyperspectral_with_downwelling_temperature', 'epsc');
%saveas(gcf, 'Figures/Hyperspectral_correction/Temperature/hyperspectral_with_downwelling_temperature', 'jpg');

figure(11)
clf;
imagesc(no_downwelling.T(:, :));
colormap('hot');
clim([cut_off_1, cut_off_2]);
cb = colorbar('northoutside');
%ylabel(cb, 'Distance (m)');
set(cb, 'Ticks', [cut_off_1, cut_off_2], 'TickLabels', {[num2str(cut_off_1),' K'], [num2str(cut_off_2), ' K']});
set(gca, 'FontSize', 20)
%set(gca, 'Position', get(gca, 'Position') + [0, 0, 0, -0.05]);
axis image off
%saveas(gcf, 'Figures/Hyperspectral_correction/Temperature/hyperspectral_no_downwelling_temperature', 'epsc');
%saveas(gcf, 'Figures/Hyperspectral_correction/Temperature/hyperspectral_no_downwelling_temperature', 'jpg');