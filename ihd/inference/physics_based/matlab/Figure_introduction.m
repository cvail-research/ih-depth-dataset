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

no_downwelling.V = no_downwelling.V(:,:,:,1:Q);
with_downwelling.V = with_downwelling.V(:,:,:,1:Q);
with_downwelling_reg.V = with_downwelling_reg.V(:,:,:,1:Q);

lambda = load([data_dir, 'lambda.mat']).lambda(1:K);
meas = load([data_dir, 'DC2P5S1.mat']).measurements(1:256, 901:1156,1:K);
attenuation = load([data_dir, 'attenuation.mat']).attenuation(1:K);
reflected = load([data_dir, 'I_downwelling_res.mat']).I_downwelling_res(1:K,:);

lambda = reshape(lambda, [1, 1, K]) - 0.120;
attenuation = reshape(attenuation, [1, 1, K]);
reflected = reshape(reflected, [1, 1, K, Q]);
%% Plot Ranging Results
cmap = colormap('parula');
cmap = [flip(cmap)];
cut_off_1 = 0;
cut_off_2 = 150;

pixel_reflective_panel_1 = [143,16];
pixel_grass_area = [68,16];
pixel_sky = [5,16];

ozone_bands = 77;

% Define colors for squares using [R, G, B] values (normalized between 0 and 1)
square_colors = [
    0.71, 0.27, 0.61;  % Light Magenta-Purple for Reflective Panel
    0.13, 0.54, 0.13;  % Forest Green for Grass Area
    0.40, 0.60, 0.85   % Slightly Darker Light Sky Blue for Sky
];

circle_centers = [100, 15; 150, 15; 191, 82]; % Define the centers as (row, col)
circle_radius = 15; % Define the radius

% No downwelling
figure(1);
clf;
set(gca, 'Position', [0.05, 0.05, 0.9, 0.75]);
imagesc(no_downwelling.d(:, :));
colormap(cmap);
clim([cut_off_1, cut_off_2]);
cb = colorbar('northoutside');
ylabel(cb, 'Distance (m)')
set(cb, 'Ticks', [cut_off_1, cut_off_2], 'TickLabels', {[num2str(cut_off_1)], [num2str(cut_off_2)]});
set(gca, 'FontSize', 25);
axis image off;
hold on;

% Add circles
for i = 1:size(circle_centers, 1)
    plot_circle(circle_centers(i, :), circle_radius, [1, 0, 0]); % Red dashed circles
end

% Add unfilled squares with dashed outlines
plot(pixel_reflective_panel_1(2), pixel_reflective_panel_1(1), 'x', ...
    'Color', square_colors(1, :), 'MarkerSize', 15, 'LineWidth', 3);
plot(pixel_grass_area(2), pixel_grass_area(1), 'x', ...
    'Color', square_colors(2, :), 'MarkerSize', 15, 'LineWidth', 3);
plot(pixel_sky(2), pixel_sky(1), 'x', ...
    'Color', square_colors(3, :), 'MarkerSize', 15, 'LineWidth', 3);

hold off;
%saveas(gcf, 'Figures/Figure_introduction/introduction_hyperspectral_no_downwelling_range', 'epsc');
%saveas(gcf, 'Figures/Figure_introduction/introduction_hyperspectral_no_downwelling_range', 'jpg');

% With downwelling
figure(2);
clf;
set(gca, 'Position', [0.05, 0.05, 0.9, 0.75]);
imagesc(with_downwelling_reg.d(:, :));
colormap(cmap);
clim([cut_off_1, cut_off_2]);
cb = colorbar('northoutside');
ylabel(cb, 'Distance (m)')
set(cb, 'Ticks', [cut_off_1, cut_off_2], 'TickLabels', {[num2str(cut_off_1)], [num2str(cut_off_2)]});
set(gca, 'FontSize', 25);
axis image off;
hold on;

% Add circles
for i = 1:size(circle_centers, 1)
    plot_circle(circle_centers(i, :), circle_radius, [1, 0, 0]); % Red dashed circles
end

% Add unfilled squares with dashed outlines
plot(pixel_reflective_panel_1(2), pixel_reflective_panel_1(1), 'x', ...
    'Color', square_colors(1, :), 'MarkerSize', 15, 'LineWidth', 3);
plot(pixel_grass_area(2), pixel_grass_area(1), 'x', ...
    'Color', square_colors(2, :), 'MarkerSize', 15, 'LineWidth', 3);
plot(pixel_sky(2), pixel_sky(1), 'x', ...
    'Color', square_colors(3, :), 'MarkerSize', 15, 'LineWidth', 3);

hold off;
%saveas(gcf, 'Figures/Figure_introduction/introduction_hyperspectral_with_downwelling_range', 'epsc');
%saveas(gcf, 'Figures/Figure_introduction/introduction_hyperspectral_with_downwelling_range', 'jpg');

% Figure 3 - Spectral Plots (Double Width)
figure(3);
set(gcf, 'Position', [200, 200, 1450, 800]); % Double width of Figures 1 and 2
clf;
hold on;
set(gca, 'FontSize', 35);

% Extract spectral data from meas at selected pixel locations
spec_reflective_panel = squeeze(meas(pixel_reflective_panel_1(1), pixel_reflective_panel_1(2), :));
spec_grass_area = squeeze(meas(pixel_grass_area(1), pixel_grass_area(2), :));
spec_sky = squeeze(meas(pixel_sky(1), pixel_sky(2), :));

% Plot spectral measurements as a function of wavelength
plot(lambda(:), spec_reflective_panel, '-', 'Color', square_colors(1, :), 'LineWidth', 3);
plot(lambda(:), spec_sky, '-', 'Color', square_colors(3, :), 'LineWidth', 3);
yyaxis right;
plot(lambda(:), spec_grass_area, '-', 'Color', square_colors(2, :), 'LineWidth', 3);
ylim([670,800])
ax = gca;
ax.YColor = square_colors(2, :);
ylabel('Radiance (μW cm^{-2} sr^{-1} μm^{-1})');
yyaxis left;
xlim([min(lambda), max(lambda)]);
ylim([500,900]);
xlabel('Wavelength (μm)');
ylabel('Radiance (μW cm^{-2} sr^{-1} μm^{-1})');
xlim([8,13]);
xticks([8:13]); 

% Move ozone fill to the end to control legend order
ozone_band_start = lambda(ozone_bands) - 0.1;
ozone_band_end = lambda(ozone_bands) + 0.5;
y_limits = [200,875];
ozone_patch = fill([ozone_band_start ozone_band_end ozone_band_end ozone_band_start], ...
    [y_limits(1) y_limits(1) y_limits(2) y_limits(2)], [0, 0, 0], ...
    'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Define legend in correct order using plot handles
legend_handles = findall(gca, 'Type', 'line');
legend_handles = flipud(legend_handles); % Bring into plotting order
legend([legend_handles(1), legend_handles(2), legend_handles(3), ozone_patch], ...
    {'Reflective panel', 'Sky', 'Grass area', 'Ozone absorption band'}, ...
    'Location', 'north','NumColumns', 2);
hold off;

% Save the figure
%saveas(gcf, 'Figures/Figure_introduction/introduction_hyperspectral_spectral_signatures', 'epsc');
%saveas(gcf, 'Figures/Figure_introduction/introduction_hyperspectral_spectral_signatures', 'jpg');
%%
function plot_circle(center, radius, color)
    theta = linspace(0, 2*pi, 100);
    x = center(2) + radius * cos(theta); % X-coordinates
    y = center(1) + radius * sin(theta); % Y-coordinates
    plot(x, y, 'LineWidth', 2, 'Color', color, 'LineStyle', '--');
end