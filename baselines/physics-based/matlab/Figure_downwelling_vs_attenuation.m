clear all;
close all;

addpath(genpath('Functions'));
data_dir = '';%Enter data directory

load([data_dir, 'attenuation.mat']);
load([data_dir, 'I_downwelling_res.mat']);
[ozone.wavelength, ozone.transmittance] = read_transmittance_file([data_dir, 'transmittance_ozone.txt']);
[water_vapor.wavelength, water_vapor.transmittance] = read_transmittance_file([data_dir, 'transmittance_water_vapor.txt']);

ozone_bands = 77;

ozone_attenuation = -10*log10(interp1(ozone.wavelength, ozone.transmittance, lambda));
water_vapor_attenuation = -10*log10(interp1(water_vapor.wavelength, water_vapor.transmittance, lambda));


figure(1)
set(gcf, 'Position', [100, 100, 2.5*560, 800]); % Set the figure size (default is 560x420, so 3 times wider)
plot(lambda, attenuation, 'LineWidth', 2);
ylabel('Attenuation (dB/m)');
yyaxis right;
plot(lambda, I_downwelling_res(:,1),'--', 'LineWidth', 3);
hold on;
ozone_band_start = lambda(ozone_bands) - 0.1; % Adjust as needed
ozone_band_end = lambda(ozone_bands) + 0.5; % Adjust as needed
y_limits = [0,800]; % Get y-axis limits
fill([ozone_band_start ozone_band_end ozone_band_end ozone_band_start], ...
    [y_limits(1) y_limits(1) y_limits(2) y_limits(2)], [0, 0, 0], ...
    'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Light black transparent fill
ylabel('Radiance (μW cm^{-2} sr^{-1} μm^{-1})');
xlim([min(lambda), max(lambda)]);
ylim([0,700])
xlabel('Wavelength (μm)');
legend({'Ground-level attenuation', 'Downwelling radiance', 'Ozone absorption band'}, 'Location', 'NorthEast');
set(gca, 'FontSize', 35);
%saveas(gcf, 'Figures/Ozone_feature/ozone_feature', 'epsc');
%saveas(gcf, 'Figures/Ozone_feature/ozone_feature', 'jpg');
%%
figure(2)
set(gcf, 'Position', [100, 100, 2.5*560, 600]); % Set figure size

hold on;
% Plot first curve (Water vapor attenuation) in default MATLAB blue
plot(lambda, water_vapor_attenuation, 'Color', [0, 0.4470, 0.7410], 'LineWidth', 3);

% Plot second curve (Ozone attenuation) in a darker blue with a dashed line
plot(lambda, ozone_attenuation, '--', 'Color', [0, 0, 0.5], 'LineWidth', 3);

% Ozone absorption band as a shaded region
ozone_band_start = lambda(ozone_bands) - 0.1; % Adjust as needed
ozone_band_end = lambda(ozone_bands) + 0.5; % Adjust as needed
y_limits = [0, 1000]; % Get y-axis limits
fill([ozone_band_start ozone_band_end ozone_band_end ozone_band_start], ...
    [y_limits(1) y_limits(1) y_limits(2) y_limits(2)], [0, 0, 0], ...
    'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Light black transparent fill

% Labels and limits
ylabel('Attenuation (dB/m)');
xlim([min(lambda), max(lambda)]);
ylim([0, 0.25]);
xlabel('Wavelength (μm)');

% Legend (updated order)
legend({'Water vapor attenuation', 'Ozone attenuation', 'Ozone absorption band'}, 'Location', 'NorthEast');

% Font size adjustment
set(gca, 'FontSize', 35);

% Save figures
%saveas(gcf, 'Figures/Ozone_feature/ozone_feature_2', 'epsc');
%saveas(gcf, 'Figures/Ozone_feature/ozone_feature_2', 'jpg');
