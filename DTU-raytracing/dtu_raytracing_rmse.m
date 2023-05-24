%% setting constants
freq = 811 * 10^6;  % Tx freq 
lambda = 3 * 10^8 / freq;  % wavelength
Tx_LAT = 55.784663;
Tx_LONG = 12.523303;

%% reading some data
df_feature = readmatrix('../raw_data/feature_matrix.csv');
df_output = readmatrix('../raw_data/output_matrix.csv');
%% setting up long, lat, ss
df_feature_only_65 = df_feature(df_feature(:, 8) == 65, :);
lat_arr = df_feature_only_65(:, 3);
long_arr = df_feature_only_65(:, 2);

df_output_only_65 = df_output(df_feature(:, 8) == 65, :);
power_empirical_arr = df_output_only_65(:, end);

%% setting up viewer for DTU
viewer = siteviewer('Buildings', 'dtu.osm');

%% antenna, tx and propagation model
sectoria = sectorInvertedAmos('ArmLength',0.5.*[0.0880 0.0710 0.0730 0.0650], ...
    'GroundPlaneWidth', 0.15, 'GroundPlaneLength', 1);

% pattern(sectoria, freq)
hant_sectoria = design(sectoria, freq);

tx = txsite("Latitude",Tx_LAT, "Longitude",Tx_LONG, ...
     "TransmitterFrequency",freq, 'AntennaHeight',30, ...
     'Antenna',hant_sectoria, 'AntennaAngle', -45);

pm = propagationModel('raytracing', 'MaxNumReflections',5, 'MaxNumDiffractions',1, ...
    'AngularSeparation', 'low', ...
    'MaxRelativePathLoss', 35);
%% calculate sig strength given rx location
every_nth = 20;
lat_arr_short = lat_arr(1:every_nth:end);
long_arr_short = long_arr(1:every_nth:end);
power_empirical_arr_short = power_empirical_arr(1:every_nth:end);

power_raytrace_arr = zeros(length(long_arr_short), 1);
parfor iii=1:length(long_arr_short)
    rx = rxsite("Latitude",lat_arr(iii), "Longitude",long_arr(iii), "AntennaHeight",1.5);
    power_raytrace_arr(iii) = sigstrength(rx,tx,pm);
end

%% calculate rmse with offset
offset_arr = linspace(-50, 0, 101);
rmse_arr = zeros(1, length(offset_arr));
for iii=1:length(offset_arr)
    rmse_arr(iii) = rmse_empirical_raytracing(power_empirical_arr_short, ...
        power_raytrace_arr, offset_arr(iii));
end
[m, idx] = min(rmse_arr);
sprintf('smallest rmse %.5f occurs at offset %.2f', m, offset_arr(cast(idx, 'uint16')))

plot(offset_arr, rmse_arr, '--o')
%% save raytracing sig strength


function [x] = rmse_empirical_raytracing(empirical, predicted, offset)
    x = sqrt(sum((predicted + offset - empirical).^2) / length(empirical));
end