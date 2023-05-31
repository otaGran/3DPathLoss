%% this script calculates the RMSE between raytracing data and real measurements
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
    'GroundPlaneWidth', 0.15, 'GroundPlaneLength', 1, Tilt=-10, TiltAxis=[0 1 0]);

pattern(sectoria, freq)
%%
hant_sectoria = design(sectoria, freq);
transmit_power = 4;
tx = txsite("Latitude",Tx_LAT, "Longitude",Tx_LONG, ...
     "TransmitterFrequency",freq, 'AntennaHeight',30, ...
     'Antenna',hant_sectoria, 'AntennaAngle',-45, 'TransmitterPower',transmit_power);

every_nth = 80;

lat_arr_short = lat_arr(1:every_nth:end);
long_arr_short = long_arr(1:every_nth:end);
power_empirical_arr_short = power_empirical_arr(1:every_nth:end);


rmse_arr_outer = [];
time_arr_outer = [];
opt_offset_arr_outer = [];

nReflections = linspace(0, 10, 11);

for jjj=1:length(nReflections)
    timeStart = tic;
    temp_pow_raytrace = raytrace_pow(tx, nReflections(jjj), lat_arr_short, long_arr_short);
    [min_rmse, opt_offset] = rmse_and_offset(temp_pow_raytrace, power_empirical_arr_short);
    timeEnd = toc(timeStart);

    rmse_arr_outer = [rmse_arr_outer, min_rmse];
    time_arr_outer = [time_arr_outer, timeEnd];
    opt_offset_arr_outer = [opt_offset_arr_outer, opt_offset];
end

%% save stuff
doc(:, 1) = nReflections;
doc(:, 2) = rmse_arr_outer;
doc(:, 3) = time_arr_outer;
doc(:, 4) = opt_offset_arr_outer;
fname = sprintf('DTU_raytrace_pow%d_conc_loam_nRefl_Tx30m.txt', transmit_power);
fopen(fname,'w');
writelines('nReflections,RMSE,Duration,OptimalOffset', fname, WriteMode='append')
for iii=1:length(doc(:, 1))
    writelines(sprintf('%.6f,%.6f,%.8f,%.1f',doc(iii,1),doc(iii,2),doc(iii,3),doc(iii,4)),fname, WriteMode='append')
end

%% plotting stuff
subplot(2, 2, 1)
plot(nReflections, rmse_arr_outer)
xlabel('Max Number of Reflections'); ylabel('RMSE')

subplot(2, 2, 2)
plot(nReflections, time_arr_outer)
xlabel('Max Number of Reflections'); ylabel('Duration of Computation')

subplot(2, 2, 3)
plot(nReflections, opt_offset_arr_outer)
xlabel('Max Number of Reflections'); ylabel('Optimal Offset')


%% save raytracing sig strength

% doc(:, 1) = long_arr_short;
% doc(:, 2) = lat_arr_short;
% doc(:, 3) = power_raytrace_arr;
% fname = sprintf('DTU_raytrace_pow%d_sigma%.4f_medAngSep.txt', transmit_power, 0.0275);
% fopen(fname,'w');
% writelines('Longitude,Latitude,Power', fname, WriteMode='append')
% for iii=1:length(doc(:, 1))
%     writelines(sprintf('%.6f,%.6f,%.8f',doc(iii,1),doc(iii,2),doc(iii,3)),fname, WriteMode='append')
% end
%
% %%
% bbb = readmatrix("DTU_raytrace_pow4_sigma0.0275.txt");
% lat_arr_short = bbb(:, 2);
% long_arr_short = bbb(:, 1);
% power_raytrace_arr = bbb(:, 3);
% 
% %% delete inf values
% lat_arr_short = lat_arr_short(~isinf(power_raytrace_arr));
% long_arr_short = long_arr_short(~isinf(power_raytrace_arr));
% power_empirical_arr_short = power_empirical_arr_short(~isinf(power_raytrace_arr));
% power_raytrace_arr = power_raytrace_arr(~isinf(power_raytrace_arr));
%
% sprintf('smallest rmse %.5f occurs at offset %.2f', m, offset_arr(cast(idx, 'uint16')))
% figure; clf
% hold on
% plot(offset_arr, rmse_arr, '--o')
% xlabel('offset'); ylabel('rmse');


function [x] = rmse_empirical_raytracing(empirical, predicted, offset)
    x = sqrt(sum((empirical + offset - predicted).^2) / length(empirical));
end


function [power_raytracing] = raytrace_pow(tx, MaxNumReflections, lat_arr_short, long_arr_short)
    pm = propagationModel('raytracing', 'MaxNumReflections',MaxNumReflections, 'MaxNumDiffractions',0, ...
        'AngularSeparation', 'medium', ...
        'MaxRelativePathLoss',40, 'BuildingsMaterial','concrete', ...
        'TerrainMaterial','loam');

    power_raytrace_arr = zeros(length(long_arr_short), 1);
    parfor iii=1:length(long_arr_short)
        rx = rxsite("Latitude",lat_arr_short(iii), "Longitude",long_arr_short(iii), "AntennaHeight",1.5);
        power_raytrace_arr(iii) = sigstrength(rx,tx,pm);
    end
    power_raytracing = power_raytrace_arr;
end


function [min_rmse, opt_offset] = rmse_and_offset(power_raytracing, power_empirical_arr_short)
    power_empirical_arr_short = power_empirical_arr_short(~isinf(power_raytracing));
    power_raytracing = power_raytracing(~isinf(power_raytracing));

    offset_arr = linspace(-50, 50, 101);
    rmse_arr = zeros(1, length(offset_arr));
    for j=1:length(offset_arr)
        rmse_arr(j) = rmse_empirical_raytracing(power_empirical_arr_short, power_raytracing, offset_arr(j));
    end
    strcat('Length of power_empirical_arr_short is: ', length(power_empirical_arr_short))
    size(rmse_arr)
    [min_rmse, idx] = min(rmse_arr);
    opt_offset = offset_arr(idx);
end