%%
dir_name = 'test_dir';
list_dir = dir(strcat(dir_name, '/*.osm'));
bounds_cell = cell(1, length(list_dir));
res = 256;
% lat_arr_outer = zeros(1, res*length(list_dir));
% lon_arr_outer = zeros(1, res*length(list_dir));
% names_list = strings(1, length(list_dir));
% %% generate the lat,lon pairs of data point
% for osmIdx=1:length(list_dir)
%     f_name = strcat(strcat(dir_name, '/'), list_dir(osmIdx).name);
%     f_ptr = fopen(f_name, 'r');
%     for rowIdx=1:2
%         useless = fgetl(f_ptr);  % reads the first two lines that don't have bounds
%     end
%     useful = fgetl(f_ptr);
%     %' <bounds minlat="35.9973800" minlon="-78.9437100" maxlat="36.0037900" maxlon="-78.9309700"/>'
%     [A, n, errmsg] = sscanf(useful, ' <bounds minlat="%f" minlon="%f" maxlat="%f" maxlon="%f"/>');
%     [minlat, minlon, maxlat, maxlon] = deal(A(1), A(2), A(3), A(4));
%     lat_arr_outer((osmIdx - 1)*res+1:osmIdx*res) = linspace(minlat, maxlat, res);
%     lon_arr_outer((osmIdx - 1)*res+1:osmIdx*res) = linspace(minlon, maxlon, res);
%     names_list(osmIdx) = f_name;
%     fclose(f_ptr);
% end
% 
% to_write = struct('lat',lat_arr_outer, 'lon',lon_arr_outer, 'f_name',names_list);
% save("coords.mat", "to_write")

%% setting constants
freq = 811 * 10^6;  % Tx freq 
lambda = 3 * 10^8 / freq;  % wavelength

%% use generated lat,lon pairs to generate the heatmaps

load('coords.mat')

lat_arr_outer = to_write.lat;
lon_arr_outer = to_write.lon;
f_names = to_write.f_name;
power_raytrace_outer = zeros(1, res*res*length(f_names));

sectoria = sectorInvertedAmos('ArmLength',0.5.*[0.0880 0.0710 0.0730 0.0650], ...
    'GroundPlaneWidth', 0.15, 'GroundPlaneLength', 1, Tilt=0, TiltAxis=[0 1 0]);
iii = 0;
hant_sectoria = design(sectoria, freq);
for fileIdx=1:length(f_names)
    viewer = siteviewer('Buildings', f_names(fileIdx));
    Tx_lat = (max(lat_arr_outer((fileIdx - 1)*res+1:fileIdx*res)) + min(lat_arr_outer((fileIdx - 1)*res+1:fileIdx*res)))/2;
    Tx_lon = (max(lon_arr_outer((fileIdx - 1)*res+1:fileIdx*res)) + min(lon_arr_outer((fileIdx - 1)*res+1:fileIdx*res)))/2;
    
    [Latitude, Longitude, TransmitterFrequency, AntennaHeight] = ...
        deal(Tx_lat, Tx_lon, freq, 1.5);

    [Antenna, AntennaAngle, TransmitterPower, MaxNumReflections, AngularSeparation] = ...
        deal(hant_sectoria, -45, 4, 5, 'medium');

    [tx, pm] = set_tx_pm(Latitude, Longitude, TransmitterFrequency, AntennaHeight, ...
                         Antenna, AntennaAngle, TransmitterPower, MaxNumReflections, AngularSeparation);

    for lonIdx=1:res
        power_raytrace_arr = zeros(1, res*res);
        ppm = ParforProgMon('Example',res);
        parfor latIdx=1:res
            rx = rxsite("Latitude",lat_arr_outer((fileIdx-1)*res+lonIdx), "Longitude",lon_arr_outer((fileIdx-1)*res+latIdx), "AntennaHeight",1.5);
            power_raytrace_arr(latIdx) = sigstrength(rx,tx,pm);
        end
        iii = iii + 1
        power_raytrace_outer((fileIdx-1)*res*res+1:fileIdx*res*res) = power_raytrace_arr;
    end
end

function [tx, pm] = set_tx_pm(Latitude, Longitude, TransmitterFrequency, AntennaHeight, ...
     Antenna, AntennaAngle, TransmitterPower, MaxNumReflections, AngularSeparation)

    tx = txsite("Latitude",Latitude, "Longitude",Longitude, ...
     "TransmitterFrequency",TransmitterFrequency, 'AntennaHeight',AntennaHeight, ...
     'Antenna',Antenna, 'AntennaAngle',AntennaAngle, 'TransmitterPower',TransmitterPower);

    pm = propagationModel('raytracing', 'MaxNumReflections',MaxNumReflections, 'MaxNumDiffractions',0, ...
        'AngularSeparation', AngularSeparation, ...
        'MaxRelativePathLoss',40, 'BuildingsMaterial','concrete', ...
        'TerrainMaterial','loam');

end