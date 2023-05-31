%% single_ray

%% setting constants
freq = 811 * 10^6;  % Tx freq 
lambda = 3 * 10^8 / freq;  % wavelength
Tx_LAT = 55.784663;
Tx_LONG = 12.523303;

%%
viewer = siteviewer('Buildings', 'dtu.osm');

sectoria = sectorInvertedAmos('ArmLength',0.1.*[0.0880 0.0710 0.0730 0.0650], ...
    'GroundPlaneWidth', 0.15, 'GroundPlaneLength', 1, Tilt=-10, TiltAxis=[0 1 0]);

% pattern(sectoria, freq)
hant_sectoria = design(sectoria, freq);
transmit_power = 4;
tx = txsite("Latitude",Tx_LAT, "Longitude",Tx_LONG, ...
     "TransmitterFrequency",freq, 'AntennaHeight',30, ...
     'Antenna',hant_sectoria, 'AntennaAngle', -45, 'TransmitterPower',transmit_power);

pm = propagationModel('raytracing', 'MaxNumReflections',5, 'MaxNumDiffractions',0, ...
    'AngularSeparation', 'low', ...
    'MaxRelativePathLoss', 45, 'BuildingsMaterialConductivity',0.0275, ...
    'TerrainMaterialConductivity',0.0275);

rx = rxsite("Latitude",55.78559340693077, "Longitude", 12.52258346654631, ...
    "AntennaHeight",1.5);

raytrace(tx,rx,pm);
%%
sigstrength(rx, tx, pm)