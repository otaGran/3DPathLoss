freq = 811 * 10^6;  % Tx freq 
lambda = 3 * 10^8 / freq;  % wavelength
Tx_LAT = 55.784663;
Tx_LONG = 12.523303;
%% setting up antenna (half-wave dipole)
n = 1;  % how many half-wavelengths the antenna is
antenna_length = lambda / 2 * n
antenna_width = 0.03;  % default antenna width
dip = dipole('Length',antenna_length, 'Width',antenna_width, 'Tilt',[0, -60], ...
    'TiltAxis',[0 1 0; 0 0 1]);  % design antenna with appropriate tilt
show(dip)
hant = design(dip, freq);


%% setting up Inverted Amos antenna
sectoria = sectorInvertedAmos('ArmLength',0.5.*[0.0880 0.0710 0.0730 0.0650], 'GroundPlaneWidth', 0.15, 'GroundPlaneLength', 1);
pattern(sectoria, freq)
hant_sectoria = design(sectoria, freq);
%% display coverage (half-wave dipole)
viewer = siteviewer('Buildings', 'dtu.osm');


% tx = txsite("Latitude",Tx_LAT, "Longitude",Tx_LONG, ...
%     "TransmitterFrequency",freq, 'AntennaHeight',30, 'Antenna',hant);

tx = txsite("Latitude",Tx_LAT, "Longitude",Tx_LONG, ...
     "TransmitterFrequency",freq, 'AntennaHeight',30, 'Antenna',hant_sectoria, 'AntennaAngle', -45, ...
     'TransmitterPower',5);
show(tx)  % show tx on viewer
pattern(tx)  % show gain of

%%
pm = propagationModel('raytracing', 'MaxNumReflections',2, 'MaxNumDIffractions',1);

pd = coverage(tx,pm,"SignalStrengths",-100:-5,"MaxRange",1000,"Resolution",60);

plot(pd)

%%
save('pd1', 'pd')



