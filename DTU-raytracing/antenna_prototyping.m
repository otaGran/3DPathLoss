%% antenna, tx and propagation model
freq = 811 * 10^6;  % Tx freq 
lambda = 3 * 10^8 / freq;  % wavelength
Tx_LAT = 55.784663;
Tx_LONG = 12.523303;

sectoria = sectorInvertedAmos('ArmLength',0.5.*[0.0880 0.0710 0.0730 0.0650], ...
    'GroundPlaneWidth', 0.15, 'GroundPlaneLength', 1);

% pattern(sectoria, freq)
hant_sectoria = design(sectoria, freq);

viewer = siteviewer('Buildings', 'dtu.osm');
transmit_power = 4;
tx = txsite("Latitude",Tx_LAT, "Longitude",Tx_LONG, ...
     "TransmitterFrequency",freq, 'AntennaHeight',1.5, ...
     'Antenna',hant_sectoria, 'AntennaAngle',0, 'TransmitterPower',transmit_power);

pattern(tx)
