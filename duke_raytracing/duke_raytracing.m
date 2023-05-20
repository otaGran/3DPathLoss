%         36.00379
%-78.94371        -78.93097
%         35.99738
%% raytracing for one point
viewer = siteviewer('Buildings', 'duke.osm');

tx = txsite("Latitude",36.003041790881795, "Longitude",-78.93704655521807, ...
    "TransmitterFrequency",3.6e9, 'AntennaHeight',150);

rx = rxsite("Latitude",36.0015869766464, "Longitude",-78.93973650941552, ...
    "AntennaHeight",100);

pm = propagationModel("raytracing");

pm.Method = "sbr";
pm.MaxNumReflections = 2;
pm.MaxNumDiffractions = 1;
raytrace(tx,rx,pm);

%% display coverage (isometric Tx)
tx = txsite("Latitude",36.003041790881795, "Longitude",-78.93704655521807, ...
    "TransmitterFrequency",3.6e9, 'AntennaHeight',150);

pm = propagationModel('raytracing', 'MaxNumReflections',2, 'MaxNumDIffractions',1);

pd = coverage(tx,pm,"SignalStrengths",-100:-5,"MaxRange",1000,"Resolution",5);
