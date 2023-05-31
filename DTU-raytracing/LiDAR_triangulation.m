lasReader = lasFileReader("data/PUNKTSKY_1km_6188_720.laz");
ptCloud = readPointCloud(lasReader, Classification=[3,4,5]);
%%
% figure
% pcshow(ptCloud)
%%
figure(5); clf
xx = ptCloud.Location(:, 1);
yy = ptCloud.Location(:, 2);
zz = ptCloud.Location(:, 3);
T = delaunay(xx, yy, zz);
%%
trisurf(T,xx,yy,zz)