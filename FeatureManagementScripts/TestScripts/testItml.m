direc = '../features/';
%files = dir(strcat(direc,'*.features')); %this is to create a$
%choose better files. or move a few good images to direc
featmat = [];
for file = files'
    %file
    vector = importdata(strcat(direc,file.name),' ',1);
    featmat = [featmat; vector.data];
end;

%run through itml

%use mahal to run through the algo you got

%compare with other


