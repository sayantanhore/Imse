direc = 'features\';
files = dir(strcat(direc,'*.features')); %this is to create a$
featmat = [];
for file = files'
    file
    vector = importdata(strcat(direc,file.name),' ',1);
    featmat = [featmat; vector.data];
end;


