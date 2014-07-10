cd ../../Desktop/features
load cropdim198.jpg.features
load cropdim189.jpg.features
load cropdim202.jpg.features
load cropdim204.jpg.features

[a,i1] = sort(cropdim189_jpg)
[a,i2] = sort(cropdim198_jpg)
[a,i3] = sort(cropdim202_jpg)
[a,i4] = sort(cropdim204_jpg)

vecs = [cropdim189_jpg;cropdim198_jpg;cropdim202_jpg;cropdim204_jpg]
V = var(vecs)

STDE = std(vecs)

[sortSTD,stind] = sort(STDE)

[sortV,vind] = sort(V)

plot(sortSTD)

load cropdim1.jpg.features
cropdim1_jpg(stind)