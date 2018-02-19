load human.txt
load water.txt
human = human'-1;
water = water'-1;

filesave = 'init_data.nc';
len_human = size(human, 2);
len_water = size(water, 2);

nccreate(filesave, 'human', 'Dimensions', {'hrow' 2 'hcol' len_human}, 'Datatype' , 'int32');
nccreate(filesave, 'water', 'Dimensions', {'wrow' 2 'wcol' len_water}, 'Datatype' , 'int32');

ncwrite(filesave, 'human', human);
ncwrite(filesave, 'water', water);